"""
Grain-resolution LLM prompt and structured proposals for SMA within-grain multiplicity (Scenario B).

This is the SMA grain prompt path (system + user JSON, gateway completion, parse). For
transformation-map prompts see :mod:`edvise.genai.mapping.schema_mapping_agent.transformation.prompt`.

Strategy strings match :mod:`edvise.genai.mapping.shared.grain.dedup_strategies` (subset
``SmaGrainMultiplicityProposalStrategy``). ``DedupProposalLLM`` is still not
:class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.DedupPolicy`: proposals carry
reviewer-facing ``label`` / ``description`` / ``reasoning`` fields that live outside the persisted
grain contract.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from edvise.genai.mapping.identity_agent.hitl.schemas import GrainResolution
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    create_openai_client_for_databricks_gateway,
    make_databricks_gateway_llm_complete,
)
from edvise.genai.mapping.shared.grain.dedup_strategies import SmaGrainMultiplicityProposalStrategy
from edvise.genai.mapping.shared.profiling.variance import (
    ColumnVarianceProfile,
    WithinGroupVarianceResult,
)
from edvise.genai.mapping.shared.strip_json_fences import strip_json_fences
from edvise.utils.llm_utils import llm_complete_with_parse_retry

logger = logging.getLogger(__name__)


class DedupProposalLLM(BaseModel):
    strategy: SmaGrainMultiplicityProposalStrategy
    label: str = Field(..., max_length=80)
    description: str
    sort_by: str | None = None
    sort_ascending: bool | None = None
    suffix_column: str | None = None
    reasoning: str


class _DedupProposalsResponse(BaseModel):
    proposals: list[DedupProposalLLM] = Field(..., min_length=2, max_length=2)


def _parse_dedup_proposals_response(raw: str) -> _DedupProposalsResponse:
    stripped = strip_json_fences(raw.strip())
    data = json.loads(stripped)
    return _DedupProposalsResponse.model_validate(data)


def _high_variance_non_measure_signal(
    column_profiles: list[ColumnVarianceProfile],
    mapped_source_columns: list[str],
) -> bool:
    """Heuristic: high variance on a likely descriptor column suggests distinct entities."""
    measure_hints = re.compile(
        r"(credit|gpa|units?|hours?|score|grade|amount|rate|pct|percent|count)$",
        re.IGNORECASE,
    )
    for p in column_profiles[:5]:
        if p.pct_groups_with_variance < 0.35:
            continue
        if measure_hints.search(p.column):
            continue
        if p.column in mapped_source_columns:
            return True
    return False


def propose_dedup_policy(
    *,
    institution_id: str,
    dataset: str,
    manifest_source_keys: list[str],
    ia_source_keys: list[str] | None,
    variance: WithinGroupVarianceResult,
    mapped_source_columns: list[str],
) -> list[DedupProposalLLM]:
    """Gateway LLM (default model + max_tokens from ``databricks_ai_gateway``) returning two proposals.

    Uses :func:`~edvise.utils.llm_utils.llm_complete_with_parse_retry` so malformed JSON or
    Pydantic validation errors trigger re-prompts (same defaults as ``llm_utils``).
    """
    top_profiles = variance.column_profiles[:8]
    high_suffix_signal = _high_variance_non_measure_signal(
        variance.column_profiles, mapped_source_columns
    )

    user_payload: dict[str, Any] = {
        "institution_id": institution_id,
        "dataset": dataset,
        "manifest_source_keys": manifest_source_keys,
        "ia_source_keys": ia_source_keys or [],
        "non_unique_rows": variance.non_unique_rows,
        "affected_groups": variance.affected_groups,
        "group_size_distribution": variance.group_size_distribution,
        "top_column_profiles": [
            {
                "column": p.column,
                "pct_groups_with_variance": p.pct_groups_with_variance,
                "sample_values": p.sample_values,
            }
            for p in top_profiles
        ],
        "mapped_source_columns": mapped_source_columns,
        "suffix_second_required": high_suffix_signal,
    }

    system = (
        "You are a data integration assistant. Output a single JSON object only, no markdown, "
        "no code fences. The JSON must match this shape exactly:\n"
        '{"proposals": [<proposal>, <proposal>]}\n'
        "There must be exactly 2 proposals.\n"
        "Each proposal fields:\n"
        '- strategy: one of "true_duplicate", "temporal_collapse", "first_by_column", '
        '"suffix_identifier"\n'
        "- label: ~4 words\n"
        "- description: one sentence for a human reviewer\n"
        "- sort_by: source column name (required for temporal_collapse and first_by_column)\n"
        "- sort_ascending: bool (required for temporal_collapse and first_by_column)\n"
        "- suffix_column: source column name (required for suffix_identifier)\n"
        "- reasoning: 1-2 sentences grounded in the variance profile\n"
        "Rules:\n"
        "- temporal_collapse: use for clear time / sequence ordering.\n"
        "- first_by_column: only when sort order on sort_by **is** the policy (e.g. numeric "
        "sequence, line id, boolean primary flag) — not alphabetical tie-breaks on labels; "
        "not sort_by on grade/GPA/credits (prefer suffix_identifier when measures differ).\n"
        "- true_duplicate: only when duplicates look identical across mapped columns.\n"
        "- suffix_identifier: rows may be distinct entities; widen the key with suffix_column.\n"
        "If suffix_second_required is true in the user payload, the SECOND proposal must use "
        'strategy "suffix_identifier" with a high-variance descriptive column when possible.'
    )

    user = json.dumps(user_payload, indent=2)

    client = create_openai_client_for_databricks_gateway()
    complete = make_databricks_gateway_llm_complete(client)
    validated = llm_complete_with_parse_retry(
        complete,
        system,
        user,
        _parse_dedup_proposals_response,
        logger=logger,
    )
    props = list(validated.proposals)
    if user_payload.get("suffix_second_required") and props[1].strategy != "suffix_identifier":
        for p in variance.column_profiles:
            if p.column in mapped_source_columns and p.pct_groups_with_variance >= 0.25:
                props[1] = DedupProposalLLM(
                    strategy="suffix_identifier",
                    label="Widen entity grain",
                    description=(
                        "Append this column to the manifest entity key so distinct rows "
                        "remain separate entities."
                    ),
                    sort_by=None,
                    sort_ascending=None,
                    suffix_column=p.column,
                    reasoning=(
                        "High within-group variance on this column suggests genuinely distinct "
                        "rows rather than duplicates."
                    ),
                )
                break
    return props


def proposal_to_grain_resolution(p: DedupProposalLLM) -> GrainResolution:
    if p.strategy == "true_duplicate":
        return GrainResolution(dedup_strategy="true_duplicate")
    if p.strategy in ("temporal_collapse", "first_by_column"):
        if not p.sort_by:
            raise ValueError(f"{p.strategy} requires sort_by")
        if p.sort_ascending is None:
            raise ValueError(f"{p.strategy} requires sort_ascending")
        return GrainResolution(
            dedup_strategy=p.strategy,
            dedup_sort_by=p.sort_by,
            dedup_sort_ascending=p.sort_ascending,
            dedup_keep="first",
        )
    if p.strategy == "suffix_identifier":
        if not p.suffix_column:
            raise ValueError("suffix_identifier requires suffix_column")
        return GrainResolution(
            dedup_strategy="suffix_identifier",
            suffix_column=p.suffix_column,
        )
    raise ValueError(f"Unknown strategy {p.strategy!r}")
