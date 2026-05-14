"""
Grain-resolution LLM prompt and structured proposals for SMA within-grain multiplicity (Scenario B).

This is the SMA grain prompt path (system + user JSON, gateway completion, parse). For
transformation-map prompts see :mod:`edvise.genai.mapping.schema_mapping_agent.transformation.prompt`.

Strategy strings match :mod:`edvise.genai.mapping.shared.grain.dedup_strategies` (subset
``SmaGrainMultiplicityProposalStrategy``). ``DedupProposalLLM`` is still not
:class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.DedupPolicy`: proposals carry
reviewer-facing ``label`` / ``description`` / ``reasoning`` fields that live outside the persisted
grain contract. Reviewer copy must never imply **widening** the manifest or target grain;
``suffix_identifier`` only rank-suffixes values in an existing **manifest entity key** column.
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

# Grade / credits / points (aligned with Identity Agent grain_inference prompt policy).
_MEASURE_COLUMN_HINTS = re.compile(
    r"(credit|gpa|units?|hours?|score|grade|letter_grade|points?|amount|rate|pct|percent|count)$",
    re.IGNORECASE,
)


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


def _column_matches_measure_hint(column: str) -> bool:
    return bool(_MEASURE_COLUMN_HINTS.search(str(column).strip()))


def _measure_variance_suffix_first_signal(
    column_profiles: list[ColumnVarianceProfile],
    mapped_source_columns: list[str],
    *,
    min_pct: float = 0.25,
    top_k: int = 8,
) -> bool:
    """
    True when duplicate-key variance is driven by grade/credit-like measures.

    Matches Identity Agent grain policy: repeat enrollments differing on measures should lead
    with ``suffix_identifier``, not collapse on grades/credits.
    """
    mapped = set(mapped_source_columns)
    for p in column_profiles[:top_k]:
        if p.column not in mapped:
            continue
        if p.pct_groups_with_variance < min_pct:
            continue
        if _column_matches_measure_hint(p.column):
            return True
    return False


def _high_variance_non_measure_signal(
    column_profiles: list[ColumnVarianceProfile],
    mapped_source_columns: list[str],
) -> bool:
    """Heuristic: high variance on a likely descriptor column suggests distinct entities."""
    for p in column_profiles[:5]:
        if p.pct_groups_with_variance < 0.35:
            continue
        if _column_matches_measure_hint(p.column):
            continue
        if p.column in mapped_source_columns:
            return True
    return False


def _pick_manifest_suffix_key_column(manifest_source_keys: list[str]) -> str:
    """
    Pick a manifest entity key column to receive ranked suffixes.

    ``suffix_column`` for SMA execution must be in ``entity_keys`` / manifest source keys
    (:func:`~edvise.genai.mapping.shared.grain.dedup_execution.apply_sma_grain_resolution_payload`).
    Prefer course/offering-like keys when names allow.
    """
    keys = [str(k).strip() for k in manifest_source_keys if str(k).strip()]
    if not keys:
        raise ValueError("manifest_source_keys must be non-empty for SMA suffix pick")
    course_like = re.compile(
        r"(course|section|crn|catalog|class|offering|instruction|comp|subject)",
        re.IGNORECASE,
    )
    for k in keys:
        if course_like.search(k):
            return k
    if len(keys) == 1:
        return keys[0]
    return keys[-1]


def _alternate_manifest_suffix_key(
    manifest_source_keys: list[str], primary: str
) -> str:
    keys = [str(k).strip() for k in manifest_source_keys if str(k).strip()]
    p = str(primary).strip()
    for k in keys:
        if k != p:
            return k
    return p


def _default_true_duplicate_proposal() -> DedupProposalLLM:
    return DedupProposalLLM(
        strategy="true_duplicate",
        label="Drop identical duplicates",
        description=(
            "When duplicate key groups are full-row duplicates on mapped columns, keep one row "
            "per manifest key tuple."
        ),
        sort_by=None,
        sort_ascending=None,
        suffix_column=None,
        reasoning="Use only when variance is not driven by measures you need to retain.",
    )


def _default_suffix_proposal_measure_policy(
    manifest_source_keys: list[str],
) -> DedupProposalLLM:
    sk = _pick_manifest_suffix_key_column(manifest_source_keys)
    return DedupProposalLLM(
        strategy="suffix_identifier",
        label="Suffix repeat enrollments",
        description=(
            "Duplicate manifest keys differ on grades, credits, GPA, or similar measures; keep "
            "every row and append -1, -2, … to values in the chosen manifest key column "
            f"({sk!r}), matching Identity Agent grain policy — do not collapse on measures."
        ),
        sort_by=None,
        sort_ascending=None,
        suffix_column=sk,
        reasoning=(
            "Top variance is on measure-like columns within duplicate key groups; "
            "suffix_identifier on a manifest entity key preserves all enrollments."
        ),
    )


def _default_suffix_proposal_descriptive_policy(suffix_column: str) -> DedupProposalLLM:
    return DedupProposalLLM(
        strategy="suffix_identifier",
        label="Suffix key ties",
        description=(
            "Within duplicate key groups, append -1, -2, … to this manifest key column’s "
            "values so rows stay distinct without dropping any rows."
        ),
        sort_by=None,
        sort_ascending=None,
        suffix_column=suffix_column,
        reasoning=(
            "High within-group variance on non-measure columns suggests genuinely distinct rows "
            "rather than identical duplicates."
        ),
    )


def _normalize_proposals_after_llm(
    props: list[DedupProposalLLM],
    *,
    manifest_source_keys: list[str],
    measure_variance_suffix_first: bool,
    suffix_second_required: bool,
) -> list[DedupProposalLLM]:
    """Enforce IA-aligned defaults the LLM may omit or mis-order."""
    out = list(props)
    if len(out) < 2:
        return out

    if measure_variance_suffix_first:
        p0 = out[0]
        sk0 = str(p0.suffix_column or "").strip()
        if (
            p0.strategy != "suffix_identifier"
            or sk0 not in manifest_source_keys
            or not sk0
        ):
            out[0] = _default_suffix_proposal_measure_policy(manifest_source_keys)
        if (
            out[1].strategy == "suffix_identifier"
            and str(out[1].suffix_column or "").strip()
            == str(out[0].suffix_column or "").strip()
        ):
            out[1] = _default_true_duplicate_proposal()

    if suffix_second_required and not measure_variance_suffix_first:
        sk = _pick_manifest_suffix_key_column(manifest_source_keys)
        p1 = out[1]
        sk1 = str(p1.suffix_column or "").strip()
        if p1.strategy != "suffix_identifier" or sk1 not in manifest_source_keys:
            out[1] = _default_suffix_proposal_descriptive_policy(sk)

    keys_seq = [str(k).strip() for k in manifest_source_keys if str(k).strip()]
    key_set = set(keys_seq)
    for i, p in enumerate(out):
        if p.strategy != "suffix_identifier":
            continue
        sk = str(p.suffix_column or "").strip()
        if sk in key_set:
            continue
        replacement = _pick_manifest_suffix_key_column(manifest_source_keys)
        logger.warning(
            "SMA dedup proposal %d: suffix_column %r not in manifest entity grain %s — "
            "coercing to %r",
            i,
            sk,
            keys_seq,
            replacement,
        )
        out[i] = p.model_copy(update={"suffix_column": replacement})

    return out


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
    measure_variance_suffix_first = _measure_variance_suffix_first_signal(
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
        "measure_variance_suffix_first": measure_variance_suffix_first,
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
        "- suffix_column: required for suffix_identifier — **must be one of manifest_source_keys** "
        "(an entity / manifest grain column in the base table, not a non-key mapped measure column).\n"
        "- reasoning: 1-2 sentences grounded in the variance profile\n"
        "Rules:\n"
        "- Never describe any proposal as widening, expanding, or adding columns to the manifest "
        "or target grain; that is not on offer.\n"
        "- temporal_collapse: use for clear time / sequence ordering.\n"
        "- first_by_column: only when sort order on sort_by **is** the policy (e.g. numeric "
        "sequence, line id, boolean primary flag) — not alphabetical tie-breaks on labels; "
        "not sort_by on grade/GPA/credits (prefer suffix_identifier when measures differ).\n"
        "- true_duplicate: only when duplicates look identical across mapped columns.\n"
        "- suffix_identifier: rows may be distinct entities sharing the same key tuple; keep every "
        "row and break ties by appending -1, -2, ... to string values in suffix_column. "
        "suffix_column must be one of **manifest_source_keys** (same as execution entity keys).\n"
        "If measure_variance_suffix_first is true in the user payload, the **FIRST** proposal must "
        'use strategy "suffix_identifier": duplicate keys differ on grades, credits, GPA, or '
        "similar measures — match Identity Agent grain policy (do not collapse or prioritize "
        "measures). Pick suffix_column from manifest_source_keys (typically a course/offering "
        "identifier key). The second proposal may be true_duplicate or a non-measure sort, but "
        "must not use first_by_column or temporal_collapse **on** grade/GPA/credit columns.\n"
        "If suffix_second_required is true (and measure_variance_suffix_first is false), the "
        'SECOND proposal must use strategy "suffix_identifier" with suffix_column chosen from '
        "manifest_source_keys (same constraint).\n"
        "If both flags are true, still put measure-driven suffix_identifier **first**; use a "
        "different manifest key or true_duplicate for the second slot if needed."
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
    return _normalize_proposals_after_llm(
        props,
        manifest_source_keys=list(manifest_source_keys),
        measure_variance_suffix_first=bool(measure_variance_suffix_first),
        suffix_second_required=bool(user_payload.get("suffix_second_required")),
    )


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
