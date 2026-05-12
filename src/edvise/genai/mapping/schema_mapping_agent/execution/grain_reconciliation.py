"""
SMA grain reconciliation: detect manifest vs. row-level grain mismatch, profile, propose dedup.

Two scenarios (see :func:`run_grain_reconciliation_gate`):

**Step-down (Scenario A)** — IdentityAgent ``post_clean_primary_key`` (in source space) is a strict
superset of the manifest-resolved entity keys. The file is intentionally finer-grained than the
manifest target (e.g. student-term rows mapping to student-level targets). This is a sanctioned
collapse: emit a confirmation HITL item and halt until acknowledged (non-blocking severity).

**Within-grain multiplicity (Scenario B)** — IA keys are not a strict superset of manifest keys,
yet rows still do not reduce cleanly to one row per manifest entity key. Variance is profiled on
mapped source columns only, an LLM proposes two dedup strategies, and a blocking HITL item is
written until a reviewer selects an option.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    InstitutionHITLItems,
    ReentryDepth,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    create_openai_client_for_databricks_gateway,
    make_databricks_gateway_llm_complete,
)
from edvise.genai.mapping.shared.profiling.variance import (
    ColumnVarianceProfile,
    WithinGroupVarianceResult,
    compute_within_group_variance,
)

logger = logging.getLogger(__name__)

_SMA_GRAIN_LLM_MODEL = "claude-sonnet-4-20250514"
_SMA_GRAIN_LLM_MAX_TOKENS = 1000


def detect_grain_scenario(
    ia_source_keys: list[str],
    manifest_source_keys: list[str],
) -> Literal["step_down", "within_grain_multiplicity"]:
    """Classify mismatch: sanctioned step-down vs. unexplained within-grain multiplicity."""
    ia_set = set(ia_source_keys)
    man_set = set(manifest_source_keys)
    if ia_set.issuperset(man_set) and ia_set != man_set:
        return "step_down"
    return "within_grain_multiplicity"


def profile_against_manifest_grain(
    df: pd.DataFrame,
    manifest_source_keys: list[str],
    mapped_source_columns: list[str],
) -> WithinGroupVarianceResult:
    """Variance on duplicate manifest-key groups, scoped to mapped (non-key) source columns."""
    return compute_within_group_variance(
        df, manifest_source_keys, profile_cols=mapped_source_columns
    )


class _DedupProposalLLM(BaseModel):
    strategy: Literal[
        "true_duplicate",
        "temporal_collapse",
        "first_by_column",
        "suffix",
    ]
    label: str = Field(..., max_length=80)
    description: str
    sort_by: str | None = None
    sort_ascending: bool | None = None
    suffix_column: str | None = None
    reasoning: str


class _DedupProposalsResponse(BaseModel):
    proposals: list[_DedupProposalLLM] = Field(..., min_length=2, max_length=2)


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
) -> list[_DedupProposalLLM]:
    """Single Sonnet call returning exactly two ranked dedup proposals (Scenario B)."""
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
        '- strategy: one of "true_duplicate", "temporal_collapse", "first_by_column", "suffix"\n'
        '- label: ~4 words\n'
        '- description: one sentence for a human reviewer\n'
        '- sort_by: source column name (required for temporal_collapse and first_by_column)\n'
        '- sort_ascending: bool (required for temporal_collapse and first_by_column)\n'
        '- suffix_column: source column name (required for suffix)\n'
        '- reasoning: 1-2 sentences grounded in the variance profile\n'
        "Rules:\n"
        "- temporal_collapse: use for clear time / sequence ordering.\n"
        "- first_by_column: use for non-time ordering (e.g. prefer non-null grade).\n"
        "- true_duplicate: only when duplicates look identical across mapped columns.\n"
        "- suffix: rows may be distinct entities; widen the key with suffix_column.\n"
        "If suffix_second_required is true in the user payload, the SECOND proposal must use "
        'strategy "suffix" with a high-variance descriptive column when possible.'
    )

    user = json.dumps(user_payload, indent=2)

    client = create_openai_client_for_databricks_gateway()
    complete = make_databricks_gateway_llm_complete(
        client, model=_SMA_GRAIN_LLM_MODEL, max_tokens=_SMA_GRAIN_LLM_MAX_TOKENS
    )
    raw = complete(system, user).strip()
    raw = _strip_json_fences(raw)
    parsed = json.loads(raw)
    validated = _DedupProposalsResponse.model_validate(parsed)
    props = list(validated.proposals)
    if user_payload.get("suffix_second_required") and props[1].strategy != "suffix":
        # Best-effort repair: swap or replace second with suffix if we have a candidate column
        for p in variance.column_profiles:
            if p.column in mapped_source_columns and p.pct_groups_with_variance >= 0.25:
                props[1] = _DedupProposalLLM(
                    strategy="suffix",
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


def _strip_json_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _llm_proposal_to_grain_resolution(p: _DedupProposalLLM) -> GrainResolution:
    if p.strategy == "true_duplicate":
        return GrainResolution(dedup_strategy="true_duplicate")
    if p.strategy in ("temporal_collapse", "first_by_column"):
        if not p.sort_by:
            raise ValueError(f"{p.strategy} requires sort_by")
        if p.sort_ascending is None:
            raise ValueError(f"{p.strategy} requires sort_ascending")
        return GrainResolution(
            dedup_strategy="temporal_collapse",
            dedup_sort_by=p.sort_by,
            dedup_sort_ascending=p.sort_ascending,
            dedup_keep="first",
        )
    if p.strategy == "suffix":
        if not p.suffix_column:
            raise ValueError("suffix requires suffix_column")
        return GrainResolution(
            dedup_strategy="suffix_identifier",
            suffix_column=p.suffix_column,
        )
    raise ValueError(f"Unknown strategy {p.strategy!r}")


def build_sma_grain_hitl_items(
    *,
    institution_id: str,
    dataset: str,
    entity_type: Literal["cohort", "course"],
    scenario: Literal["step_down", "within_grain_multiplicity"],
    base_rows: int,
    entity_rows: int,
    manifest_source_keys: list[str],
    mapped_source_columns: list[str],
    ia_source_keys: list[str] | None,
    proposals: list[_DedupProposalLLM] | None,
    sma_manifest_path: Path | None,
    variance: WithinGroupVarianceResult | None = None,
) -> list[HITLItem]:
    """Construct HITL items for ``sma_grain_hitl.json``."""
    delta = base_rows - entity_rows
    meta_base: dict[str, Any] = {
        "base_rows": base_rows,
        "entity_rows": entity_rows,
        "delta_rows": delta,
        "scenario": scenario,
        "dataset": dataset,
        "ia_source_keys": list(ia_source_keys or []),
        "manifest_source_keys": list(manifest_source_keys),
        "entity_type": entity_type,
    }
    if sma_manifest_path is not None:
        meta_base["sma_manifest_path"] = str(sma_manifest_path)

    target = HITLTarget(
        institution_id=institution_id,
        table=dataset,
        config="sma_execution_grain",
        field="base_df_reduction",
    )

    if scenario == "step_down":
        q = (
            f"Dataset '{dataset}': manifest entity grain ({manifest_source_keys}) is coarser than "
            f"IdentityAgent post-clean keys ({ia_source_keys or []}). "
            f"Multiple base rows ({base_rows}) map to {entity_rows} entities — confirm this "
            "intentional step-down (sanctioned collapse) or flag as unexpected."
        )
        return [
            HITLItem(
                item_id=f"{institution_id}_sma_grain_step_down_{dataset}_{uuid.uuid4().hex[:8]}",
                institution_id=institution_id,
                table=dataset,
                domain=HITLDomain.SMA_GRAIN,
                hitl_question=q,
                hitl_context=None,
                options=[
                    HITLOption(
                        option_id="confirm_step_down",
                        label="Confirm intentional step-down",
                        description=(
                            "Collapse finer source rows to manifest entity grain; "
                            "no manifest change."
                        ),
                        resolution=GrainResolution(
                            dedup_strategy="intentional_step_down"
                        ).model_dump(mode="json"),
                        reentry=ReentryDepth.TERMINAL,
                    ),
                    HITLOption(
                        option_id="custom",
                        label="Unexpected — needs follow-up",
                        description="Escalate: this collapse was not expected for this dataset.",
                        resolution=None,
                        reentry=ReentryDepth.TERMINAL,
                    ),
                ],
                target=target,
                choice=None,
                severity="warning",
                metadata=meta_base,
            )
        ]

    ctx_json: str | None = None
    if variance is not None:
        ctx_json = json.dumps(
            {
                "top_column_profiles": [
                    {
                        "column": p.column,
                        "pct_groups_with_variance": p.pct_groups_with_variance,
                        "sample_values": p.sample_values,
                    }
                    for p in variance.column_profiles[:12]
                ],
                "group_size_distribution": variance.group_size_distribution,
                "sampled": variance.sampled,
            },
            indent=2,
        )

    assert proposals is not None and len(proposals) >= 2
    q = (
        f"Dataset '{dataset}': rows do not reduce cleanly to manifest keys {manifest_source_keys} "
        f"({base_rows} base rows vs {entity_rows} unique key groups). "
        "Pick a dedup / grain strategy informed by the variance profile."
    )
    opt_models: list[HITLOption] = []
    for i, prop in enumerate(proposals[:2]):
        opt_models.append(
            HITLOption(
                option_id=f"proposal_{i + 1}_{prop.strategy}",
                label=prop.label[:80],
                description=prop.description,
                resolution=_llm_proposal_to_grain_resolution(prop).model_dump(mode="json"),
                reentry=ReentryDepth.TERMINAL,
            )
        )
    opt_models.append(
        HITLOption(
            option_id="custom",
            label="Custom handling",
            description="None of the proposals fit; follow up manually.",
            resolution=None,
            reentry=ReentryDepth.TERMINAL,
        )
    )
    return [
        HITLItem(
            item_id=f"{institution_id}_sma_grain_multiplicity_{dataset}_{uuid.uuid4().hex[:8]}",
            institution_id=institution_id,
            table=dataset,
            domain=HITLDomain.SMA_GRAIN,
            hitl_question=q,
            hitl_context=ctx_json,
            options=opt_models,
            target=target,
            choice=None,
            severity="error",
            metadata=meta_base,
        )
    ]


def run_grain_reconciliation_gate(
    df: pd.DataFrame,
    institution_id: str,
    dataset: str,
    entity_type: Literal["cohort", "course"],
    manifest_source_keys: list[str],
    mapped_source_columns: list[str],
    ia_source_keys: list[str] | None,
    hitl_output_path: Path,
    *,
    sma_manifest_path: Path | None = None,
) -> None:
    """
    Write ``sma_grain_hitl.json`` for a grain mismatch (see module docstring).

    Orchestrates scenario detection, optional variance profiling + LLM proposals,
    HITL item construction, and disk write (same envelope pattern as IA grain HITL).
    """
    base_rows = len(df)
    entity_rows = df.drop_duplicates(subset=manifest_source_keys).shape[0]
    if entity_rows >= base_rows:
        logger.info(
            "[%s] run_grain_reconciliation_gate: no mismatch (base_rows=%d entity_rows=%d) — skipping",
            dataset,
            base_rows,
            entity_rows,
        )
        return

    scenario = (
        "step_down"
        if ia_source_keys is not None
        and detect_grain_scenario(ia_source_keys, manifest_source_keys) == "step_down"
        else "within_grain_multiplicity"
    )

    logger.info(
        "[%s] SMA grain gate: institution=%s scenario=%s base_rows=%d entity_rows=%d keys=%s",
        dataset,
        institution_id,
        scenario,
        base_rows,
        entity_rows,
        manifest_source_keys,
    )

    proposals: list[_DedupProposalLLM] | None = None
    variance: WithinGroupVarianceResult | None = None
    if scenario == "within_grain_multiplicity":
        variance = profile_against_manifest_grain(
            df, manifest_source_keys, mapped_source_columns
        )
        if variance.non_unique_rows == 0:
            logger.warning(
                "[%s] Expected within-grain multiplicity but variance profile shows no "
                "non-unique rows — treating as within_grain_multiplicity anyway",
                dataset,
            )
        try:
            proposals = propose_dedup_policy(
                institution_id=institution_id,
                dataset=dataset,
                manifest_source_keys=manifest_source_keys,
                ia_source_keys=ia_source_keys,
                variance=variance,
                mapped_source_columns=mapped_source_columns,
            )
        except Exception as e:
            logger.warning(
                "[%s] LLM dedup proposal failed (%s); using deterministic fallback options",
                dataset,
                e,
            )
            proposals = [
                _DedupProposalLLM(
                    strategy="true_duplicate",
                    label="Drop identical duplicates",
                    description="Keep one row per manifest key when rows are full duplicates.",
                    sort_by=None,
                    sort_ascending=None,
                    suffix_column=None,
                    reasoning="Fallback when automated proposal is unavailable.",
                ),
                _DedupProposalLLM(
                    strategy="suffix",
                    label="Widen manifest grain",
                    description="Append a disambiguating source column to entity keys in the manifest.",
                    sort_by=None,
                    sort_ascending=None,
                    suffix_column=(
                        variance.column_profiles[0].column
                        if variance.column_profiles
                        else mapped_source_columns[0]
                    ),
                    reasoning="Fallback suffix proposal from top variance column.",
                ),
            ]

    items = build_sma_grain_hitl_items(
        institution_id=institution_id,
        dataset=dataset,
        entity_type=entity_type,
        scenario=scenario,
        base_rows=base_rows,
        entity_rows=entity_rows,
        manifest_source_keys=manifest_source_keys,
        mapped_source_columns=mapped_source_columns,
        ia_source_keys=ia_source_keys,
        proposals=proposals,
        sma_manifest_path=sma_manifest_path,
        variance=variance,
    )

    hitl_output_path = Path(hitl_output_path)
    hitl_output_path.parent.mkdir(parents=True, exist_ok=True)
    env = InstitutionHITLItems(
        institution_id=institution_id,
        domain="sma_grain",
        items=items,
    )
    hitl_output_path.write_text(env.model_dump_json(indent=2))
    logger.info(
        "[%s] Wrote SMA grain HITL (%d item(s)) -> %s",
        dataset,
        len(items),
        hitl_output_path,
    )
