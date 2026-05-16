"""
SMA grain reconciliation gate: scenario detection, variance, HITL file write.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import pandas as pd

from edvise.genai.mapping.identity_agent.hitl.schemas import InstitutionHITLItems
from edvise.genai.mapping.schema_mapping_agent.grain_resolution.prompt import (
    DedupProposalLLM,
    build_sma_dedup_proposals_without_llm,
    propose_dedup_policy,
    _pick_manifest_suffix_key_column,
)
from .items import build_sma_grain_hitl_items
from edvise.genai.mapping.shared.profiling.variance import (
    WithinGroupVarianceResult,
    compute_within_group_variance,
)

logger = logging.getLogger(__name__)


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
    Write ``sma_grain_hitl.json`` for a grain mismatch.

    Orchestrates scenario detection, optional variance profiling, dedup proposals
    (gateway LLM when IA entity keys differ from manifest keys; heuristics-only when
    they match), HITL item construction, and disk write (same envelope pattern as IA grain HITL).
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

    scenario: Literal["step_down", "within_grain_multiplicity"] = (
        "step_down"
        if ia_source_keys is not None
        and detect_grain_scenario(ia_source_keys, manifest_source_keys) == "step_down"
        else "within_grain_multiplicity"
    )

    aligned_ia_manifest_keys = (
        ia_source_keys is not None
        and set(ia_source_keys) == set(manifest_source_keys)
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

    proposals: list[DedupProposalLLM] | None = None
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
            if aligned_ia_manifest_keys:
                logger.info(
                    "[%s] IA post_clean_primary_key matches manifest entity keys — "
                    "building dedup proposals from variance heuristics (no LLM)",
                    dataset,
                )
                proposals = build_sma_dedup_proposals_without_llm(
                    manifest_source_keys=manifest_source_keys,
                    variance=variance,
                    mapped_source_columns=mapped_source_columns,
                )
            else:
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
                "[%s] Dedup proposal generation failed (%s); using deterministic fallback options",
                dataset,
                e,
            )
            proposals = [
                DedupProposalLLM(
                    strategy="true_duplicate",
                    label="Drop identical duplicates",
                    description="Keep one row per manifest key when rows are full duplicates.",
                    sort_by=None,
                    sort_ascending=None,
                    suffix_column=None,
                    reasoning="Fallback when automated proposal is unavailable.",
                ),
                DedupProposalLLM(
                    strategy="suffix_identifier",
                    label="Suffix key ties",
                    description=(
                        "Within duplicate manifest-key groups, append -1, -2, … to this key "
                        "column’s values; row count unchanged (manifest grain columns unchanged)."
                    ),
                    sort_by=None,
                    sort_ascending=None,
                    suffix_column=_pick_manifest_suffix_key_column(
                        manifest_source_keys
                    ),
                    reasoning="Fallback suffix proposal: column chosen from manifest entity grain.",
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
        aligned_ia_manifest_entity_keys=aligned_ia_manifest_keys
        and scenario == "within_grain_multiplicity",
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
