"""Post-LLM grain contract verification: key profile, dedup impact, policy coercion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.shared.profiling.grain_key_profile import (
    DedupImpact,
    KeyProfileSummary,
    categorical_variance_columns,
    prepare_profiling_frame,
    profile_key,
    simulate_dedup_impact,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class GrainContractVerification:
    contract_key_profile: KeyProfileSummary | None
    dedup_impact: DedupImpact | None
    coerced: bool
    coercion_reason: str | None
    categorical_variance_columns: list[str]
    error: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "contract_key_profile": (
                self.contract_key_profile.to_jsonable()
                if self.contract_key_profile is not None
                else None
            ),
            "dedup_impact": (
                self.dedup_impact.to_jsonable()
                if self.dedup_impact is not None
                else None
            ),
            "coerced": self.coerced,
            "coercion_reason": self.coercion_reason,
            "categorical_variance_columns": self.categorical_variance_columns,
            "error": self.error,
        }


def _coerce_true_duplicate_to_policy_required(
    contract: GrainContract,
    *,
    reason: str,
) -> GrainContract:
    notes = contract.dedup_policy.notes.strip()
    extra = f"Coerced from true_duplicate: {reason}"
    merged_notes = f"{notes} {extra}".strip() if notes else extra
    return contract.model_copy(
        update={
            "dedup_policy": contract.dedup_policy.model_copy(
                update={
                    "strategy": "policy_required",
                    "notes": merged_notes,
                }
            ),
            "hitl_flag": True,
        }
    )


def verify_grain_contract(
    contract: GrainContract,
    df: pd.DataFrame,
) -> tuple[GrainContract, GrainContractVerification]:
    """
    Profile the contract key, simulate dedup impact, and coerce ``true_duplicate`` to
    ``policy_required`` when duplicate groups vary on non-measure columns.
    """
    from edvise.genai.mapping.identity_agent.execution.contract_utilities import (
        canonicalize_grain_contract_learner_id_alias,
    )

    prepared = prepare_profiling_frame(df)
    canonical_learner = (
        "learner_id" if "learner_id" in prepared.columns else "student_id"
    )
    profile_contract = canonicalize_grain_contract_learner_id_alias(
        contract,
        canonical_column=canonical_learner,
    )
    key_cols = list(profile_contract.post_clean_primary_key)

    try:
        key_profile = profile_key(prepared, key_cols)
    except ValueError as e:
        _LOG.warning(
            "[ia grain verify] table=%s cannot profile contract key: %s",
            contract.table,
            e,
        )
        return contract, GrainContractVerification(
            contract_key_profile=None,
            dedup_impact=None,
            coerced=False,
            coercion_reason=None,
            categorical_variance_columns=[],
            error=str(e),
        )

    cat_var_cols = categorical_variance_columns(key_profile)
    work_contract = contract
    coerced = False
    coercion_reason: str | None = None

    if (
        contract.dedup_policy.strategy == "true_duplicate"
        and key_profile.affected_groups > 0
        and cat_var_cols
    ):
        coercion_reason = (
            f"true_duplicate invalid: {len(cat_var_cols)} non-measure column(s) vary "
            f"within contract key groups: {cat_var_cols[:5]!r}"
        )
        work_contract = _coerce_true_duplicate_to_policy_required(
            contract, reason=coercion_reason
        )
        coerced = True
        _LOG.warning(
            "[ia grain verify] table=%s coerced dedup_policy to policy_required — %s",
            contract.table,
            coercion_reason,
        )

    try:
        dedup_impact = simulate_dedup_impact(prepared, work_contract, prepared=True)
    except (ValueError, KeyError) as e:
        _LOG.warning(
            "[ia grain verify] table=%s dedup simulation failed: %s",
            contract.table,
            e,
        )
        return work_contract, GrainContractVerification(
            contract_key_profile=key_profile,
            dedup_impact=None,
            coerced=coerced,
            coercion_reason=coercion_reason,
            categorical_variance_columns=cat_var_cols,
            error=str(e),
        )

    _LOG.info(
        "[ia grain verify] table=%s key=%s strategy=%s rows %d -> %d (dropped %d) "
        "affected_groups=%d categorical_variance_cols=%d",
        contract.table,
        key_cols,
        work_contract.dedup_policy.strategy,
        dedup_impact.rows_before,
        dedup_impact.rows_after,
        dedup_impact.rows_dropped,
        key_profile.affected_groups,
        len(cat_var_cols),
    )

    return work_contract, GrainContractVerification(
        contract_key_profile=key_profile,
        dedup_impact=dedup_impact,
        coerced=coerced,
        coercion_reason=coercion_reason,
        categorical_variance_columns=cat_var_cols,
    )


__all__ = ["GrainContractVerification", "verify_grain_contract"]
