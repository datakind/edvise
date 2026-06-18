"""Tests for grain key profiling, dedup impact simulation, and contract verification."""

import json

import pandas as pd
import pytest

from edvise.genai.mapping.identity_agent.grain_inference.contract_validation import (
    verify_grain_contract,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
)
from edvise.genai.mapping.shared.profiling.grain_key_profile import (
    categorical_variance_columns,
    prepare_profiling_frame,
    profile_key,
    simulate_dedup_impact,
)


def _iit_shaped_student_frame(
    *, n_students: int = 100, rows_per_student: int = 3
) -> pd.DataFrame:
    """Multiple rows per (student_id, program_at_graduation) with differing major."""
    rows: list[dict[str, object]] = []
    majors = ["Psychology", "Business", "Criminal Justice"]
    for sid in range(1, n_students + 1):
        for i in range(rows_per_student):
            rows.append(
                {
                    "student_id": f"S{sid:04d}",
                    "program_at_graduation": "BS",
                    "major_at_graduation": majors[i % len(majors)],
                    "total_credits_attempted": 90 + i,
                    "student_type": "First-time",
                }
            )
    return pd.DataFrame(rows)


def _bad_iit_contract() -> GrainContract:
    return GrainContract(
        institution_id="iit",
        table="student",
        learner_id_alias=None,
        post_clean_primary_key=["student_id", "program_at_graduation"],
        dedup_policy=DedupPolicy(strategy="true_duplicate", notes=""),
        row_selection_required=False,
        join_keys_for_2a=["student_id"],
        confidence=0.85,
        hitl_flag=True,
        reasoning="test",
    )


def test_profile_key_reports_duplicate_groups_on_iit_shaped_key() -> None:
    df = prepare_profiling_frame(_iit_shaped_student_frame())
    profile = profile_key(df, ["student_id", "program_at_graduation"])
    assert profile.affected_groups == 100
    assert profile.non_unique_rows == 300
    assert profile.uniqueness_score == pytest.approx(0.3333, abs=0.0001)
    assert "major_at_graduation" in categorical_variance_columns(profile)


def test_simulate_dedup_impact_true_duplicate_drops_most_rows() -> None:
    df = _iit_shaped_student_frame()
    contract = _bad_iit_contract()
    impact = simulate_dedup_impact(df, contract)
    assert impact.rows_before == 300
    assert impact.rows_after == 100
    assert impact.rows_dropped == 200
    assert impact.strategy_applied == "true_duplicate"


def test_verify_grain_contract_coerces_true_duplicate_with_categorical_variance() -> (
    None
):
    df = _iit_shaped_student_frame()
    contract = _bad_iit_contract()
    coerced, verification = verify_grain_contract(contract, df)

    assert verification.coerced is True
    assert verification.coercion_reason is not None
    assert "major_at_graduation" in verification.categorical_variance_columns
    assert coerced.dedup_policy.strategy == "policy_required"
    assert coerced.hitl_flag is True
    assert verification.dedup_impact is not None
    assert verification.dedup_impact.rows_dropped == 0
    json.dumps(verification.to_jsonable())


def test_verify_grain_contract_leaves_true_duplicate_when_only_measures_vary() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["a", "a", "b"],
            "program_at_graduation": ["BS", "BS", "BA"],
            "total_credits_attempted": [90, 95, 60],
        }
    )
    contract = GrainContract(
        institution_id="x",
        table="student",
        learner_id_alias=None,
        post_clean_primary_key=["student_id", "program_at_graduation"],
        dedup_policy=DedupPolicy(strategy="true_duplicate", notes=""),
        row_selection_required=False,
        join_keys_for_2a=["student_id"],
        confidence=0.9,
        hitl_flag=False,
        reasoning="measure-only variance",
    )
    out, verification = verify_grain_contract(contract, df)
    assert verification.coerced is False
    assert out.dedup_policy.strategy == "true_duplicate"
    assert verification.dedup_impact is not None
    assert verification.dedup_impact.rows_dropped == 1
