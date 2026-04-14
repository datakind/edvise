"""Tests for identity_agent.grain_inference.runner."""

import json

import pandas as pd
import pytest

from edvise.genai.mapping.identity_agent.grain_inference.runner import (
    run_identity_agent_with_hitl,
    run_identity_agents_for_institution_with_hitl,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
)
from edvise.genai.mapping.identity_agent.profiling import (
    CandidateKey,
    CandidateProfile,
    RankedCandidateProfiles,
)


def _kp() -> RankedCandidateProfiles:
    ck = CandidateKey(
        columns=["student_id"],
        uniqueness_score=1.0,
        null_rate=0.0,
        rank=1,
    )
    prof = CandidateProfile(
        candidate_key=ck,
        non_unique_rows=0,
        affected_groups=0,
        group_size_distribution={},
        within_group_variance=[],
    )
    return RankedCandidateProfiles(candidate_key_profiles=[prof])


def _contract_json(*, confidence: float, hitl_flag: bool, table: str = "t") -> str:
    return json.dumps(
        {
            "institution_id": "inst",
            "table": table,
            "post_clean_primary_key": ["student_id"],
            "dedup_policy": {
                "strategy": "no_dedup",
                "sort_by": None,
                "keep": None,
                "notes": "",
            },
            "row_selection_required": True,
            "join_keys_for_2a": ["student_id", "term"],
            "confidence": confidence,
            "hitl_flag": hitl_flag,
            "hitl_question": None,
            "reasoning": "test",
        }
    )


def test_run_identity_agent_calls_llm_and_parse():
    df = pd.DataFrame({"student_id": [1]})
    raw = _contract_json(confidence=0.9, hitl_flag=False)

    def llm(system: str, user: str) -> str:
        assert "IdentityAgent" in system or len(system) > 100
        assert "inst" in user and "students" in user
        return raw

    c, items = run_identity_agent_with_hitl(
        institution_id="inst",
        dataset_name="students",
        key_profile=_kp(),
        df=df,
        llm_complete=llm,
    )
    assert c.table == "t"
    assert c.confidence == 0.9
    assert items == []


def test_run_identity_agents_for_institution_with_hitl_routes_callbacks():
    df_a = pd.DataFrame({"student_id": [1]})
    df_b = pd.DataFrame({"student_id": [2]})
    profiles = {"a": _kp(), "b": _kp()}
    dfs = {"a": df_a, "b": df_b}

    calls: list[tuple[str, str]] = []

    n = 0

    def llm(_system: str, _user: str) -> str:
        nonlocal n
        n += 1
        # first: auto path; second: HITL via hitl_flag
        if n == 1:
            return _contract_json(confidence=0.95, hitl_flag=False, table="a")
        return _contract_json(confidence=0.95, hitl_flag=True, table="b")

    def q(c):
        calls.append(("hitl", c.table))

    def auto(c):
        calls.append(("auto", c.table))

    out, _hitl = run_identity_agents_for_institution_with_hitl(
        institution_id="inst",
        institution_profiles=profiles,
        dfs=dfs,
        llm_complete=llm,
        confidence_threshold=IDENTITY_CONFIDENCE_HITL_THRESHOLD,
        queue_for_hitl_review=q,
        auto_approve_and_apply=auto,
    )
    assert set(out.keys()) == {"a", "b"}
    assert calls == [("auto", "a"), ("hitl", "b")]


def test_run_identity_agents_for_institution_with_hitl_missing_df():
    profiles = {"a": _kp()}
    dfs: dict = {}

    with pytest.raises(KeyError, match="No DataFrame"):
        run_identity_agents_for_institution_with_hitl(
            institution_id="inst",
            institution_profiles=profiles,
            dfs=dfs,
            llm_complete=lambda s, u: _contract_json(confidence=0.9, hitl_flag=False),
        )
