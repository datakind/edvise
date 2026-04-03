import json

import pandas as pd
import pytest

from edvise.genai.identity_agent.grain_contract.prompt_builder import (
    build_identity_agent_user_message,
    format_column_list,
    parse_identity_grain_contract,
    strip_json_fences,
)
from edvise.genai.identity_agent.grain_contract.schemas import Confidence, DedupStrategy
from edvise.genai.identity_agent.profiling.key_profiler import (
    CandidateKey,
    CandidateKeyProfile,
    KeyProfile,
)


def _minimal_key_profile() -> KeyProfile:
    ck = CandidateKey(
        columns=["student_id"],
        uniqueness_score=1.0,
        null_rate=0.0,
        rank=1,
    )
    prof = CandidateKeyProfile(
        candidate_key=ck,
        non_unique_rows=0,
        affected_groups=0,
        group_size_distribution={},
        within_group_variance=[],
    )
    return KeyProfile(candidate_key_profiles=[prof])


def test_format_column_list():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    text = format_column_list(df)
    assert "a:" in text and "b:" in text


def test_build_identity_agent_user_message_with_df():
    df = pd.DataFrame({"student_id": [1]})
    msg = build_identity_agent_user_message(
        "school_a",
        "students",
        _minimal_key_profile(),
        df=df,
    )
    assert "school_a" in msg
    assert "students" in msg
    assert "student_id" in msg
    assert "candidate_key_profiles" in msg


def test_build_identity_agent_user_message_column_list():
    msg = build_identity_agent_user_message(
        "school_a",
        "students",
        _minimal_key_profile(),
        column_list="  id: int64",
    )
    assert "id: int64" in msg


def test_build_identity_agent_user_message_rejects_both_or_neither():
    df = pd.DataFrame({"x": [1]})
    kp = _minimal_key_profile()
    with pytest.raises(ValueError, match="only one"):
        build_identity_agent_user_message("i", "t", kp, df=df, column_list="x")
    with pytest.raises(ValueError, match="exactly one"):
        build_identity_agent_user_message("i", "t", kp)


def test_parse_identity_grain_contract_dict():
    raw = {
        "institution_id": "x",
        "table": "t",
        "post_clean_primary_key": ["student_id"],
        "dedup_policy": {
            "strategy": "no_dedup",
            "sort_by": None,
            "keep": None,
            "notes": "intentional multi-row",
        },
        "cleaning_collapses_to_student_grain": False,
        "row_selection_required": True,
        "join_keys_for_2a": ["student_id", "term"],
        "confidence": "HIGH",
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Course grain.",
    }
    c = parse_identity_grain_contract(raw)
    assert c.post_clean_primary_key == ["student_id"]
    assert c.unique_keys == ["student_id"]
    assert c.dedup_policy.strategy == DedupStrategy.no_dedup


def test_parse_identity_grain_contract_fenced_json():
    inner = {
        "institution_id": "x",
        "table": "t",
        "post_clean_primary_key": ["id"],
        "dedup_policy": {
            "strategy": "true_duplicate",
            "sort_by": None,
            "keep": "first",
            "notes": "drop dupes",
        },
        "cleaning_collapses_to_student_grain": True,
        "row_selection_required": False,
        "join_keys_for_2a": ["id"],
        "confidence": "MEDIUM",
        "hitl_flag": True,
        "hitl_question": "Confirm?",
        "reasoning": "Demo table.",
    }
    text = "```json\n" + json.dumps(inner) + "\n```"
    c = parse_identity_grain_contract(text)
    assert c.confidence == Confidence.MEDIUM


def test_low_confidence_requires_hitl():
    raw = {
        "institution_id": "x",
        "table": "t",
        "post_clean_primary_key": ["a"],
        "dedup_policy": {
            "strategy": "no_dedup",
            "sort_by": None,
            "keep": None,
            "notes": "",
        },
        "cleaning_collapses_to_student_grain": False,
        "row_selection_required": True,
        "join_keys_for_2a": ["a", "b"],
        "confidence": "LOW",
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Ambiguous.",
    }
    with pytest.raises(ValueError, match="hitl_flag"):
        parse_identity_grain_contract(raw)


def test_strip_json_fences():
    assert strip_json_fences('```\n{"a": 1}\n```').strip() == '{"a": 1}'
