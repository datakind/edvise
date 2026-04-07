import json

import pandas as pd
import pytest
from pydantic import ValidationError

from edvise.genai.identity_agent.grain_inference.prompt_builder import (
    build_identity_agent_user_message,
    format_column_list,
    parse_identity_grain_contract,
    parse_institution_grain_contracts,
    strip_json_fences,
)
from edvise.genai.identity_agent.grain_inference.schemas import (
    IDENTITY_CONFIDENCE_HITL_THRESHOLD,
    IdentityGrainContract,
    InstitutionGrainContracts,
    TermOrderConfig,
    build_institution_grain_contracts,
)
from edvise.genai.identity_agent.profiling import (
    CandidateKey,
    CandidateProfile,
    RankedCandidateProfiles,
)


def _minimal_key_profile() -> RankedCandidateProfiles:
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
        "row_selection_required": True,
        "join_keys_for_2a": ["student_id", "term"],
        "confidence": 0.95,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Course grain.",
    }
    c = parse_identity_grain_contract(raw)
    assert c.post_clean_primary_key == ["student_id"]
    assert c.unique_keys == ["student_id"]
    assert c.dedup_policy.strategy == "no_dedup"


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
        "row_selection_required": False,
        "join_keys_for_2a": ["id"],
        "confidence": 0.72,
        "hitl_flag": True,
        "hitl_question": "Confirm?",
        "reasoning": "Demo table.",
    }
    text = "```json\n" + json.dumps(inner) + "\n```"
    c = parse_identity_grain_contract(text)
    assert c.confidence == 0.72


def _valid_minimal_contract_dict() -> dict:
    return {
        "institution_id": "x",
        "table": "t",
        "post_clean_primary_key": ["student_id"],
        "dedup_policy": {
            "strategy": "no_dedup",
            "sort_by": None,
            "keep": None,
            "notes": "",
        },
        "row_selection_required": True,
        "join_keys_for_2a": ["student_id", "term"],
        "confidence": 0.95,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "ok",
    }


def test_dedup_policy_rejects_invalid_keep():
    raw = _valid_minimal_contract_dict()
    raw["dedup_policy"]["keep"] = "any_row"
    with pytest.raises(ValidationError):
        parse_identity_grain_contract(raw)


def test_dedup_policy_rejects_invalid_strategy():
    raw = _valid_minimal_contract_dict()
    raw["dedup_policy"]["strategy"] = "any_row"
    with pytest.raises(ValidationError):
        parse_identity_grain_contract(raw)


def test_parse_policy_required_dedup_strategy():
    raw = _valid_minimal_contract_dict()
    raw["dedup_policy"]["strategy"] = "policy_required"
    c = parse_identity_grain_contract(raw)
    assert c.dedup_policy.strategy == "policy_required"


def test_term_config_parses_standard_with_hook_spec_null():
    raw = _valid_minimal_contract_dict()
    raw["term_config"] = {
        "term_col": "term_descr",
        "season_map": [{"raw": "FA", "canonical": "FALL"}],
        "term_extraction": "standard",
        "hook_spec": None,
    }
    c = parse_identity_grain_contract(raw)
    assert c.term_config is not None
    assert c.term_config.term_col == "term_descr"
    assert c.term_config.term_extraction == "standard"


def test_term_config_custom_requires_hook_spec():
    with pytest.raises(ValidationError, match="hook_spec"):
        TermOrderConfig(
            term_col="t",
            season_map=[],
            term_extraction="custom",
            hook_spec=None,
        )


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
        "row_selection_required": True,
        "join_keys_for_2a": ["a", "b"],
        "confidence": IDENTITY_CONFIDENCE_HITL_THRESHOLD - 0.01,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Ambiguous.",
    }
    with pytest.raises(ValueError, match="hitl_flag"):
        parse_identity_grain_contract(raw)


def test_strip_json_fences():
    assert strip_json_fences('```\n{"a": 1}\n```').strip() == '{"a": 1}'


def _minimal_contract(
    institution_id: str = "x", table: str = "t"
) -> IdentityGrainContract:
    return IdentityGrainContract(
        institution_id=institution_id,
        table=table,
        post_clean_primary_key=["k"],
        dedup_policy={
            "strategy": "no_dedup",
            "sort_by": None,
            "keep": None,
            "notes": "",
        },
        row_selection_required=False,
        join_keys_for_2a=["k"],
        confidence=0.95,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
    )


def test_institution_grain_contracts_roundtrip():
    c1 = _minimal_contract("school_a", "student")
    c2 = _minimal_contract("school_a", "course")
    env = build_institution_grain_contracts(
        "school_a",
        {"student": c1, "course": c2},
    )
    text = env.model_dump_json(indent=2)
    back = parse_institution_grain_contracts(text)
    assert back.institution_id == "school_a"
    assert set(back.datasets) == {"student", "course"}
    assert back.datasets["student"].table == "student"


def test_institution_grain_contracts_rejects_mismatched_institution_id():
    c = _minimal_contract("other", "t")
    with pytest.raises(ValueError, match="institution_id"):
        InstitutionGrainContracts(institution_id="here", datasets={"t": c})
