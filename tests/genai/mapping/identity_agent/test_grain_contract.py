import json

import pandas as pd
import pytest
from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.grain_inference.hitl_uniqueness_backfill import (
    backfill_hitl_uniqueness_scores_from_key_profile,
)
from edvise.genai.mapping.identity_agent.grain_inference.prompt import (
    build_identity_agent_user_message,
    format_column_list,
    parse_grain_contract,
    parse_grain_contract_with_hitl,
    parse_institution_grain_contracts,
    strip_json_fences,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainAmbiguityHITLContext,
    GrainCandidateKeyEntry,
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    ReentryDepth,
)
from edvise.genai.mapping.shared.hitl import PIPELINE_HITL_CONFIDENCE_THRESHOLD
from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    GrainContract,
    InstitutionGrainContract,
    build_institution_grain_contracts,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import GrainCandidateKeyEntry
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    TermOrderConfig,
)
from edvise.genai.mapping.identity_agent.profiling import (
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


def test_parse_grain_contract_legacy_student_id_alias_maps_to_learner_id_alias():
    raw = {
        "institution_id": "x",
        "table": "t",
        "student_id_alias": "legacy_col",
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
    c = parse_grain_contract(raw)
    assert c.learner_id_alias == "legacy_col"


def test_parse_grain_contract_dict():
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
    c = parse_grain_contract(raw)
    assert c.post_clean_primary_key == ["student_id"]
    assert c.unique_keys == ["student_id"]
    assert c.dedup_policy.strategy == "no_dedup"


def test_parse_grain_contract_fenced_json():
    inner = {
        "institution_id": "x",
        "table": "t",
        "post_clean_primary_key": ["id"],
        "dedup_policy": {
            "strategy": "true_duplicate",
            "sort_by": None,
            "keep": None,
            "notes": "drop dupes",
        },
        "row_selection_required": False,
        "join_keys_for_2a": ["id"],
        "confidence": 0.72,
        "hitl_flag": False,
        "hitl_question": "Confirm?",
        "reasoning": "Demo table.",
    }
    text = "```json\n" + json.dumps(inner) + "\n```"
    c = parse_grain_contract(text)
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
        parse_grain_contract(raw)


def test_dedup_policy_rejects_invalid_strategy():
    raw = _valid_minimal_contract_dict()
    raw["dedup_policy"]["strategy"] = "any_row"
    with pytest.raises(ValidationError):
        parse_grain_contract(raw)


def test_parse_policy_required_dedup_strategy():
    raw = _valid_minimal_contract_dict()
    raw["dedup_policy"]["strategy"] = "policy_required"
    c = parse_grain_contract(raw)
    assert c.dedup_policy.strategy == "policy_required"


def test_legacy_term_config_stripped_from_grain_json():
    """Older combined prompts included term_config; grain-stage schema is grain-only."""
    raw = _valid_minimal_contract_dict()
    raw["term_config"] = {
        "term_col": "term_descr",
        "season_map": [{"raw": "FA", "canonical": "FALL"}],
        "term_extraction": "standard",
        "hook_spec": None,
    }
    c = parse_grain_contract(raw)
    assert isinstance(c, GrainContract)


def test_term_config_hook_required_requires_hook_spec():
    with pytest.raises(ValidationError, match="hook_spec"):
        TermOrderConfig(
            term_col="t",
            season_map=[],
            term_extraction="hook_required",
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
        "confidence": PIPELINE_HITL_CONFIDENCE_THRESHOLD - 0.01,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Ambiguous.",
    }
    with pytest.raises(ValueError, match="hitl_flag"):
        parse_grain_contract(raw)


def test_confidence_at_threshold_requires_hitl():
    """Slightly below pipeline threshold: hitl_flag must be true."""
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
        "confidence": PIPELINE_HITL_CONFIDENCE_THRESHOLD - 1e-6,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "Slightly below threshold.",
    }
    with pytest.raises(ValueError, match="hitl_flag"):
        parse_grain_contract(raw)


def test_parse_grain_hitl_flag_true_requires_non_empty_hitl_items():
    """When hitl_flag is true, top-level hitl_items must be non-empty (enforced in parse)."""
    raw = {
        "institution_id": "x",
        "table": "t",
        "learner_id_alias": None,
        "post_clean_primary_key": ["a"],
        "dedup_policy": {
            "strategy": "no_dedup",
            "sort_by": None,
            "keep": None,
            "notes": "",
        },
        "row_selection_required": True,
        "join_keys_for_2a": ["a", "b"],
        "confidence": 0.9,
        "hitl_flag": True,
        "reasoning": "r",
        "hitl_items": [],
    }
    with pytest.raises(ValueError, match="hitl_flag=true"):
        parse_grain_contract_with_hitl(raw)


def test_confidence_exactly_at_pipeline_threshold_without_hitl_flag_fails():
    """At PIPELINE_HITL_CONFIDENCE_THRESHOLD, hitl_flag must be true (validator uses <=)."""
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
        "confidence": PIPELINE_HITL_CONFIDENCE_THRESHOLD,
        "hitl_flag": False,
        "hitl_question": None,
        "reasoning": "On the threshold.",
    }
    with pytest.raises(ValueError, match="hitl_flag"):
        parse_grain_contract(raw)


def test_strip_json_fences():
    assert strip_json_fences('```\n{"a": 1}\n```').strip() == '{"a": 1}'


def _minimal_grain(institution_id: str = "x", table: str = "t") -> GrainContract:
    return GrainContract(
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
    c1 = _minimal_grain("school_a", "student")
    c2 = _minimal_grain("school_a", "course")
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
    c = _minimal_grain("other", "t")
    with pytest.raises(ValueError, match="institution_id"):
        InstitutionGrainContract(institution_id="here", datasets={"t": c})


def test_grain_candidate_key_entry_uniqueness_score_coerces_null():
    a = GrainCandidateKeyEntry.model_validate(
        {"rank": 1, "columns": ["a"], "uniqueness_score": None}
    )
    assert a.uniqueness_score == 0.0
    b = GrainCandidateKeyEntry.model_validate({"rank": 1, "columns": ["a"]})
    assert b.uniqueness_score == 0.0


def test_grain_candidate_key_entry_uniqueness_score_accepts_percent_scale():
    """LLMs often send 99.8 meaning 99.8%; must normalize to 0.998, not fail le=1.0."""
    x = GrainCandidateKeyEntry.model_validate(
        {"rank": 1, "columns": ["a"], "uniqueness_score": 99.8}
    )
    assert abs(x.uniqueness_score - 0.998) < 1e-9
    y = GrainCandidateKeyEntry.model_validate(
        {"rank": 2, "columns": ["a", "b"], "uniqueness_score": 100}
    )
    assert y.uniqueness_score == 1.0
    z = GrainCandidateKeyEntry.model_validate(
        {"rank": 1, "columns": ["a"], "uniqueness_score": 0.998}
    )
    assert z.uniqueness_score == 0.998


def test_backfill_hitl_uniqueness_scores_replaces_invented_zero_from_profile():
    """Model may emit 0 for 'semantic' rank-1; profile is source of truth for that column set."""
    ck = CandidateKey(
        columns=["student_id"],
        uniqueness_score=0.5244,
        null_rate=0.0,
        rank=1,
    )
    key_profile = RankedCandidateProfiles(
        candidate_key_profiles=[
            CandidateProfile(
                candidate_key=ck,
                non_unique_rows=10,
                affected_groups=2,
                group_size_distribution={2: 5},
                within_group_variance=[],
            )
        ]
    )
    item = HITLItem(
        item_id="inst_t_q",
        institution_id="inst",
        table="t",
        domain=HITLDomain.IDENTITY_GRAIN,
        hitl_question="q",
        hitl_context=GrainAmbiguityHITLContext(
            candidate_keys=[
                GrainCandidateKeyEntry(
                    rank=1, columns=["student_id"], uniqueness_score=0.0, notes="wrong"
                )
            ],
            variance_profile={},
        ),
        options=[
            HITLOption(
                option_id="nd",
                label="nd",
                description="d",
                resolution=GrainResolution(dedup_strategy="no_dedup").model_dump(
                    mode="json"
                ),
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="c",
                resolution=None,
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id="inst",
            table="t",
            config="grain_contract",
            field="dedup_policy",
        ),
    )
    out = backfill_hitl_uniqueness_scores_from_key_profile([item], key_profile)
    ctx = out[0].hitl_context
    assert isinstance(ctx, GrainAmbiguityHITLContext)
    assert ctx.candidate_keys[0].uniqueness_score == 0.5244
