"""Tests for identity_agent.execution (in-memory transforms + school config / schema contract)."""

import pandas as pd
import pytest

from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.genai.mapping.identity_agent.execution import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_term_order_from_contract,
    apply_term_order_from_config,
    build_dedupe_fn_from_grain_contract,
    merge_grain_contracts_into_school_config,
    merge_grain_learner_id_alias_into_school_config,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    DedupPolicy,
    GrainContract,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    TermContract,
    TermOrderConfig,
)
from edvise.genai.mapping.identity_agent.term_normalization.term_order import (
    term_order_column_for_clean_dataset,
    term_order_fn_from_term_order_config,
)


def _grain(**kwargs) -> GrainContract:
    defaults: dict = dict(
        institution_id="x",
        table="t",
        post_clean_primary_key=["k"],
        dedup_policy=DedupPolicy(
            strategy="true_duplicate",
            sort_by=None,
            keep="first",
            notes="",
        ),
        row_selection_required=False,
        join_keys_for_2a=["k"],
        confidence=0.95,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
    )
    defaults.update(kwargs)
    return GrainContract(**defaults)


def _term_pass(term_order: TermOrderConfig) -> TermContract:
    return TermContract(
        institution_id="x",
        table="t",
        term_config=term_order,
        confidence=0.95,
        hitl_flag=False,
        hitl_question=None,
        reasoning="test",
    )


def test_no_dedup_leaves_rows():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _grain(
        dedup_policy=DedupPolicy(
            strategy="no_dedup",
            sort_by=None,
            keep=None,
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 2


def test_true_duplicate_collapses():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _grain(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k"],
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 1


def test_temporal_collapse_keep_last():
    df = pd.DataFrame(
        {
            "k": [1, 1, 1],
            "t": [1, 2, 3],
        }
    )
    c = _grain(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k", "t"],
        dedup_policy=DedupPolicy(
            strategy="temporal_collapse",
            sort_by="t",
            sort_ascending=False,
            keep="first",
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 1
    assert int(out["t"].iloc[0]) == 3


def test_apply_term_order_from_contract_adds_columns():
    df = pd.DataFrame({"term": ["Fall 2020", "Spring 2021"], "k": [1, 2]})
    tp = _term_pass(
        TermOrderConfig(
            term_col="term",
            season_map=[
                {"raw": "Spring", "canonical": "SPRING"},
                {"raw": "Fall", "canonical": "FALL"},
            ],
            term_extraction="standard",
        ),
    )
    out = apply_term_order_from_contract(df, tp)
    assert "_term_order" in out.columns


def test_apply_term_order_from_contract_yyyytt_with_season_map():
    df = pd.DataFrame({"term": ["2018FA", "2019SP"], "k": [1, 2]})
    tp = _term_pass(
        TermOrderConfig(
            term_col="term",
            season_map=[
                {"raw": "SP", "canonical": "SPRING"},
                {"raw": "FA", "canonical": "FALL"},
            ],
            term_extraction="standard",
        ),
    )
    out = apply_term_order_from_contract(df, tp)
    assert "_term_order" in out.columns
    assert "_year" in out.columns
    assert "_season" in out.columns


def test_apply_grain_execution_order_dedup_then_term():
    df = pd.DataFrame(
        {
            "k": [1, 1],
            "term": ["Fall 2020", "Fall 2020"],
        }
    )
    g = _grain(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k", "term"],
        dedup_policy=DedupPolicy(
            strategy="true_duplicate",
            sort_by=None,
            keep="first",
            notes="",
        ),
    )
    tp = _term_pass(
        TermOrderConfig(
            term_col="term",
            season_map=[
                {"raw": "Fall", "canonical": "FALL"},
            ],
            term_extraction="standard",
        ),
    )
    out = apply_grain_execution(df, g, tp)
    assert len(out) == 1
    assert "_term_order" in out.columns


def test_build_dedupe_fn_from_grain_contract():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _grain(
        post_clean_primary_key=["k"],
        dedup_policy=DedupPolicy(
            strategy="true_duplicate",
            sort_by=None,
            keep="first",
            notes="",
        ),
    )
    fn = build_dedupe_fn_from_grain_contract(c)
    out = fn(df)
    assert len(out) == 1


def test_missing_key_columns_raises():
    df = pd.DataFrame({"x": [1]})
    c = _grain(post_clean_primary_key=["k"])
    with pytest.raises(ValueError, match="apply_grain_dedup"):
        apply_grain_dedup(df, c)


def test_apply_grain_dedup_resolves_term_desc_prefix_to_term_descr():
    """Grain contract typo / abbreviation vs normalized header (e.g. UCF TERM_DESCR)."""
    df = pd.DataFrame(
        {
            "student_id": ["a", "a"],
            "term_descr": ["Fall 2020", "Fall 2020"],
            "v": [1, 2],
        }
    )
    c = _grain(
        post_clean_primary_key=["student_id", "TERM_DESC"],
        join_keys_for_2a=["student_id", "term_descr"],
        dedup_policy=DedupPolicy(
            strategy="true_duplicate",
            sort_by=None,
            keep="first",
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 1


def test_policy_required_skips_dedup():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _grain(
        dedup_policy=DedupPolicy(
            strategy="policy_required",
            sort_by=None,
            keep=None,
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 2


def test_apply_term_order_raises_when_hook_required_hooks_not_wired():
    df = pd.DataFrame({"term": ["1192"]})
    c = TermOrderConfig(
        term_col="term",
        season_map=[],
        term_extraction="hook_required",
        hook_spec=HookSpec(file="pipelines/x/helpers/term_hooks.py", functions=[]),
    )
    with pytest.raises(ValueError, match="hook_required"):
        apply_term_order_from_config(df, c)


def test_apply_term_order_split_year_season_columns():
    df = pd.DataFrame(
        {
            "yr": [2020, 2021],
            "sem": ["FA", "SP"],
            "k": [1, 2],
        }
    )
    cfg = TermOrderConfig(
        year_col="yr",
        season_col="sem",
        season_map=[
            {"raw": "SP", "canonical": "SPRING"},
            {"raw": "FA", "canonical": "FALL"},
        ],
        term_extraction="standard",
    )
    out = apply_term_order_from_config(df, cfg)
    assert list(out["_year"]) == [2020, 2021]
    assert list(out["_season"]) == ["fa", "sp"]
    assert "_term_order" in out.columns


def test_term_order_fn_wrapper_and_clean_column_hint():
    cfg = TermOrderConfig(
        year_col="yr",
        season_col="sem",
        season_map=[{"raw": "FA", "canonical": "FALL"}],
        term_extraction="standard",
    )
    assert term_order_column_for_clean_dataset(cfg) == "yr"
    fn = term_order_fn_from_term_order_config(cfg)
    df = pd.DataFrame({"yr": [2019], "sem": ["FA"]})
    with pytest.raises(ValueError, match="term_column"):
        fn(df, "sem")
    out = fn(df, "yr")
    assert int(out["_year"].iloc[0]) == 2019
    assert out["_season"].iloc[0] == "fa"


def test_term_order_column_for_combined_term_col():
    cfg = TermOrderConfig(
        term_col="term",
        season_map=[{"raw": "FA", "canonical": "FALL"}],
        term_extraction="standard",
    )
    assert term_order_column_for_clean_dataset(cfg) == "term"


def test_term_order_column_matches_clean_dataset_normalization():
    """LLM may emit uppercase headers; clean_dataset uses snake_case column names."""
    cfg = TermOrderConfig(
        term_col="TERM_DESC",
        season_map=[
            {"raw": "Spring", "canonical": "SPRING"},
            {"raw": "Fall", "canonical": "FALL"},
        ],
        term_extraction="standard",
    )
    assert term_order_column_for_clean_dataset(cfg) == "term_desc"
    fn = term_order_fn_from_term_order_config(cfg)
    df = pd.DataFrame({"term_desc": ["Fall 2020"]})
    out = fn(df, "term_desc")
    assert "_term_order" in out.columns


def test_apply_grain_dedup_maps_contract_learner_id_alias_to_student_id():
    """After clean_dataset rename, the frame has student_id; keys may still use the pre-rename name."""
    df = pd.DataFrame({"student_id": ["a", "a"], "x": [1, 2]})
    c = _grain(
        learner_id_alias="student_id_randomized_datakind",
        post_clean_primary_key=["student_id_randomized_datakind"],
        join_keys_for_2a=["student_id_randomized_datakind"],
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 1


# --- merge_grain_contracts_into_school_config ---


def _school_config() -> SchoolMappingConfig:
    return SchoolMappingConfig(
        institution_id="test_inst",
        institution_name="Test",
        datasets={
            "students": DatasetConfig(
                primary_keys=["student_id"],
                files=["/tmp/a.csv"],
            ),
            "courses": DatasetConfig(
                primary_keys=["student_id", "term"],
                files=["/tmp/b.csv"],
            ),
        },
    )


def _merge_contract(
    table: str,
    uks: list[str],
    *,
    learner_id_alias: str | None = None,
) -> GrainContract:
    return GrainContract(
        institution_id="test_inst",
        table=table,
        learner_id_alias=learner_id_alias,
        post_clean_primary_key=uks,
        dedup_policy=DedupPolicy(
            strategy="no_dedup",
            sort_by=None,
            keep=None,
            notes="",
        ),
        row_selection_required=True,
        join_keys_for_2a=uks,
        confidence=0.95,
        hitl_flag=False,
        hitl_question=None,
        reasoning="test",
    )


def test_merge_updates_only_listed_datasets():
    school = _school_config()
    gc_students = _merge_contract("students", ["student_id", "cohort_id"])
    out = merge_grain_contracts_into_school_config(
        school,
        {"students": gc_students},
    )
    assert out.datasets["students"].primary_keys == ["student_id", "cohort_id"]
    assert out.datasets["courses"].primary_keys == ["student_id", "term"]


def test_merge_canonicalizes_learner_id_alias_in_primary_keys():
    school = _school_config()
    gc = _merge_contract(
        "students",
        ["legacy_student_col", "term"],
        learner_id_alias="legacy_student_col",
    )
    out = merge_grain_contracts_into_school_config(school, {"students": gc})
    assert out.datasets["students"].primary_keys == ["student_id", "term"]


def test_merge_preserves_institution_when_partial():
    school = _school_config()
    out = merge_grain_contracts_into_school_config(school, {})
    assert out.institution_id == school.institution_id
    assert out.datasets["students"].primary_keys == ["student_id"]


def test_merge_sets_cleaning_student_id_alias_from_grain_learner_id_alias():
    school = _school_config()
    gc = _merge_contract(
        "students",
        ["student_id"],
        learner_id_alias="student_id_randomized_datakind",
    )
    out = merge_grain_contracts_into_school_config(school, {"students": gc})
    assert out.cleaning is not None
    assert out.cleaning.student_id_alias == "student_id_randomized_datakind"


def test_merge_grain_learner_id_alias_only_is_idempotent():
    school = _school_config()
    gc = _merge_contract(
        "students",
        ["student_id"],
        learner_id_alias="col_a",
    )
    once = merge_grain_learner_id_alias_into_school_config(school, {"students": gc})
    twice = merge_grain_learner_id_alias_into_school_config(once, {"students": gc})
    assert twice.cleaning and twice.cleaning.student_id_alias == "col_a"


def test_merge_conflicting_grain_learner_id_alias_raises():
    school = _school_config()
    a = _merge_contract("students", ["student_id"], learner_id_alias="a")
    b = _merge_contract("courses", ["student_id", "term"], learner_id_alias="b")
    with pytest.raises(ValueError, match="disagree on learner_id_alias"):
        merge_grain_contracts_into_school_config(school, {"students": a, "courses": b})


def test_merge_unknown_dataset_raises():
    school = _school_config()
    gc = _merge_contract("students", ["student_id"])
    with pytest.raises(KeyError, match="unknown"):
        merge_grain_contracts_into_school_config(
            school,
            {"students": gc, "nope": gc},
        )


def test_merge_empty_unique_keys_raises():
    school = _school_config()
    bad = _merge_contract("students", [])
    bad = bad.model_copy(update={"post_clean_primary_key": []})
    try:
        merge_grain_contracts_into_school_config(school, {"students": bad})
    except ValueError as e:
        assert "empty" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
