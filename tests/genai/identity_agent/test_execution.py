"""Tests for identity_agent.execution (in-memory transforms + school config / schema contract)."""

import pandas as pd
import pytest

from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.genai.identity_agent.execution import (
    apply_grain_dedup,
    apply_grain_execution,
    apply_grain_term_order,
    build_dedupe_fn_from_grain_contract,
    merge_grain_contracts_into_school_config,
)
from edvise.genai.identity_agent.grain_inference.schemas import (
    Confidence,
    DedupPolicy,
    DedupStrategy,
    IdentityGrainContract,
)


def _contract(**kwargs) -> IdentityGrainContract:
    defaults: dict = dict(
        institution_id="x",
        table="t",
        post_clean_primary_key=["k"],
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.true_duplicate,
            sort_by=None,
            keep="first",
            notes="",
        ),
        cleaning_collapses_to_student_grain=True,
        row_selection_required=False,
        join_keys_for_2a=["k"],
        confidence=Confidence.HIGH,
        hitl_flag=False,
        hitl_question=None,
        reasoning="",
        term_order_column=None,
    )
    defaults.update(kwargs)
    return IdentityGrainContract(**defaults)


def test_no_dedup_leaves_rows():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _contract(
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.no_dedup,
            sort_by=None,
            keep=None,
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 2


def test_true_duplicate_collapses():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _contract(
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
    c = _contract(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k", "t"],
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.temporal_collapse,
            sort_by="t",
            keep="last",
            notes="",
        ),
    )
    out = apply_grain_dedup(df, c)
    assert len(out) == 1
    assert int(out["t"].iloc[0]) == 3


def test_apply_grain_term_order_adds_columns():
    df = pd.DataFrame({"term": ["Fall 2020", "Spring 2021"], "k": [1, 2]})
    c = _contract(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k", "term"],
        term_order_column="term",
    )
    out = apply_grain_term_order(df, c)
    assert "term_order" in out.columns


def test_apply_grain_execution_order_dedup_then_term():
    df = pd.DataFrame(
        {
            "k": [1, 1],
            "term": ["Fall 2020", "Fall 2020"],
        }
    )
    c = _contract(
        post_clean_primary_key=["k"],
        join_keys_for_2a=["k", "term"],
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.true_duplicate,
            sort_by=None,
            keep="first",
            notes="",
        ),
        term_order_column="term",
    )
    out = apply_grain_execution(df, c)
    assert len(out) == 1
    assert "term_order" in out.columns


def test_build_dedupe_fn_from_grain_contract():
    df = pd.DataFrame({"k": [1, 1], "v": [1, 2]})
    c = _contract(
        post_clean_primary_key=["k"],
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.true_duplicate,
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
    c = _contract(post_clean_primary_key=["k"])
    with pytest.raises(ValueError, match="missing columns"):
        apply_grain_dedup(df, c)


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


def _merge_contract(table: str, uks: list[str]) -> IdentityGrainContract:
    return IdentityGrainContract(
        institution_id="test_inst",
        table=table,
        post_clean_primary_key=uks,
        dedup_policy=DedupPolicy(
            strategy=DedupStrategy.no_dedup,
            sort_by=None,
            keep=None,
            notes="",
        ),
        cleaning_collapses_to_student_grain=False,
        row_selection_required=True,
        join_keys_for_2a=uks,
        confidence=Confidence.HIGH,
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


def test_merge_preserves_institution_when_partial():
    school = _school_config()
    out = merge_grain_contracts_into_school_config(school, {})
    assert out.institution_id == school.institution_id
    assert out.datasets["students"].primary_keys == ["student_id"]


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
