import pandas as pd
import pytest

from edvise.configs.genai import KeyCollisionDedupeConfig

from edvise.data_audit.identity_agent import deduplication as d


def test_drop_exact_row_duplicates():
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    out = d.drop_exact_row_duplicates(df)
    assert len(out) == 1


def test_drop_duplicate_keys_sort_prefers_row():
    df = pd.DataFrame(
        {
            "k": [1, 1, 1],
            "score": [10, 30, 20],
        }
    )
    out = d.drop_duplicate_keys(df, ["k"], sort_by=["score"], ascending=False)
    assert len(out) == 1
    assert int(out["score"].iloc[0]) == 30


def test_drop_duplicate_keys_missing_column():
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError, match="key_cols"):
        d.drop_duplicate_keys(df, ["missing"])


def test_suffix_disambiguate_only_duplicate_groups():
    df = pd.DataFrame(
        {
            "sid": [1, 1, 2],
            "term": ["FA", "FA", "FA"],
            "cn": ["X100", "X100", "Y200"],
        }
    )
    out = d.suffix_disambiguate_within_keys(df, ["sid", "term"], "cn")
    assert out["cn"].tolist() == ["X100-1", "X100-2", "Y200"]


def test_suffix_disambiguate_sort_within_group():
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["FA", "FA"],
            "cn": ["X100", "X100"],
            "credits": [3.0, 4.0],
        }
    )
    out = d.suffix_disambiguate_within_keys(
        df,
        ["sid", "term"],
        "cn",
        sort_within_group_by=["credits"],
        ascending=False,
    )
    # Higher credits first -> gets -1
    hi = out.loc[out["credits"] == 4.0, "cn"].iloc[0]
    lo = out.loc[out["credits"] == 3.0, "cn"].iloc[0]
    assert hi.endswith("-1")
    assert lo.endswith("-2")


def test_resolve_key_collisions_disambig_when_classification_differs():
    # Same key, different "classification" -> suffix class_number
    df = pd.DataFrame(
        {
            "student_id": ["u1", "u1"],
            "course_name": ["CHEM-207-1", "CHEM-207-1"],
            "term": ["2019FA", "2019FA"],
            "class_number": ["37559", "37559"],
            "course_classification": ["Lab", "Lecture"],
        }
    )
    key = ["student_id", "course_name", "term", "class_number"]
    out = d.resolve_key_collisions(
        df,
        key_cols=key,
        conflict_columns=["course_classification"],
        disambiguate_column="class_number",
    )
    assert len(out) == 2
    nums = set(out["class_number"].astype(str))
    assert len(nums) == 2
    assert all("37559" in x for x in nums)


def test_apply_key_collision_dedupe_from_spec_skips_when_key_cols_missing():
    df = pd.DataFrame({"a": [1]})
    spec = KeyCollisionDedupeConfig(
        key_cols=["missing"],
        conflict_columns=["x"],
        disambiguate_column="a",
    ).model_dump(exclude_none=True)
    out = d.apply_key_collision_dedupe_from_spec(df, spec)
    pd.testing.assert_frame_equal(out, df)


def test_apply_key_collision_dedupe_from_spec_model_dump():
    df = pd.DataFrame(
        {
            "student_id": ["u1", "u1"],
            "term": ["FA", "FA"],
            "class_number": ["1", "1"],
            "course_classification": ["Lab", "Lecture"],
        }
    )
    spec = KeyCollisionDedupeConfig(
        key_cols=["student_id", "term", "class_number"],
        conflict_columns=["course_classification"],
        disambiguate_column="class_number",
    ).model_dump(exclude_none=True)
    out = d.apply_key_collision_dedupe_from_spec(df, spec)
    assert len(out) == 2
    assert out["class_number"].astype(str).nunique() == 2


def test_resolve_key_collisions_drops_identical_key_copies():
    df = pd.DataFrame(
        {
            "k": [1, 1],
            "x": [9, 9],
            "course_classification": ["Lec", "Lec"],
        }
    )
    out = d.resolve_key_collisions(
        df,
        key_cols=["k"],
        conflict_columns=["course_classification"],
        disambiguate_column="x",
    )
    assert len(out) == 1
