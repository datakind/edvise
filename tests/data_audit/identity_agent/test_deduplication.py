import pandas as pd
import pytest

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


def test_suffix_disambiguate_int64_target_column():
    """Training dtypes often cast class_number to Int64 before dedupe; suffix must still work."""
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["FA", "FA"],
            "class_number": pd.Series([37559, 37559], dtype="Int64"),
        }
    )
    out = d.suffix_disambiguate_within_keys(df, ["sid", "term"], "class_number")
    assert str(out["class_number"].dtype) == "string"
    assert set(out["class_number"].astype(str)) == {"37559-1", "37559-2"}


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


def test_resolve_key_collisions_int64_disambiguate_column():
    df = pd.DataFrame(
        {
            "student_id": ["u1", "u1"],
            "course_name": ["CHEM-207-1", "CHEM-207-1"],
            "term": ["2019FA", "2019FA"],
            "class_number": pd.Series([37559, 37559], dtype="Int64"),
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
    assert str(out["class_number"].dtype) == "string"


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
