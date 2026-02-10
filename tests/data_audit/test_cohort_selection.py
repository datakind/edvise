"""Tests for edvise.data_audit.cohort_selection.select_inference_cohort."""

import pandas as pd
import pytest

from edvise.data_audit.cohort_selection import select_inference_cohort


def test_select_inference_cohort_single_match():
    """Single cohort in list matches one row; returns filtered DataFrame without cohort_selection column."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL", "SPRING"],
            "cohort": ["2024-25", "2023-24"],
            "id": [1, 2],
        }
    )
    result = select_inference_cohort(df, cohorts_list=["fall 2024-25"])
    assert len(result) == 1
    assert result["id"].iloc[0] == 1
    assert "cohort_selection" not in result.columns


def test_select_inference_cohort_multi_cohort():
    """Multiple cohorts in list; union of matching rows returned."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL", "SPRING", "FALL"],
            "cohort": ["2024-25", "2024-25", "2023-24"],
            "id": [1, 2, 3],
        }
    )
    result = select_inference_cohort(
        df, cohorts_list=["fall 2024-25", "spring 2024-25"]
    )
    assert len(result) == 2
    assert set(result["id"]) == {1, 2}
    assert "cohort_selection" not in result.columns


def test_select_inference_cohort_case_insensitive():
    """cohorts_list with mixed case matches data built lowercase (case-insensitive)."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    result = select_inference_cohort(df, cohorts_list=["Fall 2024-25"])
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_select_inference_cohort_empty_list_raises():
    """Empty cohorts_list raises ValueError with message about no non-empty labels."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(ValueError, match="cohorts_list had no non-empty cohort labels"):
        select_inference_cohort(df, cohorts_list=[])


def test_select_inference_cohort_whitespace_only_raises():
    """cohorts_list with only whitespace raises ValueError."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(ValueError, match="cohorts_list had no non-empty cohort labels"):
        select_inference_cohort(df, cohorts_list=["  ", "\t", ""])


def test_select_inference_cohort_no_match_raises():
    """Valid list but no rows match raises ValueError (empty DataFrames)."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(
        ValueError, match="Selected cohorts resulted in empty DataFrames"
    ):
        select_inference_cohort(df, cohorts_list=["spring 2023-24"])


def test_select_inference_cohort_custom_columns():
    """Custom cohort_term_column and cohort_column are used."""
    df = pd.DataFrame(
        {
            "term": ["FALL", "SPRING"],
            "cohort_yr": ["2024-25", "2023-24"],
            "id": [1, 2],
        }
    )
    result = select_inference_cohort(
        df,
        cohorts_list=["fall 2024-25"],
        cohort_term_column="term",
        cohort_column="cohort_yr",
    )
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_select_inference_cohort_drops_cohort_selection_column():
    """Output DataFrame does not contain the temporary cohort_selection column."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    result = select_inference_cohort(df, cohorts_list=["fall 2024-25"])
    assert "cohort_selection" not in result.columns
