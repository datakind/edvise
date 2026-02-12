"""Tests for edvise.student_selection.filter_inference functions."""

import pandas as pd
import pytest

from edvise.student_selection.filter_inference import (
    _filter_by_joined_columns,
    filter_inference_cohort,
    filter_inference_term,
)


# Tests for _filter_by_joined_columns
def test_filter_by_joined_columns_single_match():
    """Single value in list matches one row; returns filtered DataFrame without temp column."""
    df = pd.DataFrame(
        {
            "first": ["FALL", "SPRING"],
            "second": ["2024-25", "2023-24"],
            "id": [1, 2],
        }
    )
    result = _filter_by_joined_columns(
        df,
        selection_list=["fall 2024-25"],
        first_column="first",
        second_column="second",
        selection_type="test",
    )
    assert len(result) == 1
    assert result["id"].iloc[0] == 1
    assert "test_selection" not in result.columns


def test_filter_by_joined_columns_multi_value():
    """Multiple values in list; union of matching rows returned."""
    df = pd.DataFrame(
        {
            "first": ["FALL", "SPRING", "FALL"],
            "second": ["2024-25", "2024-25", "2023-24"],
            "id": [1, 2, 3],
        }
    )
    result = _filter_by_joined_columns(
        df,
        selection_list=["fall 2024-25", "spring 2024-25"],
        first_column="first",
        second_column="second",
        selection_type="test",
    )
    assert len(result) == 2
    assert set(result["id"]) == {1, 2}
    assert "test_selection" not in result.columns


def test_filter_by_joined_columns_case_insensitive():
    """Selection list with mixed case matches data built lowercase (case-insensitive)."""
    df = pd.DataFrame(
        {
            "first": ["FALL"],
            "second": ["2024-25"],
            "id": [1],
        }
    )
    result = _filter_by_joined_columns(
        df,
        selection_list=["Fall 2024-25"],
        first_column="first",
        second_column="second",
        selection_type="test",
    )
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_filter_by_joined_columns_empty_list_raises():
    """Empty selection_list raises ValueError with message about no non-empty labels."""
    df = pd.DataFrame(
        {
            "first": ["FALL"],
            "second": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(ValueError, match="test_list had no non-empty test labels"):
        _filter_by_joined_columns(
            df,
            selection_list=[],
            first_column="first",
            second_column="second",
            selection_type="test",
        )


def test_filter_by_joined_columns_whitespace_only_raises():
    """Selection list with only whitespace raises ValueError."""
    df = pd.DataFrame(
        {
            "first": ["FALL"],
            "second": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(ValueError, match="test_list had no non-empty test labels"):
        _filter_by_joined_columns(
            df,
            selection_list=["  ", "\t", ""],
            first_column="first",
            second_column="second",
            selection_type="test",
        )


def test_filter_by_joined_columns_no_match_raises():
    """Valid list but no rows match raises ValueError (empty DataFrame)."""
    df = pd.DataFrame(
        {
            "first": ["FALL"],
            "second": ["2024-25"],
            "id": [1],
        }
    )
    with pytest.raises(ValueError, match="Filtered test resulted in empty DataFrame"):
        _filter_by_joined_columns(
            df,
            selection_list=["spring 2023-24"],
            first_column="first",
            second_column="second",
            selection_type="test",
        )


def test_filter_by_joined_columns_drops_temp_column():
    """Output DataFrame does not contain the temporary selection column."""
    df = pd.DataFrame(
        {
            "first": ["FALL"],
            "second": ["2024-25"],
            "id": [1],
        }
    )
    result = _filter_by_joined_columns(
        df,
        selection_list=["fall 2024-25"],
        first_column="first",
        second_column="second",
        selection_type="test",
    )
    assert "test_selection" not in result.columns


def test_filter_by_joined_columns_preserves_original():
    """Original dataframe is not modified by filtering."""
    df = pd.DataFrame(
        {
            "first": ["FALL", "SPRING"],
            "second": ["2024-25", "2023-24"],
            "id": [1, 2],
        }
    )
    original_len = len(df)
    result = _filter_by_joined_columns(
        df,
        selection_list=["fall 2024-25"],
        first_column="first",
        second_column="second",
        selection_type="test",
    )
    assert len(df) == original_len  # Original unchanged
    assert len(result) == 1  # Result is filtered


# Tests for filter_inference_cohort wrapper
def test_filter_inference_cohort_uses_correct_defaults():
    """Wrapper uses correct default column names."""
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    result = filter_inference_cohort(df, cohorts_list=["fall 2024-25"])
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_filter_inference_cohort_custom_columns():
    """Wrapper accepts custom column names."""
    df = pd.DataFrame(
        {
            "term": ["FALL"],
            "cohort_yr": ["2024-25"],
            "id": [1],
        }
    )
    result = filter_inference_cohort(
        df,
        cohorts_list=["fall 2024-25"],
        cohort_term_column="term",
        cohort_column="cohort_yr",
    )
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


# Tests for filter_inference_term wrapper
def test_filter_inference_term_uses_correct_defaults():
    """Wrapper uses correct default column names."""
    df = pd.DataFrame(
        {
            "academic_term": ["FALL"],
            "academic_year": ["2024-25"],
            "id": [1],
        }
    )
    result = filter_inference_term(df, term_list=["fall 2024-25"])
    assert len(result) == 1
    assert result["id"].iloc[0] == 1


def test_filter_inference_term_custom_columns():
    """Wrapper accepts custom column names."""
    df = pd.DataFrame(
        {
            "term": ["FALL"],
            "year": ["2024-25"],
            "id": [1],
        }
    )
    result = filter_inference_term(
        df,
        term_list=["fall 2024-25"],
        academic_term_col="term",
        academic_year_col="year",
    )
    assert len(result) == 1
    assert result["id"].iloc[0] == 1
