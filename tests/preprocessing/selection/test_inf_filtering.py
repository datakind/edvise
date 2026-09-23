"""Tests for edvise.student_selection.filter_inference functions."""

import pandas as pd
import pytest

from edvise.student_selection.filter_inference import (
    _filter_by_joined_columns,
    exclude_training_cohort_students,
    filter_inference_cohort,
    filter_inference_term,
    graduation_open_window,
    latest_as_of_term,
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


# Tests for exclude_training_cohort_students
def test_exclude_training_cohort_students_removes_stop_outs():
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL", "FALL", "SPRING"],
            "cohort": ["2023-24", "2024-25", "2024-25"],
            "id": [1, 2, 3],
        }
    )
    result = exclude_training_cohort_students(
        df,
        training_cohorts=["fall 2023-24"],
    )
    assert set(result["id"]) == {2, 3}


def test_exclude_training_cohort_students_no_training_cohorts():
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2024-25"],
            "id": [1],
        }
    )
    result = exclude_training_cohort_students(df, training_cohorts=[])
    assert len(result) == 1


def test_exclude_training_cohort_students_missing_columns_returns_unchanged():
    df = pd.DataFrame({"id": [1]})
    result = exclude_training_cohort_students(
        df,
        training_cohorts=["fall 2024-25"],
    )
    assert len(result) == 1


def test_exclude_training_cohort_students_all_excluded_raises():
    df = pd.DataFrame(
        {
            "cohort_term": ["FALL"],
            "cohort": ["2023-24"],
            "id": [1],
        }
    )
    with pytest.raises(
        ValueError,
        match="Excluding training cohort students resulted in empty DataFrame",
    ):
        exclude_training_cohort_students(
            df,
            training_cohorts=["fall 2023-24"],
        )


_INTENSITY_LIMITS = {
    "FULL-TIME": (3.0, "year"),
    "PART-TIME": (4.5, "year"),
}


def _checkpoint_row(
    *,
    student_id: int,
    cohort_term: str,
    cohort: str,
    intensity: str,
    ckpt_term: str = "SPRING",
    ckpt_year: str = "2025-26",
) -> dict:
    return {
        "id": student_id,
        "cohort_term": cohort_term,
        "cohort": cohort,
        "student_term_enrollment_intensity": intensity,
        "academic_term": ckpt_term,
        "academic_year": ckpt_year,
    }


def test_latest_as_of_term_picks_latest_on_two_term_calendar():
    assert (
        latest_as_of_term(
            ["fall 2024-25", "spring 2025-26", "spring 2024-25"],
            num_terms_in_year=2,
        )
        == "spring 2025-26"
    )


def test_exclude_training_cohort_keeps_in_window_part_time():
    """As of spring 2025-26 a shared cohort drops labelable FT and keeps open PT."""
    df = pd.DataFrame(
        [
            # FT fall 2023-24: 6 terms = 3.0 years → labelable
            _checkpoint_row(
                student_id=1,
                cohort_term="FALL",
                cohort="2023-24",
                intensity="FULL-TIME",
            ),
            # PT fall 2023-24: 3.0 years < 4.5 → still open, same training cohort
            _checkpoint_row(
                student_id=2,
                cohort_term="FALL",
                cohort="2023-24",
                intensity="PART-TIME",
            ),
            # PT spring 2021-22: 9 terms = 4.5 years → labelable
            _checkpoint_row(
                student_id=3,
                cohort_term="SPRING",
                cohort="2021-22",
                intensity="PART-TIME",
            ),
            # Labelable, but this cohort was not used in training
            _checkpoint_row(
                student_id=4,
                cohort_term="FALL",
                cohort="2021-22",
                intensity="FULL-TIME",
            ),
        ]
    )
    result = exclude_training_cohort_students(
        df,
        training_cohorts=["fall 2023-24", "spring 2021-22"],
        as_of_term="spring 2025-26",
        intensity_time_limits=_INTENSITY_LIMITS,
        num_terms_in_year=2,
    )
    assert set(result["id"]) == {2, 4}


def test_exclude_training_cohort_unknown_intensity_stays_excluded():
    df = pd.DataFrame(
        [
            _checkpoint_row(
                student_id=1,
                cohort_term="FALL",
                cohort="2024-25",
                intensity="FULL-TIME",
            ),
            {
                "id": 2,
                "cohort_term": "FALL",
                "cohort": "2024-25",
                "student_term_enrollment_intensity": pd.NA,
                "academic_term": "SPRING",
                "academic_year": "2025-26",
            },
        ]
    )
    result = exclude_training_cohort_students(
        df,
        training_cohorts=["fall 2024-25"],
        as_of_term="spring 2025-26",
        intensity_time_limits=_INTENSITY_LIMITS,
        num_terms_in_year=2,
    )
    assert set(result["id"]) == {1}


def test_graduation_open_window_only_for_graduation():
    graduation = type(
        "Preprocessing",
        (),
        {
            "target": type(
                "Target",
                (),
                {
                    "type_": "graduation",
                    "intensity_time_limits": _INTENSITY_LIMITS,
                    "num_terms_in_year": 2,
                },
            )()
        },
    )()
    credits = type(
        "Preprocessing",
        (),
        {
            "target": type(
                "Target",
                (),
                {"type_": "credits_earned", "intensity_time_limits": _INTENSITY_LIMITS},
            )()
        },
    )()
    assert graduation_open_window(graduation, ["spring 2025-26"])["as_of_term"] == (
        "spring 2025-26"
    )
    assert graduation_open_window(credits, ["spring 2025-26"]) == {}
