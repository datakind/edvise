import pandas as pd

from edvise.dataio.es_course_converters import handle_missing_grades


def test_dup_null_earned_kept_as_m_fills_credits():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1"],
            "course_prefix": ["ENG", "ENG"],
            "course_number": ["101", "101-1"],
            "academic_year": ["2020-21", "2021-22"],
            "academic_term": ["FALL", "FALL"],
            "grade": [pd.NA, "A"],
            "course_credits_earned": [pd.NA, 3.0],
            "course_credits_attempted": [pd.NA, 3.0],
        }
    )
    out = handle_missing_grades(df)
    assert list(out.index) == [0, 1]
    assert out.loc[0, "grade"] == "M"
    assert float(out.loc[0, "course_credits_earned"]) == 0.0
    assert float(out.loc[0, "course_credits_attempted"]) == 0.0
    assert out.loc[1, "course_number"] == "101-1"  # suffix not stripped on store


def test_all_null_zero_earned_siblings_keep_first_only():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1"],
            "course_prefix": ["ENG", "ENG"],
            "course_number": ["101", "101"],
            "grade": [pd.NA, pd.NA],
            "course_credits_earned": [pd.NA, 0.0],
            "course_credits_attempted": [pd.NA, pd.NA],
        }
    )
    out = handle_missing_grades(df)
    assert list(out.index) == [0]
    assert out.loc[0, "grade"] == "M"


def test_unique_null_grade_dropped():
    df = pd.DataFrame(
        {
            "learner_id": ["1"],
            "course_prefix": ["ENG"],
            "course_number": ["101"],
            "grade": [pd.NA],
            "course_credits_earned": [3.0],
            "course_credits_attempted": [pd.NA],
        }
    )
    assert handle_missing_grades(df).empty


def test_reproducible_same_input_same_output():
    df = pd.DataFrame(
        {
            "learner_id": ["1", "1", "2"],
            "course_prefix": ["ENG", "ENG", "MAT"],
            "course_number": ["101", "101-2", "200"],
            "grade": [pd.NA, "B", pd.NA],
            "course_credits_earned": [2.0, 3.0, 0.0],
            "course_credits_attempted": [pd.NA, 3.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(handle_missing_grades(df), handle_missing_grades(df))
