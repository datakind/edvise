import pandas as pd
import pytest

from edvise.feature_generation.cip_columns import (
    column_cip_match_fraction,
    has_sufficient_cip_values,
    is_cip_like_value,
    resolve_term_program_of_study_source,
)
from edvise.feature_generation.column_names import (
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.feature_resolution import (
    resolve_course_feature_spec,
    resolve_student_term_add_feature_spec,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("52.0301", True),
        ("24.01", True),
        ("Undergraduate", False),
        ("Marketing", False),
        ("A.S.", False),
        (None, False),
    ],
)
def test_is_cip_like_value(value, expected):
    assert is_cip_like_value(value) is expected


def test_resolve_term_program_of_study_source_prefers_declared_major_cip():
    df = pd.DataFrame(
        {
            "term_degree": ["Undergraduate", "Undergraduate"],
            "term_declared_major": ["52.0301", "52.0301"],
        }
    )
    assert (
        resolve_term_program_of_study_source(df, ES_COURSE_INPUT_COLUMNS)
        == "term_declared_major"
    )


def test_resolve_term_program_of_study_source_returns_none_for_non_cip():
    df = pd.DataFrame({"term_degree": ["Undergraduate", "Other"]})
    assert resolve_term_program_of_study_source(df, ES_COURSE_INPUT_COLUMNS) is None


def test_resolve_course_feature_spec_skips_subject_area_without_cip():
    df = pd.DataFrame(
        {
            "grade": ["A"],
            "course_prefix": ["MATH"],
            "course_number": ["101"],
            "department": ["Marketing"],
        }
    )
    spec = resolve_course_feature_spec(df, ES_COURSE_INPUT_COLUMNS)
    assert spec.course_subject_area is False


def test_resolve_student_term_add_skips_program_features_for_term_degree_only():
    df_course = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "academic_year": ["2020-21"],
            "academic_term": ["FALL"],
            "term_degree": ["Undergraduate"],
            "department": ["Marketing"],
            "grade": ["A"],
            "course_credits_attempted": [3.0],
            "course_credits_earned": [3.0],
        }
    )
    df_cohort = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "entry_year": ["2020-21"],
            "entry_term": ["FALL"],
        }
    )
    spec = resolve_student_term_add_feature_spec(
        df_cohort,
        df_course,
        cohort_cols=ES_COHORT_INPUT_COLUMNS,
        course_cols=ES_COURSE_INPUT_COLUMNS,
        has_cohort_keys=True,
        term_on=True,
        section_enrolled=False,
        g=True,
    )
    assert spec.program_of_study_area is False
    assert spec.num_courses_in_program_area is False
    assert spec.program_change_from_prior_term is False
    assert spec.term_degree_changed_prev_term is True


def test_column_cip_match_fraction():
    ser = pd.Series(["52.0301", "Undergraduate", None])
    assert column_cip_match_fraction(ser) == pytest.approx(1 / 2)
    assert has_sufficient_cip_values(pd.DataFrame({"x": ["52.0301", "52.0201"]}), "x")
