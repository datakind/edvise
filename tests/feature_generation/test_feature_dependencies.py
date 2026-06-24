import pandas as pd

from edvise.feature_generation.column_names import (
    ES_COURSE_INPUT_COLUMNS,
    CourseFeatureSpec,
    SectionFeatureSpec,
    StudentTermAddFeatureSpec,
)
from edvise.feature_generation.feature_dependencies import (
    MULTICOL_GRADE_COLUMNS,
    columns_present,
    enable_when,
    multicol_grade_enabled,
)
from edvise.feature_generation.feature_resolution import (
    resolve_course_feature_spec,
    resolve_section_feature_spec,
    resolve_student_term_aggregate_spec,
)
from edvise.feature_generation import student_term


def test_multicol_grade_disabled_without_section_id():
    df = pd.DataFrame(
        {
            "course_prefix": ["MATH"],
            "course_number": ["101"],
            "grade": ["A"],
        }
    )
    course = resolve_course_feature_spec(df, ES_COURSE_INPUT_COLUMNS)
    section = resolve_section_feature_spec(df, ES_COURSE_INPUT_COLUMNS)
    agg = resolve_student_term_aggregate_spec(
        df,
        ES_COURSE_INPUT_COLUMNS,
        course_flags=course,
        section_flags=section,
    )
    assert course.course_grade_numeric is True
    assert section.section_course_grade_numeric_mean is False
    assert agg.multicol_grade is False


def test_multicol_grade_enabled_with_full_section_stack():
    df = pd.DataFrame(
        {
            "course_prefix": ["MATH"],
            "course_number": ["101"],
            "grade": ["A"],
            "course_section_id": ["S1"],
        }
    )
    course = resolve_course_feature_spec(df, ES_COURSE_INPUT_COLUMNS)
    section = resolve_section_feature_spec(df, ES_COURSE_INPUT_COLUMNS)
    agg = resolve_student_term_aggregate_spec(
        df,
        ES_COURSE_INPUT_COLUMNS,
        course_flags=course,
        section_flags=section,
    )
    assert multicol_grade_enabled(course_flags=course, section_flags=section) is True
    assert agg.multicol_grade is True


def test_aggregate_skips_missing_department_columns():
    df = pd.DataFrame(
        {
            "learner_id": ["s1", "s1"],
            "term_id": ["t1", "t1"],
            "institution_id": ["i1", "i1"],
            "academic_year": ["20-21", "20-21"],
            "academic_term": ["FALL", "FALL"],
            "term_start_dt": pd.to_datetime(["2020-09-01", "2020-09-01"]),
            "term_rank": [1, 1],
            "term_rank_core": [1, 1],
            "term_rank_noncore": [1, 1],
            "term_is_core": [True, True],
            "term_is_noncore": [False, False],
            "term_in_peak_covid": [False, False],
            "term_degree": ["A.S.", "A.S."],
            "course_id": ["c1", "c2"],
            "course_passed": [True, True],
            "course_completed": [True, True],
            "course_credits_attempted": [3.0, 3.0],
            "course_credits_earned": [3.0, 3.0],
        }
    )
    out = student_term.aggregate_from_course_level_features(
        df,
        student_term_id_cols=["learner_id", "term_id"],
        cols=ES_COURSE_INPUT_COLUMNS,
    )
    assert "course_subject_areas" not in out.columns
    assert "num_courses" in out.columns


def test_multicol_grade_aggs_without_section_mean_column():
    df = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "term_id": ["t1"],
            "grade": ["F"],
            "course_grade_numeric": [0.0],
        }
    )
    out = student_term.multicol_grade_aggs_by_group(
        df,
        min_passing_grade=1.0,
        grp_cols=["learner_id", "term_id"],
    )
    assert "num_courses_grade_is_failing_or_withdrawal" in out.columns
    assert "num_courses_grade_above_section_avg" not in out.columns


def test_enable_when_requires_all_columns():
    df = pd.DataFrame({"a": [1]})
    assert columns_present(df, "a") is True
    assert columns_present(df, "a", "b") is False
    assert enable_when(True, df, "a") is True
    assert enable_when(True, df, "a", "b") is False
    assert enable_when(False, df, "a") is False


def test_add_features_program_area_uses_canonical_term_program_of_study():
    """Aggregation renames ES ``term_degree`` to ``term_program_of_study``."""
    df = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "term_id": ["t1"],
            "term_program_of_study": ["24.0101"],
            "course_subject_areas": [["24", "27"]],
            "num_courses": [2],
        }
    )
    spec = StudentTermAddFeatureSpec(
        year_of_enrollment_at_cohort_inst=False,
        student_certificates=False,
        term_is_pre_cohort=False,
        term_is_while_student_enrolled_at_other_inst=False,
        program_of_study_area=True,
        credit_fraction_and_intensity=False,
        num_courses_in_program_area=True,
        num_course_by_category_fracs=False,
        section_student_fractions=False,
        student_rate_vs_section_fractions=False,
        program_change_from_prior_term=False,
    )
    out = student_term.add_features(
        df,
        cols=ES_COURSE_INPUT_COLUMNS,
        min_num_credits_full_time=12.0,
        spec=spec,
    )
    assert out["term_program_of_study_area"].iloc[0] == "24"
    assert out["num_courses_in_term_program_of_study_area"].iloc[0] == 1


def test_multicol_grade_columns_constant_matches_resolution():
    assert MULTICOL_GRADE_COLUMNS == (
        "grade",
        "course_grade_numeric",
        "section_course_grade_numeric_mean",
    )
    assert CourseFeatureSpec.all().course_grade
    assert SectionFeatureSpec.all().section_course_grade_numeric_mean
