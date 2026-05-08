"""
Auto-resolve feature specs from :class:`CohortInputColumns` / :class:`CourseInputColumns`
plus frame column presence, with optional per-field overrides (``dict`` of field name -> bool).
"""

from __future__ import annotations

import dataclasses
import typing as t

import pandas as pd

from .column_names import (
    CohortInputColumns,
    CourseInputColumns,
    CumulativeExpandingColumnSpec,
    CumulativeFeatureSpec,
    CourseFeatureSpec,
    SectionFeatureSpec,
    StudentFeatureSpec,
    StudentTermAddFeatureSpec,
    StudentTermAggregateSpec,
    TermFeatureSpec,
    ES_STUDENT_FEATURE_SPEC_DEFAULT,
)

_TSpec = t.TypeVar("_TSpec")


def has_data_col(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in df.columns


def merge_spec_fields(
    base: _TSpec, patch: t.Mapping[str, bool] | None
) -> _TSpec:
    """``patch`` keys replace same-named fields on ``base`` (typically a frozen dataclass)."""
    if not patch:
        return base
    valid = {k: v for k, v in patch.items() if hasattr(base, k)}
    return dataclasses.replace(base, **valid) if valid else base


def resolve_student_feature_spec(
    df: pd.DataFrame,
    cols: CohortInputColumns,
    *,
    base: StudentFeatureSpec = ES_STUDENT_FEATURE_SPEC_DEFAULT,
) -> StudentFeatureSpec:
    has_cohort_keys = has_data_col(
        df, cols.cohort_year_col
    ) and has_data_col(df, cols.cohort_term_col)
    has_pell = has_data_col(df, cols.pell_status_col)
    has_gpa = (
        cols.gpa_group_term_1_col is not None
        and cols.gpa_group_year_1_col is not None
        and has_data_col(df, cols.gpa_group_term_1_col)
        and has_data_col(df, cols.gpa_group_year_1_col)
    )
    has_frac_credits = False
    if (
        cols.credits_earned_year_template
        and cols.credits_attempted_year_template
    ):
        for yr in (1, 2, 3, 4):
            e = cols.earned_col(yr)
            a = cols.attempted_col(yr)
            if e in df.columns and a in df.columns:
                has_frac_credits = True
                break
    return dataclasses.replace(
        base,
        cohort_id=has_cohort_keys,
        cohort_start_dt=has_cohort_keys,
        pell=has_pell,
        diff_gpa=has_gpa,
        frac_credits_by_year=has_frac_credits,
    )


def resolve_course_feature_spec(
    df: pd.DataFrame, cols: CourseInputColumns
) -> CourseFeatureSpec:
    g = has_data_col(df, cols.grade)
    cip = has_data_col(df, cols.course_cip)
    pfx = has_data_col(df, cols.course_prefix)
    num = has_data_col(df, cols.course_number)
    return CourseFeatureSpec(
        course_id=pfx and num,
        course_subject_area=cip,
        course_passed=g,
        course_completed=g,
        course_level=bool(num and g),
        course_grade_numeric=g,
        course_grade=g,
    )


def resolve_term_feature_spec(df: pd.DataFrame, cols: CourseInputColumns) -> TermFeatureSpec:
    has_ay = has_data_col(df, cols.academic_year)
    has_at = has_data_col(df, cols.academic_term)
    term_on = has_ay and has_at
    return TermFeatureSpec(
        term_id=term_on,
        term_start_dt=term_on,
        term_rank=term_on,
        term_rank_core=term_on,
        term_rank_noncore=term_on,
        term_in_peak_covid=term_on,
        term_is_core=term_on,
        term_is_noncore=term_on,
    )


def resolve_section_feature_spec(
    df: pd.DataFrame, cols: CourseInputColumns
) -> SectionFeatureSpec:
    g = has_data_col(df, cols.grade)
    pfx = has_data_col(df, cols.course_prefix)
    num = has_data_col(df, cols.course_number)
    sec_id = has_data_col(df, cols.section_id)
    base = sec_id and pfx and num and g
    return SectionFeatureSpec(
        section_num_students_enrolled=base,
        section_num_students_passed=base,
        section_num_students_completed=base,
        section_course_grade_numeric_mean=base,
    )


def _optional_course_dummy_sources(df: pd.DataFrame, cols: CourseInputColumns) -> list[str]:
    out: list[str] = []
    for attr in (
        "course_type",
        "co_requisite_course",
        "course_instructor_rank",
    ):
        name: str | None = getattr(cols, attr, None)
        if name and name in df.columns:
            out.append(name)
    for attr in ("delivery_method", "math_or_english_gateway", "course_instructor_employment_status", "core_course"):
        name = getattr(cols, attr)
        if name in df.columns:
            out.append(name)
    return out


def resolve_student_term_aggregate_spec(
    df: pd.DataFrame,
    cols: CourseInputColumns,
    *,
    course_flags: CourseFeatureSpec,
) -> StudentTermAggregateSpec:
    g = has_data_col(df, cols.grade)
    pfx = has_data_col(df, cols.course_prefix)
    num = has_data_col(df, cols.course_number)
    dummy_from_cols = bool(_optional_course_dummy_sources(df, cols))
    dummies = dummy_from_cols or (g and num and course_flags.course_level and course_flags.course_grade)
    has_ct = has_data_col(df, cols.course_type) if cols.course_type else False
    has_core = has_data_col(df, cols.core_course)
    has_oth = (
        has_data_col(df, cols.enrolled_at_other_institution_s)
        if cols.enrolled_at_other_institution_s
        else False
    )
    value_equality = has_core or has_ct or has_oth
    return StudentTermAggregateSpec(
        summary_aggregations=True,
        dummies=dummies,
        value_equality=bool(value_equality),
        multicol_grade=bool(g),
    )


def resolve_student_term_add_feature_spec(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    cohort_cols: CohortInputColumns,
    course_cols: CourseInputColumns,
    *,
    has_cohort_keys: bool,
    term_on: bool,
    section_enrolled: bool,
    g: bool,
) -> StudentTermAddFeatureSpec:
    c_cert = bool(
        cohort_cols.first_year_to_certificate_at_cohort_inst
        and cohort_cols.years_to_latest_certificate_at_cohort_inst
        and has_data_col(df_cohort, cohort_cols.first_year_to_certificate_at_cohort_inst)
        and has_data_col(df_cohort, cohort_cols.years_to_latest_certificate_at_cohort_inst)
    )
    o_cert = bool(
        cohort_cols.first_year_to_certificate_at_other_inst
        and cohort_cols.years_to_latest_certificate_at_other_inst
        and has_data_col(df_cohort, cohort_cols.first_year_to_certificate_at_other_inst)
        and has_data_col(df_cohort, cohort_cols.years_to_latest_certificate_at_other_inst)
    )
    will_have_cohort_start = has_cohort_keys
    has_term_prog = has_data_col(df_course, course_cols.term_program_of_study)
    has_cr = has_data_col(
        df_course, course_cols.number_of_credits_earned
    ) and has_data_col(df_course, course_cols.number_of_credits_attempted)
    has_oth = (
        has_data_col(df_course, course_cols.enrolled_at_other_institution_s)
        if course_cols.enrolled_at_other_institution_s
        else False
    )
    return StudentTermAddFeatureSpec(
        year_of_enrollment_at_cohort_inst=will_have_cohort_start,
        student_certificates=will_have_cohort_start and (c_cert or o_cert),
        term_is_pre_cohort=will_have_cohort_start,
        term_is_while_student_enrolled_at_other_inst=bool(
            will_have_cohort_start and has_oth
        ),
        program_of_study_area=has_term_prog,
        credit_fraction_and_intensity=has_cr,
        num_courses_in_program_area=has_term_prog,
        num_course_by_category_fracs=True,
        section_student_fractions=bool(section_enrolled and g),
        student_rate_vs_section_fractions=bool(section_enrolled and g),
        program_change_from_prior_term=bool(has_term_prog and term_on),
    )


def resolve_cumulative_feature_spec(
    _df: pd.DataFrame,
    course_cols: CourseInputColumns,
    *,
    will_have_cohort_start: bool,
    term_on: bool,
    section_enrolled: bool,
    g: bool,
    num: bool,
    has_cr: bool,
    has_oth: bool,
) -> CumulativeFeatureSpec:
    exp = CumulativeExpandingColumnSpec(
        term_id=term_on,
        term_in_peak_covid=term_on,
        term_is_core=term_on,
        term_is_noncore=term_on,
        term_is_while_student_enrolled_at_other_inst=bool(has_oth and term_on),
        term_is_pre_cohort=bool(will_have_cohort_start and term_on),
        course_level_mean=bool(num and g),
        course_grade_numeric_mean=bool(g),
        num_courses=True,
        num_credits_attempted=has_cr,
        num_credits_earned=has_cr,
        student_pass_rate_above_sections_avg=bool(section_enrolled and g),
        student_completion_rate_above_sections_avg=bool(section_enrolled and g),
    )
    return CumulativeFeatureSpec(
        expanding_aggregate=True,
        expanding_columns=exp,
        cumnum_unique_repeated=True,
        cumfrac_terms_enrolled=term_on,
        term_differences=term_on,
    )
