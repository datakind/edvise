"""
Build feature-generation specs from :class:`CohortInputColumns` / :class:`CourseInputColumns`
and column presence, with optional per-field overrides.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from .column_names import (
    CohortInputColumns,
    CourseInputColumns,
    CumulativeFeatureSpec,
    CourseFeatureSpec,
    SectionFeatureSpec,
    StudentFeatureSpec,
    StudentTermAddFeatureSpec,
    StudentTermAggregateSpec,
    TermFeatureSpec,
    ES_STUDENT_FEATURE_SPEC_DEFAULT,
)
from .feature_resolution import (
    has_data_col,
    merge_spec_fields,
    resolve_course_feature_spec,
    resolve_cumulative_feature_spec,
    resolve_section_feature_spec,
    resolve_student_feature_spec,
    resolve_student_term_add_feature_spec,
    resolve_student_term_aggregate_spec,
    resolve_term_feature_spec,
)


@dataclass(frozen=True, slots=True)
class EdviseFeatureSpecBundle:
    """Auto-resolved + optionally overridden feature specs for one Edvise run."""

    student: StudentFeatureSpec
    course: CourseFeatureSpec
    term: TermFeatureSpec
    section: SectionFeatureSpec
    student_term_aggregate: StudentTermAggregateSpec
    student_term_add: StudentTermAddFeatureSpec
    cumulative: CumulativeFeatureSpec


@dataclass(frozen=True, slots=True)
class EdviseFeatureSpecOverrides:
    """
    Optional per-field overrides for auto-resolved specs.

    Each mapping uses **dataclass field names** -> bool. Omitted keys keep the
    auto-resolved value.
    """

    student: dict[str, bool] | None = None
    course: dict[str, bool] | None = None
    term: dict[str, bool] | None = None
    section: dict[str, bool] | None = None
    student_term_aggregate: dict[str, bool] | None = None
    student_term_add: dict[str, bool] | None = None
    cumulative: dict[str, bool] | None = None
    cumulative_expanding_columns: dict[str, bool] | None = None


def build_edvise_feature_specs(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    cohort_cols: CohortInputColumns,
    course_cols: CourseInputColumns,
    overrides: EdviseFeatureSpecOverrides | None = None,
) -> EdviseFeatureSpecBundle:
    """
    Resolve specs from ``cols`` + frame columns, then apply ``overrides`` (if any).
    """
    o = overrides or EdviseFeatureSpecOverrides()

    has_cohort_keys = has_data_col(
        df_cohort, cohort_cols.cohort_year_col
    ) and has_data_col(df_cohort, cohort_cols.cohort_term_col)
    g = has_data_col(df_course, course_cols.grade)
    num = has_data_col(df_course, course_cols.course_number)
    has_cr = has_data_col(
        df_course, course_cols.number_of_credits_earned
    ) and has_data_col(df_course, course_cols.number_of_credits_attempted)
    has_oth = (
        has_data_col(df_course, course_cols.enrolled_at_other_institution_s)
        if course_cols.enrolled_at_other_institution_s
        else False
    )

    student = merge_spec_fields(
        resolve_student_feature_spec(df_cohort, cohort_cols, base=ES_STUDENT_FEATURE_SPEC_DEFAULT),
        o.student,
    )
    course = merge_spec_fields(
        resolve_course_feature_spec(df_course, course_cols),
        o.course,
    )
    term = merge_spec_fields(
        resolve_term_feature_spec(df_course, course_cols),
        o.term,
    )
    section = merge_spec_fields(
        resolve_section_feature_spec(df_course, course_cols),
        o.section,
    )
    st_agg = merge_spec_fields(
        resolve_student_term_aggregate_spec(
            df_course,
            course_cols,
            course_flags=course,
        ),
        o.student_term_aggregate,
    )
    term_on = term.term_id
    st_add = merge_spec_fields(
        resolve_student_term_add_feature_spec(
            df_cohort,
            df_course,
            cohort_cols,
            course_cols,
            has_cohort_keys=has_cohort_keys,
            term_on=term_on,
            section_enrolled=bool(section.section_num_students_enrolled),
            g=g,
        ),
        o.student_term_add,
    )
    cum = resolve_cumulative_feature_spec(
        df_course,
        course_cols,
        will_have_cohort_start=has_cohort_keys,
        term_on=term_on,
        section_enrolled=bool(section.section_num_students_enrolled),
        g=g,
        num=num,
        has_cr=has_cr,
        has_oth=has_oth,
    )
    cum = merge_spec_fields(cum, o.cumulative)
    if o.cumulative_expanding_columns:
        new_exp = merge_spec_fields(
            cum.expanding_columns, o.cumulative_expanding_columns
        )
        cum = dataclasses.replace(cum, expanding_columns=new_exp)

    return EdviseFeatureSpecBundle(
        student=student,
        course=course,
        term=term,
        section=section,
        student_term_aggregate=st_agg,
        student_term_add=st_add,
        cumulative=cum,
    )
