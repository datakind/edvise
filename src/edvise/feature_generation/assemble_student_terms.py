"""Shared helpers to assemble student-term features from cohort/course tables."""

from __future__ import annotations

import os
import typing as t

import pandas as pd

from edvise import feature_generation, utils
from edvise.feature_generation.course import GradeSemantics
from edvise.dataio.read import read_resolved_parquet
from edvise.feature_generation.column_names import (
    CohortInputColumns,
    CourseFeatureSpec,
    CourseInputColumns,
    CumulativeFeatureSpec,
    SectionFeatureSpec,
    StudentFeatureSpec,
    StudentTermAddFeatureSpec,
    StudentTermAggregateSpec,
    TermFeatureSpec,
)
from edvise.shared.validation import require, warn_if


def read_standardized_silver_pair(run_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read standardized course/cohort parquet inputs from a run folder."""
    course_path = os.path.join(run_path, "df_course_standardized.parquet")
    cohort_path = os.path.join(run_path, "df_cohort_standardized.parquet")
    return read_resolved_parquet(course_path), read_resolved_parquet(cohort_path)


def student_level_merge_keys(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    cohort_cols: CohortInputColumns,
) -> list[str]:
    """Resolve student-level merge keys present in both frames."""
    keys: list[str] = []
    if (
        cohort_cols.institution_id
        and cohort_cols.institution_id in df_cohort.columns
        and cohort_cols.institution_id in df_course.columns
    ):
        keys.append(cohort_cols.institution_id)
    keys.append(cohort_cols.student_id)
    return keys


def warn_cohort_course_key_overlap(
    df_cohort: pd.DataFrame, df_course: pd.DataFrame, keys: list[str]
) -> None:
    """Warn when cohort/course join key overlap is unexpectedly low."""
    cohort_keys = set(zip(*[df_cohort[k] for k in keys])) if keys else set()
    course_keys = set(zip(*[df_course[k] for k in keys])) if keys else set()
    require(
        bool(cohort_keys) and bool(course_keys),
        f"No {keys} keys found in cohort or course for overlap check",
    )
    overlap = len(cohort_keys & course_keys) / min(len(cohort_keys), len(course_keys))
    warn_if(
        overlap < 0.10,
        f"Cohort/course key overlap is low ({overlap:.3f}).",
    )


def make_student_term_dataset(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    merge_on: list[str],
    cohort_input_columns: CohortInputColumns,
    course_input_columns: CourseInputColumns,
    min_passing_grade: float = feature_generation.constants.DEFAULT_MIN_PASSING_GRADE,
    min_num_credits_full_time: float = feature_generation.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
    course_level_pattern: str = feature_generation.constants.DEFAULT_COURSE_LEVEL_PATTERN,
    core_terms: set[str] = feature_generation.constants.DEFAULT_CORE_TERMS,
    peak_covid_terms: set[
        tuple[str, str]
    ] = feature_generation.constants.DEFAULT_PEAK_COVID_TERMS,
    key_course_subject_areas: t.Optional[list[t.Union[str, list[str]]]] = None,
    key_course_ids: t.Optional[list[t.Union[str, list[str]]]] = None,
    course_feature_spec: CourseFeatureSpec | None = None,
    student_feature_spec: StudentFeatureSpec | None = None,
    term_feature_spec: TermFeatureSpec | None = None,
    section_feature_spec: SectionFeatureSpec | None = None,
    student_term_aggregate_spec: StudentTermAggregateSpec | None = None,
    student_term_add_feature_spec: StudentTermAddFeatureSpec | None = None,
    cumulative_feature_spec: CumulativeFeatureSpec | None = None,
    grade_semantics: GradeSemantics = "pdp",
) -> pd.DataFrame:
    """Generate student-term features from standardized cohort/course dataframes."""
    first_term = utils.infer_data_terms.infer_first_term_of_year(
        df_course[course_input_columns.academic_term]
    )

    df_students = df_cohort.pipe(
        feature_generation.student.add_features,
        first_term_of_year=first_term,
        cols=cohort_input_columns,
        spec=student_feature_spec,
    )

    df_courses_plus = (
        df_course.pipe(
            feature_generation.course.add_features,
            cols=course_input_columns,
            spec=course_feature_spec,
            min_passing_grade=min_passing_grade,
            course_level_pattern=course_level_pattern,
            grade_semantics=grade_semantics,
        )
        .pipe(
            feature_generation.term.add_features,
            first_term_of_year=first_term,
            core_terms=core_terms,
            peak_covid_terms=peak_covid_terms,
            year_col=course_input_columns.academic_year,
            term_col=course_input_columns.academic_term,
            spec=term_feature_spec,
        )
        .pipe(
            feature_generation.section.add_features,
            section_id_cols=["term_id", "course_id", course_input_columns.section_id],
            student_id_col=course_input_columns.student_id,
            spec=section_feature_spec,
        )
    )

    student_term_id_cols = [course_input_columns.student_id, "term_id"]
    df_student_terms = (
        feature_generation.student_term.aggregate_from_course_level_features(
            df_courses_plus,
            student_term_id_cols=student_term_id_cols,
            cols=course_input_columns,
            min_passing_grade=min_passing_grade,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
            spec=student_term_aggregate_spec,
        )
        .merge(df_students, how="inner", on=merge_on)
        .pipe(
            feature_generation.student_term.add_features,
            cols=course_input_columns,
            min_num_credits_full_time=min_num_credits_full_time,
            spec=student_term_add_feature_spec,
        )
    )

    cumulative_ids = [k for k in merge_on if k in df_student_terms.columns]
    return feature_generation.cumulative.add_features(
        df_student_terms,
        student_id_cols=cumulative_ids,
        sort_cols=[
            course_input_columns.academic_year,
            course_input_columns.academic_term,
        ],
        spec=cumulative_feature_spec,
    ).rename(columns=utils.data_cleaning.convert_to_snake_case)
