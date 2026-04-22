"""
Shared feature-generation step. Schema-specific behavior via FeatureGenerationBackend
(config type and log filename).
"""

from __future__ import annotations

import argparse
import logging
import os
import typing as t
from typing import Literal

import pandas as pd

from edvise import feature_generation, utils
from edvise.dataio.read import read_config
from edvise.feature_generation.cohort_columns import StudentCohortColumns
from edvise.feature_generation.course_columns import CourseStandardizedColumns
from edvise.feature_generation.pipeline_columns import (
    FeaturePipelineColumns,
    es_feature_pipeline_columns,
    pdp_feature_pipeline_columns,
)
from edvise.shared.logger import local_fs_path, resolve_run_path
from edvise.shared.validation import require, require_cols, require_no_nulls, warn_if

LOGGER = logging.getLogger(__name__)


class FeatureGenerationBackend(t.NamedTuple):
    """Inject config schema, log file name, and schema-specific column maps."""

    config_schema: type
    log_file_name: str
    student_cohort_columns: StudentCohortColumns
    course_standardized_columns: CourseStandardizedColumns
    pipeline_flavor: Literal["pdp", "es"]


def _feature_pipeline_columns_for_config(
    *,
    student_id_col: str,
    pipeline_flavor: Literal["pdp", "es"],
) -> FeaturePipelineColumns:
    """Build term/course/student-term/cumulative key layout from project config."""
    if pipeline_flavor == "es":
        return es_feature_pipeline_columns(student_id_col)
    return pdp_feature_pipeline_columns(student_id_col)


class FeatureGenerationTask:
    """Build ``student_terms.parquet`` from standardized cohort/course inputs."""

    def __init__(self, args: argparse.Namespace, backend: FeatureGenerationBackend):
        self.args = args
        self._backend = backend
        self.cfg = read_config(
            self.args.config_file_path, schema=backend.config_schema
        )

    def _feature_pipeline(self) -> FeaturePipelineColumns:
        return _feature_pipeline_columns_for_config(
            student_id_col=self.cfg.student_id_col,
            pipeline_flavor=self._backend.pipeline_flavor,
        )

    def run(self) -> None:
        """Load standardized parquet outputs, run feature pipes, write student terms."""
        if self.cfg.preprocessing is None:
            raise ValueError("Config must define preprocessing for feature generation.")
        features_cfg = self.cfg.preprocessing.features
        min_passing_grade = features_cfg.min_passing_grade
        min_num_credits_full_time = features_cfg.min_num_credits_full_time
        course_level_pattern = features_cfg.course_level_pattern
        core_terms = features_cfg.core_terms
        key_course_subject_areas = features_cfg.key_course_subject_areas
        key_course_ids = features_cfg.key_course_ids

        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        course_path = os.path.join(current_run_path, "df_course_standardized.parquet")
        cohort_path = os.path.join(current_run_path, "df_cohort_standardized.parquet")
        course_path_local = local_fs_path(course_path)
        cohort_path_local = local_fs_path(cohort_path)

        if not os.path.exists(course_path_local):
            raise FileNotFoundError(
                f"Missing df_course_standardized at: {course_path} (local: {course_path_local})"
            )
        if not os.path.exists(cohort_path_local):
            raise FileNotFoundError(
                f"Missing df_cohort_standardized at: {cohort_path} (local: {cohort_path_local})"
            )

        df_course = pd.read_parquet(course_path_local)
        df_cohort = pd.read_parquet(cohort_path_local)

        sid = self.cfg.student_id_col
        pc = self._feature_pipeline()
        require_cols(
            df_course,
            [
                "institution_id",
                sid,
                pc.term.academic_year_col,
                pc.term.academic_term_col,
            ],
            "Course standardized",
        )
        require_cols(df_cohort, ["institution_id", sid], "Cohort standardized")

        require_no_nulls(
            df_course,
            ["institution_id", sid],
            "Course standardized",
        )
        require_no_nulls(
            df_cohort, ["institution_id", sid], "Cohort standardized"
        )

        cohort_pairs = set(zip(df_cohort["institution_id"], df_cohort[sid]))
        course_pairs = set(zip(df_course["institution_id"], df_course[sid]))
        require(
            cohort_pairs and course_pairs,
            f"No valid (institution_id, {sid}) pairs found in cohort/course.",
        )

        overlap = len(cohort_pairs & course_pairs) / min(
            len(cohort_pairs), len(course_pairs)
        )
        warn_if(
            overlap >= 0.10,
            f"Cohort/course appear mismatched: overlap too low ({overlap:.3f}).",
        )

        df_student_terms = self.make_student_term_dataset(
            df_cohort=df_cohort,
            df_course=df_course,
            min_passing_grade=min_passing_grade,
            min_num_credits_full_time=min_num_credits_full_time,
            course_level_pattern=course_level_pattern,
            core_terms=core_terms,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
        )

        require(len(df_student_terms) > 0, "student_terms.parquet is empty.")

        out_path = os.path.join(current_run_path, "student_terms.parquet")
        df_student_terms.to_parquet(local_fs_path(out_path), index=False)

    def make_student_term_dataset(
        self,
        df_cohort: pd.DataFrame,
        df_course: pd.DataFrame,
        *,
        min_passing_grade: float = feature_generation.constants.DEFAULT_MIN_PASSING_GRADE,
        min_num_credits_full_time: float = feature_generation.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
        course_level_pattern: str = feature_generation.constants.DEFAULT_COURSE_LEVEL_PATTERN,
        core_terms: set[str] = feature_generation.constants.DEFAULT_CORE_TERMS,
        peak_covid_terms: set[
            tuple[str, str]
        ] = feature_generation.constants.DEFAULT_PEAK_COVID_TERMS,
        key_course_subject_areas: t.Optional[list[str]] = None,
        key_course_ids: t.Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Run term → course → section → student-term → cumulative using config keys."""
        pc = self._feature_pipeline()
        first_term = utils.infer_data_terms.infer_first_term_of_year(
            df_course[pc.term.academic_term_col]
        )

        df_students = df_cohort.pipe(
            feature_generation.student.add_features,
            first_term_of_year=first_term,
            columns=self._backend.student_cohort_columns,
        )

        df_courses_plus = (
            df_course.pipe(
                feature_generation.course.add_features,
                min_passing_grade=min_passing_grade,
                course_level_pattern=course_level_pattern,
                columns=self._backend.course_standardized_columns,
            )
            .pipe(
                feature_generation.term.add_features,
                first_term_of_year=first_term,
                core_terms=core_terms,
                peak_covid_terms=peak_covid_terms,
                columns=pc.term,
            )
            .pipe(
                feature_generation.section.add_features,
                columns=pc.section,
            )
        )

        df_student_terms = (
            feature_generation.student_term.aggregate_from_course_level_features(
                df_courses_plus,
                columns=pc.student_term_agg,
                min_passing_grade=min_passing_grade,
                key_course_subject_areas=key_course_subject_areas,
                key_course_ids=key_course_ids,
            )
            .merge(
                df_students,
                how="inner",
                on=list(pc.student_term_agg.merge_student_on),
            )
            .pipe(
                feature_generation.student_term.add_features,
                min_num_credits_full_time=min_num_credits_full_time,
                columns=pc.student_term_features,
                group_by_for_prev_term=pc.student_term_agg.merge_student_on,
            )
        )

        df_student_terms_plus = feature_generation.cumulative.add_features(
            df_student_terms,
            columns=pc.cumulative,
        ).rename(columns=utils.data_cleaning.convert_to_snake_case)

        return df_student_terms_plus
