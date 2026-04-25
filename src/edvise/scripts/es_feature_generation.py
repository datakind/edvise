"""Edvise (ES) feature generation from standardized cohort/course silver tables."""

import argparse
import logging
import os
import sys
import typing as t

import pandas as pd

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from edvise import feature_generation, utils
from edvise.configs.es import ESProjectConfig
from edvise.dataio.read import read_config
from edvise.feature_generation.column_names import (
    CohortInputColumns,
    CourseFeatureSpec,
    CourseInputColumns,
    CumulativeFeatureSpec,
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
    SectionFeatureSpec,
    StudentFeatureSpec,
    StudentTermAddFeatureSpec,
    StudentTermAggregateSpec,
    TermFeatureSpec,
)
from edvise.feature_generation.es_feature_specs import build_edvise_feature_specs
from edvise.shared.logger import init_file_logging, local_fs_path, resolve_run_path
from edvise.shared.validation import require, require_cols, require_no_nulls, warn_if

LOGGER = logging.getLogger(__name__)


class ESFeatureGenerationTask:
    """Edvise project feature generation (standardized inputs -> student_terms.parquet)."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=ESProjectConfig)

    def run(self) -> None:
        preprocessing = self.cfg.preprocessing
        if preprocessing is None or preprocessing.features is None:
            raise ValueError("ES project config must define preprocessing.features")
        fc = preprocessing.features

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
                f"Missing df_course_standardized: {course_path} (local: {course_path_local})"
            )
        if not os.path.exists(cohort_path_local):
            raise FileNotFoundError(
                f"Missing df_cohort_standardized: {cohort_path} (local: {cohort_path_local})"
            )

        df_course = pd.read_parquet(course_path_local)
        df_cohort = pd.read_parquet(cohort_path_local)

        require_cols(
            df_course,
            [
                ES_COURSE_INPUT_COLUMNS.academic_year,
                ES_COURSE_INPUT_COLUMNS.academic_term,
                ES_COURSE_INPUT_COLUMNS.student_id,
            ],
            "Course standardized (Edvise)",
        )
        require_cols(
            df_cohort,
            [
                ES_COHORT_INPUT_COLUMNS.cohort_year_col,
                ES_COHORT_INPUT_COLUMNS.cohort_term_col,
                ES_COHORT_INPUT_COLUMNS.student_id,
            ],
            "Cohort standardized (Edvise)",
        )

        spec_bundle = build_edvise_feature_specs(
            df_cohort,
            df_course,
            cohort_cols=ES_COHORT_INPUT_COLUMNS,
            course_cols=ES_COURSE_INPUT_COLUMNS,
        )
        LOGGER.info(
            "Edvise feature specs: student=%s course=%s term=%s",
            spec_bundle.student,
            spec_bundle.course,
            spec_bundle.term,
        )

        merge_on = _student_level_merge_keys(
            df_cohort,
            df_course,
            cohort_cols=ES_COHORT_INPUT_COLUMNS,
        )
        require_no_nulls(df_cohort, merge_on, "Cohort standardized (Edvise)")
        require_no_nulls(df_course, merge_on, "Course standardized (Edvise)")
        _warn_join_overlap(df_cohort, df_course, merge_on)

        df_out = self.make_student_term_dataset(
            df_cohort=df_cohort,
            df_course=df_course,
            merge_on=merge_on,
            min_passing_grade=fc.min_passing_grade,
            min_num_credits_full_time=fc.min_num_credits_full_time,
            course_level_pattern=fc.course_level_pattern,
            core_terms=fc.core_terms,
            peak_covid_terms=fc.peak_covid_terms,
            key_course_subject_areas=fc.key_course_subject_areas,
            key_course_ids=fc.key_course_ids,
            cohort_input_columns=ES_COHORT_INPUT_COLUMNS,
            course_input_columns=ES_COURSE_INPUT_COLUMNS,
            student_feature_spec=spec_bundle.student,
            course_feature_spec=spec_bundle.course,
            term_feature_spec=spec_bundle.term,
            section_feature_spec=spec_bundle.section,
            student_term_aggregate_spec=spec_bundle.student_term_aggregate,
            student_term_add_feature_spec=spec_bundle.student_term_add,
            cumulative_feature_spec=spec_bundle.cumulative,
        )

        require(len(df_out) > 0, "student_terms.parquet is empty.")
        out_path = os.path.join(current_run_path, "student_terms.parquet")
        df_out.to_parquet(local_fs_path(out_path), index=False)
        LOGGER.info("Wrote %s", out_path)

    def make_student_term_dataset(
        self,
        df_cohort: pd.DataFrame,
        df_course: pd.DataFrame,
        *,
        merge_on: list[str],
        min_passing_grade: float = feature_generation.constants.DEFAULT_MIN_PASSING_GRADE,
        min_num_credits_full_time: float = feature_generation.constants.DEFAULT_MIN_NUM_CREDITS_FULL_TIME,
        course_level_pattern: str = feature_generation.constants.DEFAULT_COURSE_LEVEL_PATTERN,
        core_terms: set[str] = feature_generation.constants.DEFAULT_CORE_TERMS,
        peak_covid_terms: set[
            tuple[str, str]
        ] = feature_generation.constants.DEFAULT_PEAK_COVID_TERMS,
        key_course_subject_areas: t.Optional[list[str]] = None,
        key_course_ids: t.Optional[list[str]] = None,
        cohort_input_columns: CohortInputColumns = ES_COHORT_INPUT_COLUMNS,
        course_input_columns: CourseInputColumns = ES_COURSE_INPUT_COLUMNS,
        course_feature_spec: CourseFeatureSpec | None = None,
        student_feature_spec: StudentFeatureSpec | None = None,
        term_feature_spec: TermFeatureSpec | None = None,
        section_feature_spec: SectionFeatureSpec | None = None,
        student_term_aggregate_spec: StudentTermAggregateSpec | None = None,
        student_term_add_feature_spec: StudentTermAddFeatureSpec | None = None,
        cumulative_feature_spec: CumulativeFeatureSpec | None = None,
    ) -> pd.DataFrame:
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
                section_id_cols=[
                    "term_id",
                    "course_id",
                    course_input_columns.section_id,
                ],
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
                min_num_credits_full_time=min_num_credits_full_time,
                spec=student_term_add_feature_spec,
            )
        )

        cumulative_ids = [c for c in merge_on if c in df_student_terms.columns]
        return feature_generation.cumulative.add_features(
            df_student_terms,
            student_id_cols=cumulative_ids,
            sort_cols=[
                course_input_columns.academic_year,
                course_input_columns.academic_term,
            ],
            spec=cumulative_feature_spec,
        ).rename(columns=utils.data_cleaning.convert_to_snake_case)


def _student_level_merge_keys(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    cohort_cols: CohortInputColumns,
) -> list[str]:
    keys: list[str] = []
    if (
        cohort_cols.institution_id
        and cohort_cols.institution_id in df_cohort.columns
        and cohort_cols.institution_id in df_course.columns
    ):
        keys.append(cohort_cols.institution_id)
    keys.append(cohort_cols.student_id)
    return keys


def _warn_join_overlap(df_cohort: pd.DataFrame, df_course: pd.DataFrame, keys: list[str]) -> None:
    cohort_keys = set(zip(*[df_cohort[k] for k in keys])) if keys else set()
    course_keys = set(zip(*[df_course[k] for k in keys])) if keys else set()
    require(
        cohort_keys and course_keys,
        f"No {keys} keys found in cohort or course for overlap check",
    )
    overlap = len(cohort_keys & course_keys) / min(len(cohort_keys), len(course_keys))
    warn_if(
        overlap < 0.10,
        f"Cohort/course key overlap is low ({overlap:.3f}).",
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edvise feature generation (standardized -> student_terms)"
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = ESFeatureGenerationTask(args)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="es_feature_generation.log",
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
