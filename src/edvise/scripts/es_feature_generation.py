"""Edvise (ES) feature generation from standardized cohort/course silver tables."""

import argparse
import logging
import os
import sys

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

from edvise.configs.es import ESProjectConfig
from edvise.dataio.read import read_config
from edvise.feature_generation.column_names import (
    ES_COHORT_INPUT_COLUMNS,
    ES_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.assemble_student_terms import (
    make_student_term_dataset,
    read_standardized_silver_pair,
    student_level_merge_keys,
    warn_cohort_course_key_overlap,
)
from edvise.feature_generation.es_feature_specs import build_edvise_feature_specs
from edvise.shared.logger import init_file_logging, local_fs_path, resolve_run_path
from edvise.shared.validation import require, require_cols, require_no_nulls

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

        df_course, df_cohort = read_standardized_silver_pair(current_run_path)

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

        merge_on = student_level_merge_keys(
            df_cohort,
            df_course,
            cohort_cols=ES_COHORT_INPUT_COLUMNS,
        )
        require_no_nulls(df_cohort, merge_on, "Cohort standardized (Edvise)")
        require_no_nulls(df_course, merge_on, "Course standardized (Edvise)")
        warn_cohort_course_key_overlap(df_cohort, df_course, merge_on)

        df_out = make_student_term_dataset(
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
