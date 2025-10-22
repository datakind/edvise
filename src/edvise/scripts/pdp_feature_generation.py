import argparse
import logging
import pandas as pd
import typing as t
import os
import sys

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Debug info
print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from edvise import feature_generation, utils
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig
from edvise.shared.logger import resolve_run_path, local_fs_path, init_file_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


class PDPFeatureGenerationTask:
    """Encapsulates the  feature generationlogic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def run(self):
        """Executes the data feature generation."""

        # --- Unpack config ---
        features_cfg = self.cfg.preprocessing.features
        min_passing_grade = features_cfg.min_passing_grade
        min_num_credits_full_time = features_cfg.min_num_credits_full_time
        course_level_pattern = features_cfg.course_level_pattern
        core_terms = features_cfg.core_terms
        key_course_subject_areas = features_cfg.key_course_subject_areas
        key_course_ids = features_cfg.key_course_ids

        # Ensure correct folder: training or inference
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        # Use local path for reading/writing so DBFS is handled correctly
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        # --- Load datasets ---
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

        # --- Generate student-term dataset ---
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

        # --- Write result ---
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
        """Main feature generation pipeline."""
        first_term = utils.infer_data_terms.infer_first_term_of_year(
            df_course["academic_term"]
        )

        df_students = df_cohort.pipe(
            feature_generation.student.add_features, first_term_of_year=first_term
        )

        df_courses_plus = (
            df_course.pipe(
                feature_generation.course.add_features,
                min_passing_grade=min_passing_grade,
                course_level_pattern=course_level_pattern,
            )
            .pipe(
                feature_generation.term.add_features,
                first_term_of_year=first_term,
                core_terms=core_terms,
                peak_covid_terms=peak_covid_terms,
            )
            .pipe(
                feature_generation.section.add_features,
                section_id_cols=["term_id", "course_id", "section_id"],
            )
        )

        df_student_terms = (
            feature_generation.student_term.aggregate_from_course_level_features(
                df_courses_plus,
                student_term_id_cols=["student_id", "term_id"],
                min_passing_grade=min_passing_grade,
                key_course_subject_areas=key_course_subject_areas,
                key_course_ids=key_course_ids,
            )
            .merge(df_students, how="inner", on=["institution_id", "student_id"])
            .pipe(
                feature_generation.student_term.add_features,
                min_num_credits_full_time=min_num_credits_full_time,
            )
        )

        df_student_terms_plus = feature_generation.cumulative.add_features(
            df_student_terms,
            student_id_cols=["institution_id", "student_id"],
            sort_cols=["academic_year", "academic_term"],
        ).rename(columns=utils.data_cleaning.convert_to_snake_case)

        return df_student_terms_plus


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature generation in the Edvise pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # no school use a custom schema for now remove and add back in iff needed
    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     sys.path.append(
    #         f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
    #     )
    #     schemas = importlib.import_module("schemas")
    #     logging.info("Running task with custom schema")
    # except Exception:
    #     logging.info("Running task with default schema")

    task = PDPFeatureGenerationTask(args)
    init_file_logging(args, task.cfg, logger_name=__name__)
    task.run()

    # Ensure all logs hit disk
    for h in logging.getLogger().handlers:
        try: h.flush()
        except Exception: pass
    logging.shutdown()
