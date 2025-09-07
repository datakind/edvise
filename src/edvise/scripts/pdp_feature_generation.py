import argparse
import importlib
import logging
import sys
import pandas as pd
import typing as t

from src.edvise import feature_generation, utils
from src.edvise.dataio.read import read_config
from src.edvise.configs.pdp import PDPProjectConfig

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

        # --- Load datasets ---
        df_course = pd.read_parquet(
            f"{self.args.silver_volume_path}/df_cohort_validated.parquet"
        )
        df_cohort = pd.read_parquet(
            f"{self.args.silver_volume_path}/df_course_validated.parquet"
        )

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
        df_student_terms.to_parquet(
            f"{self.args.silver_volume_path}/student_terms.parquet", index=False
        )

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
        first_term = utils.infer_first_term_of_year(
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
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    #no school use a custom schema for now remove and add back in iff needed
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
    task.run()
