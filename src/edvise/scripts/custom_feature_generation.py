### THIS IS A DRAFT POC SCRIPT

import argparse
import importlib
import logging
import sys
import pandas as pd

from edvise.feature_generation.assemble_student_terms import (
    make_student_term_dataset,
    student_level_merge_keys,
)
from edvise.feature_generation.column_names import (
    PDP_COHORT_INPUT_COLUMNS,
    PDP_COURSE_INPUT_COLUMNS,
)
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


class CustomFeatureGenerationTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.toml_file_path, schema=PDPProjectConfig)

    def run(self):
        """Executes the data preprocessing pipeline."""

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
            f"{self.args.course_dataset_validated_path}/df_cohort.parquet"
        )
        df_cohort = pd.read_parquet(
            f"{self.args.cohort_dataset_validated_path}/df_course.parquet"
        )

        # --- Generate student-term dataset ---
        merge_on = student_level_merge_keys(
            df_cohort,
            df_course,
            cohort_cols=PDP_COHORT_INPUT_COLUMNS,
        )
        df_student_terms = make_student_term_dataset(
            df_cohort=df_cohort,
            df_course=df_course,
            merge_on=merge_on,
            min_passing_grade=min_passing_grade,
            min_num_credits_full_time=min_num_credits_full_time,
            course_level_pattern=course_level_pattern,
            core_terms=core_terms,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
            cohort_input_columns=PDP_COHORT_INPUT_COLUMNS,
            course_input_columns=PDP_COURSE_INPUT_COLUMNS,
        )

        # --- Write result ---
        df_student_terms.to_parquet(
            f"{self.args.student_term_path}/student_terms.parquet", index=False
        )

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument("--cohort_dataset_validated_path", type=str, required=True)
    parser.add_argument("--course_dataset_validated_path", type=str, required=True)
    parser.add_argument("--toml_file_path", type=str, required=True)
    parser.add_argument("--custom_schemas_path", required=False)
    parser.add_argument("--student_term_path", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        sys.path.append(args.custom_schemas_path)
        sys.path.append(
            f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
        )
        schemas = importlib.import_module("schemas")
        logging.info("Running task with custom schema")
    except Exception:
        logging.info("Running task with default schema")

    task = CustomFeatureGenerationTask(args)
    task.run()
