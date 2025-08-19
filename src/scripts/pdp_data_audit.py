import argparse
import importlib
import logging
import sys
import pandas as pd
import typing as t

from .. import utils
from .. import feature_generation
from src.data_audit.standardizer import (
    PDPCohortStandardizer,
    PDPCourseStandardizer,
    StudentTermStandardizer,
)
from src.utils.databricks import read_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


class PDPDataAuditTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.toml_file_path)
        self.cohort_std = PDPCohortStandardizer()
        self.course_std = PDPCourseStandardizer()
        self.student_term_std = StudentTermStandardizer()

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
        df_course = pd.read_parquet(f"{self.args.course_dataset_validated_path}/df_cohort.parquet")
        df_cohort = pd.read_parquet(f"{self.args.cohort_dataset_validated_path}/df_course.parquet")

        ###DATA AUDIT STEPS##
        # Call the standardizers # 

        # --- Write results ---
        df_course = pd.to_parquet(f"{self.args.course_dataset_validated_path}/df_cohort.parquet")
        df_cohort = pd.to_parquet(f"{self.args.cohort_dataset_validated_path}/df_course.parquet")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data preprocessing for inference in the SST pipeline.")
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
        from dataio.schemas import pdp as schemas
        logging.info("Running task with default schema")

    task = PDPFeatureGenerationTask(args)
    task.run()
