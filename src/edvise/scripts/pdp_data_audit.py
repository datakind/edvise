import argparse
import importlib
import logging
import typing as t
import sys
import pandas as pd
import os

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

from edvise.data_audit.schemas import RawPDPCohortDataSchema, RawPDPCourseDataSchema
from edvise.data_audit.standardizer import (
    PDPCohortStandardizer,
    PDPCourseStandardizer,
)
from edvise.utils.databricks import get_spark_session
from edvise.utils.data_cleaning import handling_duplicates

from edvise.dataio.read import (
    read_config,
    read_raw_pdp_cohort_data,
    read_raw_pdp_course_data,
)
from edvise.dataio.write import write_parquet
from edvise.configs.pdp import PDPProjectConfig
from edvise.data_audit.eda import compute_gateway_course_ids_and_cips, log_record_drops, log_most_recent_terms
from edvise.utils.update_config import update_key_courses_and_cips

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Create callable type
ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class PDPDataAuditTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(
        self,
        args: argparse.Namespace,
        course_converter_func: t.Optional[ConverterFunc] = None,
        cohort_converter_func: t.Optional[ConverterFunc] = None,
    ):
        self.args = args
        self.cfg = read_config(
            file_path=self.args.config_file_path, schema=PDPProjectConfig
        )
        self.spark = get_spark_session()
        self.cohort_std = PDPCohortStandardizer()
        self.course_std = PDPCourseStandardizer()
        # self.course_converter_func: t.Optional[ConverterFunc] = course_converter_func
        # Use default converter to handle duplicates if none provided
        self.course_converter_func: ConverterFunc = (
            handling_duplicates
            if course_converter_func is None
            else course_converter_func
        )
        self.cohort_converter_func: t.Optional[ConverterFunc] = cohort_converter_func

    def run(self):
        """Executes the data preprocessing pipeline."""
        cohort_dataset_raw_path = self.cfg.datasets.bronze.raw_cohort.file_path
        course_dataset_raw_path = self.cfg.datasets.bronze.raw_course.file_path

        # --- Load datasets ---

        # Cohort
        # Raw cohort data
        df_cohort_raw = read_raw_pdp_cohort_data(
            file_path=cohort_dataset_raw_path,
            schema=None,
            spark_session=self.spark,
        )

        # Schema validate cohort data
        LOGGER.info("Reading and schema validating cohort data:")
        df_cohort_validated = read_raw_pdp_cohort_data(
            file_path=cohort_dataset_raw_path,
            schema=RawPDPCohortDataSchema,
            converter_func=self.cohort_converter_func,
            spark_session=self.spark,
        )

        # Standardize cohort data
        LOGGER.info("Standardizing cohort data:")
        df_cohort_standardized = self.cohort_std.standardize(df_cohort_validated)

        LOGGER.info("Cohort data standardized.")

        # Course
        dttm_formats = ["ISO8601", "%Y%m%d.0"]

        # Schema validate course data and handle duplicates
        LOGGER.info(
            "Reading and schema validating course data, handling any duplicates:"
        )

        for fmt in dttm_formats:
            try:
                # Raw course data
                df_course_raw = read_raw_pdp_cohort_data(
                    file_path=cohort_dataset_raw_path,
                    schema=None,
                    dttm_format=fmt,
                    spark_session=self.spark,
                )
                df_course_validated = read_raw_pdp_course_data(
                    file_path=course_dataset_raw_path,
                    schema=RawPDPCourseDataSchema,
                    dttm_format=fmt,
                    converter_func=self.course_converter_func,
                    # converter_func=handling_duplicates,
                    spark_session=self.spark,
                )
                break  # success — exit loop
            except ValueError:
                continue  # try next format
        else:
            raise ValueError(
                "Failed to parse course data with all known datetime formats."
            )
        LOGGER.info("Course data read and schema validated, duplicates handled.")

        # Standardize course data
        LOGGER.info("Standardizing course data:")
        df_course_standardized = self.course_std.standardize(df_course_validated)

        LOGGER.info("Course data standardized.")
        
        # Log Math/English gateway courses and add to config
        ids_cips = compute_gateway_course_ids_and_cips(df_course_standardized)
        LOGGER.info("Auto-populating config with below course IDs and cip codes: change if necessary")
        update_key_courses_and_cips(
            self.args.config_file_path,  
            key_course_ids=ids_cips[0],
            key_course_subject_areas=ids_cips[1]
        )

        # Log changes before and after pre-processing
        log_record_drops(
            df_cohort_raw, 
            df_cohort_standardized, 
            df_course_raw,
            df_course_standardized,
        )

        # Logs most recent terms 
        log_most_recent_terms(
            df_course_standardized,
            df_cohort_standardized,
        )

        # --- Write results ---
        write_parquet(
            df_cohort_standardized,
            f"{self.args.silver_volume_path}/df_cohort_standardized.parquet",
        )
        write_parquet(
            df_course_standardized,
            f"{self.args.silver_volume_path}/df_course_standardized.parquet",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--bronze_volume_path", type=str, required=False)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--DB_workspace", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.bronze_volume_path:
        sys.path.append(f"{args.bronze_volume_path}/training_inputs")
    try:
        converter_func = importlib.import_module("dataio")
        cohort_converter_func = converter_func.converter_func_cohort
        LOGGER.info("Running task with custom cohort converter func")
    except Exception as e:
        cohort_converter_func = None
        LOGGER.info("Running task with default cohort converter func")
        LOGGER.warning(f"Failed to load custom converter functions: {e}")
    try:
        converter_func = importlib.import_module("dataio")
        course_converter_func = converter_func.converter_func_course
        LOGGER.info("Running task with custom course converter func")
    except Exception as e:
        course_converter_func = None
        LOGGER.info("Running task default course converter func")
        LOGGER.warning(f"Failed to load custom converter functions: {e}")
    # try:
    #     schemas = importlib.import_module("schemas")
    #     LOGGER.info("Running task with custom schema")
    # except Exception as e:
    #     from data_audit import schemas as schemas
    #     LOGGER.info("Running task with default schema")
    #     LOGGER.warning(f"Failed to load custom schema: {e}")

    task = PDPDataAuditTask(
        args,
        cohort_converter_func=cohort_converter_func,
        course_converter_func=course_converter_func,
    )
    task.run()
