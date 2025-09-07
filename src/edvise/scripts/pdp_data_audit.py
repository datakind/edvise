import argparse
import importlib
import logging
import typing as t
import sys
import pandas as pd
import os

# Add 'src' to sys.path if not already there
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sys.argv[0]))))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.edvise import data_audit
from src.edvise.data_audit.standardizer import (
    PDPCohortStandardizer,
    PDPCourseStandardizer,
)
from src.edvise.utils.databricks import get_spark_session
from src.edvise.dataio.read import (
    read_config,
    read_raw_pdp_cohort_data,
    read_raw_pdp_course_data,
)
from src.edvise.dataio.write import write_parquet
from src.edvise.configs.pdp import PDPProjectConfig

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
        self.course_converter_func: t.Optional[ConverterFunc] = course_converter_func
        self.cohort_converter_func: t.Optional[ConverterFunc] = cohort_converter_func

    def run(self):
        """Executes the data preprocessing pipeline."""
        cohort_dataset_raw_path = self.cfg.datasets.bronze.raw_cohort.file_path
        course_dataset_raw_path = self.cfg.datasets.bronze.raw_course.file_path

        # --- Load datasets ---
        # Cohort
        df_cohort_raw = read_raw_pdp_cohort_data(
            file_path=cohort_dataset_raw_path,
            schema=data_audit.schemas.RawPDPCohortDataSchema,
            converter_func=self.cohort_converter_func,
            spark_session=self.spark,
        )
        LOGGER.info("Cohort data read and schema validated.")

        # Standardize cohort data
        df_cohort_validated = self.cohort_std.standardize(df_cohort_raw)

        # Course
        dttm_formats = ["ISO8601", "%Y%m%d.0"]

        for fmt in dttm_formats:
            try:
                df_course_raw = read_raw_pdp_course_data(
                    file_path=course_dataset_raw_path,
                    schema=data_audit.schemas.RawPDPCourseDataSchema,
                    dttm_format=fmt,
                    converter_func=self.course_converter_func,
                    spark_session=self.spark,
                )
                break  # success — exit loop
            except ValueError:
                continue  # try next format
        else:
            raise ValueError(
                "Failed to parse course data with all known datetime formats."
            )

        # Standardize course data
        df_course_validated = self.course_std.standardize(df_course_raw)

        LOGGER.info("Course data read and schema validated.")

        # --- Write results ---
        write_parquet(
            df_cohort_validated,
            f"{self.args.silver_volume_path}/df_cohort_validated.parquet",
        )
        write_parquet(
            df_course_validated,
            f"{self.args.silver_volume_path}/df_course_validated.parquet",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--bronze_volume_path", type=str, required=False, default=None)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--DB_workspace", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.bronze_volume_path:
        sys.path.append(f'{args.bronze_volume_path}/training_inputs')
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
