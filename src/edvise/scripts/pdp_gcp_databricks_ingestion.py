"""
This script ingests course and cohort data for the Student Success Tool (SST) pipeline.

It reads data from CSV files stored in a Google Cloud Storage (GCS) bucket,
performs schema validation using the `pdp` library, and writes the validated data
to Delta Lake tables in Databricks Unity Catalog.

The script is designed to run within a Databricks environment as a job, leveraging
Databricks utilities for job task values, and Spark session management.

This is a POC script, it is advised to review and tests before using in production.
"""

import typing as t
import logging
import os
import argparse
import sys

from databricks.sdk.runtime import dbutils
from google.cloud import storage

import utils
import dataio
import importlib
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger

# Create callable type
ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class DataIngestionTask:
    """
    Encapsulates the data ingestion logic for the SST pipeline.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        course_converter_func: t.Optional[ConverterFunc] = None,
        cohort_converter_func: t.Optional[ConverterFunc] = None,
    ):
        """
        Initializes the DataIngestionTask with parsed arguments.
        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.spark_session = utils.databricks.get_spark_session()
        self.args = args
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.args.gcp_bucket_name)
        self.course_converter_func: t.Optional[ConverterFunc] = course_converter_func
        self.cohort_converter_func: t.Optional[ConverterFunc] = cohort_converter_func

    def download_data_from_gcs(self, internal_pipeline_path: str) -> tuple[str, str]:
        """
        Downloads course and cohort data from GCS to the internal pipeline directory.

        Args:
            internal_pipeline_path (str): The path to the internal pipeline directory.

        Returns:
            tuple[str, str]: The file paths of the downloaded course and cohort data.
        """
        sst_container_folder = "validated"
        try:
            # Download course data from GCS
            course_blob_name = f"{sst_container_folder}/{self.args.course_file_name}"
            course_blob = self.bucket.blob(course_blob_name)
            course_file_path = f"{internal_pipeline_path}{self.args.course_file_name}"
            course_blob.download_to_filename(course_file_path)
            logging.info("Course data downloaded from GCS: %s", course_file_path)

            # Download cohort data from GCS
            cohort_blob_name = f"{sst_container_folder}/{self.args.cohort_file_name}"
            cohort_blob = self.bucket.blob(cohort_blob_name)
            cohort_file_path = f"{internal_pipeline_path}{self.args.cohort_file_name}"
            cohort_blob.download_to_filename(cohort_file_path)
            logging.info("Cohort data downloaded from GCS: %s", cohort_file_path)

            return course_file_path, cohort_file_path
        except Exception as e:
            logging.error(f"GCS download error: {e}")
            raise

    def read_and_validate_data(
        self, fpath_course: str, fpath_cohort: str
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # -> tuple[raw_course.RawPDPCourseDataSchema, raw_cohort.RawPDPCohortDataSchema]:
        """
        Reads course and cohort data from CSV files and validates their schemas.

        Args:
            fpath_course (str): Path to the course data file.
            fpath_cohort (str): Path to the cohort data file.

        Returns:
            tuple[raw_course.RawPDPCourseDataSchema, raw_cohort.RawPDPCohortDataSchema]:
                Validated course and cohort data.
        """

        # Read data from CSV files into Pandas DataFrames and validate schema
        dttm_formats = ["ISO8601", "%Y%m%d.0"]

        for fmt in dttm_formats:
            try:
                df_course = dataio.read.read_raw_pdp_course_data(
                    file_path=fpath_course,
                    schema=raw_course.RawPDPCourseDataSchema,
                    dttm_format=fmt,
                    converter_func=self.course_converter_func,
                )
                break  # success — exit loop
            except ValueError:
                continue  # try next format
        else:
            raise ValueError(
                "Failed to parse course data with all known datetime formats."
            )

        logging.info("Course data read and schema validated.")
        df_cohort = dataio.read.read_raw_pdp_cohort_data(
            file_path=fpath_cohort,
            schema=raw_cohort.RawPDPCohortDataSchema,
            converter_func=self.cohort_converter_func,
        )
        logging.info("Cohort data read and schema validated.")
        return df_course, df_cohort

    def write_data_to_delta_lake(
        self,
        df_course,
        df_cohort,
    ):
        """
        Writes the validated DataFrames to Delta Lake tables in Unity Catalog.

        Args:
            df_course (raw_course.RawPDPCourseDataSchema): Validated course data.
            df_cohort (raw_cohort.RawPDPCohortDataSchema): Validated cohort data.
        """

        catalog = self.args.DB_workspace
        write_schema = f"{self.args.databricks_institution_name}_bronze"
        course_dataset_validated_path = (
            f"{catalog}.{write_schema}.{self.args.db_run_id}_course_dataset_validated"
        )
        cohort_dataset_validated_path = (
            f"{catalog}.{write_schema}.{self.args.db_run_id}_cohort_dataset_validated"
        )
        try:
            dataio.write.to_delta_table(
                df_course,
                course_dataset_validated_path,
                spark_session=self.spark_session,
            )
            logging.info(
                "Course data written to Delta Lake table: %s.%s.%s_course_dataset_validated",
                catalog,
                write_schema,
                self.args.db_run_id,
            )

            dataio.write.to_delta_table(
                df_cohort,
                cohort_dataset_validated_path,
                spark_session=self.spark_session,
            )
            logging.info(
                "Cohort data written to Delta Lake table: %s.%s.%s_cohort_dataset_validated",
                catalog,
                write_schema,
                self.args.db_run_id,
            )

            return course_dataset_validated_path, cohort_dataset_validated_path
        except Exception as e:
            logging.error("Error writing to Delta Lake: %s", e)
            raise

    def verify_delta_lake_write(
        self, course_dataset_validated_path, cohort_dataset_validated_path
    ):
        """
        Verifies the Delta Lake write by reading the data back from the tables.
        """
        try:
            df_course_from_catalog = raw_course.RawPDPCourseDataSchema(
                dataio.from_delta_table(
                    course_dataset_validated_path,
                    spark_session=self.spark_session,
                )
            )
            logging.info(
                "Course DataFrame shape from catalog: %s", df_course_from_catalog.shape
            )

            df_cohort_from_catalog = raw_cohort.RawPDPCohortDataSchema(
                dataio.from_delta_table(
                    cohort_dataset_validated_path,
                    spark_session=self.spark_session,
                )
            )
            logging.info(
                "Cohort DataFrame shape from catalog: %s", df_cohort_from_catalog.shape
            )
        except Exception as e:
            logging.error("Error writing to Delta Lake: %s", e)
            raise

    def run(self):
        """
        Executes the data ingestion task.
        """
        raw_files_path = f"{self.args.job_root_dir}/raw_files/"
        print("raw_files_path:", raw_files_path)
        dbutils.fs.mkdirs(raw_files_path)

        fpath_course, fpath_cohort = self.download_data_from_gcs(raw_files_path)
        df_course, df_cohort = self.read_and_validate_data(fpath_course, fpath_cohort)

        course_dataset_validated_path, cohort_dataset_validated_path = (
            self.write_data_to_delta_lake(df_course, df_cohort)
        )
        self.verify_delta_lake_write(
            course_dataset_validated_path, cohort_dataset_validated_path
        )
        # Setting task variables for downstream tasks
        dbutils.jobs.taskValues.set(
            key="course_dataset_validated_path", value=course_dataset_validated_path
        )
        dbutils.jobs.taskValues.set(
            key="cohort_dataset_validated_path", value=cohort_dataset_validated_path
        )
        dbutils.jobs.taskValues.set(key="job_root_dir", value=self.args.job_root_dir)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.
    Returns:
        argparse.Namespace: An object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Ingest course and cohort data for the SST pipeline."
    )
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace identifier"
    )
    parser.add_argument(
        "--databricks_institution_name",
        required=True,
        help="Databricksified institution name",
    )
    parser.add_argument(
        "--course_file_name", required=True, help="Name of the course data file"
    )
    parser.add_argument(
        "--cohort_file_name", required=True, help="Name of the cohort data file"
    )
    parser.add_argument(
        "--db_run_id", required=True, help="Databricks job run identifier"
    )
    parser.add_argument(
        "--gcp_bucket_name", required=True, help="Name of the GCP bucket"
    )
    parser.add_argument(
        "--job_root_dir", required=True, help="Folder path to store job output files"
    )
    parser.add_argument(
        "--custom_schemas_path",
        required=False,
        help="Folder path to store custom schemas folders",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    sys.path.append(
        f"/Volumes/staging_sst_01/{args.databricks_institution_name}_gold/gold_volume/inference_inputs"
    )
    logging.info(
        "Files in the inference inputs path: %s",
        os.listdir(
            f"/Volumes/staging_sst_01/{args.databricks_institution_name}_gold/gold_volume/inference_inputs"
        ),
    )
    try:
        converter_func = importlib.import_module("dataio")
        cohort_converter_func = converter_func.converter_func_cohort
        logging.info("Running task with custom cohort converter func")
    except Exception:
        cohort_converter_func = None
        logging.info("Running task with default cohort converter func")
    try:
        converter_func = importlib.import_module("dataio")
        course_converter_func = converter_func.converter_func_course
        logging.info("Running task with custom course converter func")
    except Exception:
        course_converter_func = None
        logging.info("Running task default course converter func")
    try:
        schemas = importlib.import_module("schemas")
        logging.info("Running task with custom schema")
    except Exception:
        from src.edvise.data_audit.schemas import raw_course, raw_cohort

        logging.info("Running task with default schema")

    task = DataIngestionTask(
        args,
        cohort_converter_func=cohort_converter_func,
        course_converter_func=course_converter_func,
    )
    task.run()
