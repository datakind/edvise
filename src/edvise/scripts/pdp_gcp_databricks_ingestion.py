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
            course_file_path = os.path.join(
                internal_pipeline_path, self.args.course_file_name
            )
            course_blob.download_to_filename(course_file_path)
            logging.info("Course data downloaded from GCS: %s", course_file_path)

            # Download cohort data from GCS
            cohort_blob_name = f"{sst_container_folder}/{self.args.cohort_file_name}"
            cohort_blob = self.bucket.blob(cohort_blob_name)
            cohort_file_path = os.path.join(
                internal_pipeline_path, self.args.cohort_file_name
            )
            cohort_blob.download_to_filename(cohort_file_path)
            logging.info("Cohort data downloaded from GCS: %s", cohort_file_path)

            return course_file_path, cohort_file_path
        except Exception as e:
            logging.error(f"GCS download error: {e}")
            raise

    def run(self):
        """
        Executes the data ingestion task.
        """
        # Use institution bronze volume instead of job_root_dir
        bronze_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_bronze/bronze_volume"
        )

        # Put raw inputs for inference under a stable subfolder in bronze
        # e.g., /Volumes/.../bronze_volume/inference_inputs/validated/<run-id or date>/
        landing_dir = os.path.join(
            bronze_root, "inference_inputs", "validated", self.args.db_run_id
        )

        # Ensure the directory exists (Volumes are accessible as local paths)
        os.makedirs(landing_dir, exist_ok=True)

        fpath_course, fpath_cohort = self.download_data_from_gcs(landing_dir)

        # Setting task variables for downstream tasks
        dbutils.jobs.taskValues.set(
            key="course_dataset_validated_path",
            value=fpath_course,
        )
        dbutils.jobs.taskValues.set(
            key="cohort_dataset_validated_path",
            value=fpath_cohort,
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

        logging.info("Running task with default schema")

    task = DataIngestionTask(args)
    task.run()
