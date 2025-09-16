import os
import argparse
import logging
import json
import typing as t
import importlib
import pandas as pd

from google.cloud import storage
from google.api_core.exceptions import Forbidden, NotFound
import google.auth


def in_databricks() -> bool:
    # Both of these are present on DBR clusters
    return bool(os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER"))


def get_dbutils():
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore

        return dbutils
    except Exception:
        return None


def active_gcp_identity() -> str:
    try:
        creds, _ = google.auth.default()
        # Best-effort extraction of a principal
        for attr in (
            "service_account_email",
            "service_account_email_address",
            "service_account",
        ):
            if hasattr(creds, attr):
                return str(getattr(creds, attr))
        return str(type(creds))
    except Exception:
        return "unknown"


def get_spark_session_or_none():
    if not in_databricks():
        return None
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:
        logging.warning("Spark not available: %s", e)
        return None


ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class DataIngestionTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils()
        self.spark_session = get_spark_session_or_none()
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.args.gcp_bucket_name)

        # Strict would be outside DBR (lenient inside DBR unless overridden via flag)
        # Basically, we don't want to raise google errors when we're only running from backend/DBR
        self.strict = not in_databricks()

    def _logging(self, code: str, message: str, extra: dict | None = None) -> None:
        payload = {"code": code, "message": message, "extra": (extra or {})}
        logging.error("AUDIT %s", json.dumps(payload))

    def download_data_from_gcs(
        self, internal_pipeline_path: str
    ) -> tuple[str | None, str | None]:
        sst_container_folder = "validated"
        course_blob_name = f"{sst_container_folder}/{self.args.course_file_name}"
        cohort_blob_name = f"{sst_container_folder}/{self.args.cohort_file_name}"

        course_file_path = os.path.join(
            internal_pipeline_path, self.args.course_file_name
        )
        cohort_file_path = os.path.join(
            internal_pipeline_path, self.args.cohort_file_name
        )

        ident = active_gcp_identity()

        try:
            # course
            self.bucket.blob(course_blob_name).download_to_filename(course_file_path)
            logging.info("Course data downloaded: %s", course_file_path)

            # cohort
            self.bucket.blob(cohort_blob_name).download_to_filename(cohort_file_path)
            logging.info("Cohort data downloaded: %s", cohort_file_path)

            return course_file_path, cohort_file_path

        except Forbidden:
            msg = (
                f"GCS 403 for identity '{ident}' on "
                f"gs://{self.args.gcp_bucket_name}/{course_blob_name} or /{cohort_blob_name}. "
                f"Grant roles/storage.objectViewer at bucket level."
            )
            if self.strict:
                raise
            self._logging(
                "gcs_forbidden",
                msg,
                {"identity": ident, "bucket": self.args.gcp_bucket_name},
            )
            return None, None

        except NotFound:
            msg = (
                "GCS 404 on one of the paths. "
                "Check object names and 'validated/' prefix."
            )
            if self.strict:
                raise
            self._logging("gcs_not_found", msg, {"bucket": self.args.gcp_bucket_name})
            return None, None

    def run(self):
        bronze_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_bronze/bronze_volume"
        )
        landing_dir = os.path.join(bronze_root, "inference_inputs", self.args.db_run_id)
        os.makedirs(landing_dir, exist_ok=True)

        fpath_course, fpath_cohort = self.download_data_from_gcs(landing_dir)

        if fpath_course:
            self.dbutils.jobs.taskValues.set(
                key="course_dataset_validated_path", value=fpath_course
            )
        if fpath_cohort:
            self.dbutils.jobs.taskValues.set(
                key="cohort_dataset_validated_path", value=fpath_cohort
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest course and cohort data for the SST pipeline."
    )
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    parser.add_argument("--course_file_name", required=True)
    parser.add_argument("--cohort_file_name", required=True)
    parser.add_argument("--db_run_id", required=True)
    parser.add_argument("--gcp_bucket_name", required=True)
    parser.add_argument("--custom_schemas_path", required=False)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    # Ensure inference_inputs exists before listdir/imports
    base_inputs = f"/Volumes/{args.DB_workspace}/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
    os.makedirs(base_inputs, exist_ok=True)
    logging.info("Files in inference inputs path: %s", os.listdir(base_inputs))

    # Optional dynamic imports for converters/schemas
    for name, attr, desc in [
        ("dataio", "converter_func_cohort", "custom cohort converter func"),
        ("dataio", "converter_func_course", "custom course converter func"),
        ("schemas", None, "custom schema"),
    ]:
        try:
            mod = importlib.import_module(name)
            if attr:
                getattr(mod, attr)
            logging.info("Running task with %s", desc)
        except Exception:
            logging.info("Running task with default %s", desc.split(" custom ")[-1])

    task = DataIngestionTask(args)
    if hasattr(args, "strict") and args.strict:
        task.strict = True
    task.run()
