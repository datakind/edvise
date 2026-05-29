import os
import argparse
import logging
import json
import typing as t
import importlib
import pandas as pd
import sys
from google.api_core.exceptions import Forbidden, NotFound

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from edvise.shared.dashboard_metadata.pipeline_runs import (
    append_pipeline_run_event,
    parse_timestamp_from_filename,
)
from edvise.utils.databricks import (
    local_fs_path,
    get_dbutils_or_none,
    get_latest_uc_model_run_id,
    find_file_in_run_folder,
    get_spark_session_or_none,
    in_databricks,
)
from edvise.utils.gcs import (
    active_gcp_identity,
    download_gcs_uri_to_filename,
    get_storage_client,
)

# Model names from get_model_name() are already UC-compatible


ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class DataIngestionTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils_or_none()
        self.spark_session = get_spark_session_or_none()
        self.storage_client = get_storage_client()

        # Strict would be outside DBR (lenient inside DBR unless overridden via flag)
        # Basically, we don't want to raise google errors when we're only running from backend/DBR
        self.strict = not in_databricks()
        # Populated by run() for downstream observability logging
        self.model_run_id: str | None = None
        self.config_file_path: str | None = None

    def _logging(self, code: str, message: str, extra: dict | None = None) -> None:
        payload = {"code": code, "message": message, "extra": (extra or {})}
        logging.error("AUDIT %s", json.dumps(payload))

    def download_data_from_gcs(
        self, internal_pipeline_path: str
    ) -> tuple[str | None, str | None]:
        sst_container_folder = "validated"
        course_blob_name = f"{sst_container_folder}/{self.args.course_file_name}"
        cohort_blob_name = f"{sst_container_folder}/{self.args.cohort_file_name}"
        course_uri = f"gs://{self.args.gcp_bucket_name}/{course_blob_name}"
        cohort_uri = f"gs://{self.args.gcp_bucket_name}/{cohort_blob_name}"

        course_file_path = os.path.join(
            internal_pipeline_path, self.args.course_file_name
        )
        cohort_file_path = os.path.join(
            internal_pipeline_path, self.args.cohort_file_name
        )

        ident = active_gcp_identity()

        try:
            # course
            download_gcs_uri_to_filename(
                course_uri,
                course_file_path,
                storage_client=self.storage_client,
            )
            logging.info("Course data downloaded: %s", course_file_path)

            # cohort
            download_gcs_uri_to_filename(
                cohort_uri,
                cohort_file_path,
                storage_client=self.storage_client,
            )
            logging.info("Cohort data downloaded: %s", cohort_file_path)

            return course_file_path, cohort_file_path

        except Forbidden:
            msg = (
                f"GCS 403 for identity '{ident}' on "
                f"{course_uri} or {cohort_uri}. "
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
        os.makedirs(local_fs_path(landing_dir), exist_ok=True)

        fpath_course, fpath_cohort = self.download_data_from_gcs(landing_dir)

        # Only set task values if dbutils is available (Databricks context)
        if self.dbutils:
            if fpath_course:
                self.dbutils.jobs.taskValues.set(
                    key="course_dataset_validated_path", value=fpath_course
                )
            if fpath_cohort:
                self.dbutils.jobs.taskValues.set(
                    key="cohort_dataset_validated_path", value=fpath_cohort
                )

        model_run_id = get_latest_uc_model_run_id(
            self.args.model_name,
            self.args.DB_workspace,
            self.args.databricks_institution_name,
        )
        self.model_run_id = model_run_id
        # Run root: <silver>/<model_run_id>/  (config may be here or in training/ or inference/)
        silver_run_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_silver/silver_volume/{model_run_id}"
        )
        config_file_path = find_file_in_run_folder(silver_run_root)
        self.config_file_path = str(config_file_path)
        logging.info("Using config file path: %s", config_file_path)
        if self.dbutils:
            self.dbutils.jobs.taskValues.set(
                key="config_file_path", value=str(config_file_path)
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
    parser.add_argument("--model_name", required=True)
    # parser.add_argument("--config_file_path", required=False)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()

    # Observability: append-only run events
    dataset_ts = parse_timestamp_from_filename(
        args.cohort_file_name
    ) or parse_timestamp_from_filename(args.course_file_name)
    append_pipeline_run_event(
        catalog=args.DB_workspace,
        run_id=args.db_run_id,
        run_type="inference",
        event="started",
        databricks_institution_name=args.databricks_institution_name,
        cohort_dataset_name=args.cohort_file_name,
        course_dataset_name=args.course_file_name,
        dataset_ts=dataset_ts,
        payload={"gcp_bucket_name": args.gcp_bucket_name},
    )

    # Ensure inference_inputs exists before listdir/imports
    base_inputs = f"/Volumes/{args.DB_workspace}/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
    os.makedirs(local_fs_path(base_inputs), exist_ok=True)
    logging.info(
        "Files in inference inputs path: %s", os.listdir(local_fs_path(base_inputs))
    )

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
    try:
        task.run()
        append_pipeline_run_event(
            catalog=args.DB_workspace,
            run_id=args.db_run_id,
            run_type="inference",
            event="completed",
            databricks_institution_name=args.databricks_institution_name,
            cohort_dataset_name=args.cohort_file_name,
            course_dataset_name=args.course_file_name,
            dataset_ts=dataset_ts,
            model_run_id=task.model_run_id,
            payload={"config_file_path": task.config_file_path},
        )
    except Exception as e:
        append_pipeline_run_event(
            catalog=args.DB_workspace,
            run_id=args.db_run_id,
            run_type="inference",
            event="failed",
            databricks_institution_name=args.databricks_institution_name,
            cohort_dataset_name=args.cohort_file_name,
            course_dataset_name=args.course_file_name,
            dataset_ts=dataset_ts,
            model_run_id=getattr(task, "model_run_id", None),
            error_message=str(e),
        )
        raise
