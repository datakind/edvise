"""
Legacy inference data ingestion: batch GCS landing (mirrors ES/PDP ``data_ingestion``).

Legacy schools upload through the same signed-URL -> ``validated/`` GCS flow as ES/GenAI
schools, but the legacy inference pipeline historically had no explicit batch ingest task:
``legacy_preprocessing`` discovered bronze files by keyword search under the institution's
top-level ``gcs_uploads/`` folder instead of a deterministic ``batch_id``-scoped landing dir.

This script closes that gap by running the same reuse-or-download batch ingest used by ES
(``run_batch_gcs_inference_ingest``) and publishing ``bronze_batch_dir`` as a task value so
``legacy_preprocessing`` can search the batch-scoped dir first, falling back to the existing
top-level keyword search when no batch was supplied (e.g. older/ manually-triggered runs).

Like ES/PDP's ``data_ingestion``, this task also resolves the registered model's latest
``run_id`` and, from its training run in the silver volume, the ``config_file_path`` and
``features_table_path`` used to train it — publishing both as task values for downstream
tasks (``legacy_preprocessing``, ``inference_h2o``). This folds in what was previously a
separate ``inference_setup`` task (``legacy_inference_inputs.py``), for parity with
ES/PDP where this resolution already lives in ``data_ingestion``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
# Layout: <git_root>/src/edvise/scripts/<this_file>
# Databricks spark_python_task often exec()s this file without defining __file__.
_here = globals().get("__file__")
if _here:
    _script_dir = os.path.dirname(os.path.abspath(_here))
else:
    _argv0 = os.path.abspath(sys.argv[0]) if sys.argv else ""
    if _argv0.endswith(".py") and os.path.isfile(_argv0):
        _script_dir = os.path.dirname(_argv0)
    else:
        _script_dir = os.path.abspath(os.getcwd())
_src_root = os.path.abspath(os.path.join(_script_dir, "..", ".."))
if os.path.isdir(_src_root) and os.path.isdir(os.path.join(_src_root, "edvise")):
    if _src_root not in sys.path:
        sys.path.insert(0, _src_root)

from edvise.dataio.batch_gcs_inference_ingest import (
    DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS,
    DEFAULT_BRONZE_SYNC_WAIT_SECONDS,
    BatchIngestResult,
    run_batch_gcs_inference_ingest,
    set_batch_ingest_task_values,
    should_skip_batch_ingest,
)
from edvise.utils.databricks import (
    find_file_in_run_folder,
    get_dbutils_or_none,
    get_latest_uc_model_run_id,
)
from edvise.utils.gcs import DEFAULT_GCS_PREFIX


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest batch validated GCS inputs for legacy inference "
            "(reuse upload-time bronze sync or download from GCS)."
        )
    )
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    parser.add_argument("--gcp_bucket_name", required=True)
    parser.add_argument("--db_run_id", required=True)
    parser.add_argument(
        "--model_name",
        required=True,
        help=(
            "Registered model short name, or full UC three-level name "
            "(catalog.institution_gold.<short>). Used to resolve the model's latest "
            "run_id and, from it, config_file_path/features_table_path."
        ),
    )
    parser.add_argument(
        "--batch_id",
        default="",
        help="Batch UUID (hex) for gcs_uploads/{batch_id}/.",
    )
    parser.add_argument(
        "--validated_blob_paths_json",
        default="[]",
        help='JSON array of full GCS object paths, e.g. ["validated/cohort.csv"].',
    )
    parser.add_argument(
        "--gcs_source_prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Prefix inside the bucket (default: validated/).",
    )
    parser.add_argument(
        "--max_objects",
        type=int,
        default=1_000,
        help="Safety cap on number of objects to copy (default: 1000).",
    )
    parser.add_argument(
        "--require_at_least_one_file",
        choices=("true", "false"),
        default="false",
        help=(
            "If true, fail when zero objects were copied (default: false, since legacy "
            "still supports the top-level gcs_uploads keyword-search fallback)."
        ),
    )
    parser.add_argument(
        "--bronze_sync_wait_seconds",
        type=float,
        default=DEFAULT_BRONZE_SYNC_WAIT_SECONDS,
        help=(
            "Seconds to poll for upload-time bronze sync before downloading from GCS "
            f"(default: {DEFAULT_BRONZE_SYNC_WAIT_SECONDS:.0f})."
        ),
    )
    parser.add_argument(
        "--bronze_sync_poll_interval_seconds",
        type=float,
        default=DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS,
        help=(
            "Poll interval while waiting for bronze sync "
            f"(default: {DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS:.0f})."
        ),
    )
    return parser.parse_args()


def resolve_model_config_and_features_table(
    model_name: str,
    db_workspace: str,
    databricks_institution_name: str,
) -> tuple[str, str]:
    """
    Look up the registered model's latest ``run_id`` in Unity Catalog and, from its
    training run in the silver volume, resolve the ``config_file_path`` and
    ``features_table_path`` used to train it. This ensures inference uses the exact
    same configuration as the trained model.

    Mirrors the former standalone ``inference_setup`` task (``legacy_inference_inputs.py``),
    now folded into ``data_ingestion`` for parity with ES/PDP.
    """
    model_run_id = get_latest_uc_model_run_id(
        model_name, db_workspace, databricks_institution_name
    )
    logging.info(
        "Found latest run_id for model '%s.%s_gold.%s': %s",
        db_workspace,
        databricks_institution_name,
        model_name,
        model_run_id,
    )

    silver_run_root = (
        f"/Volumes/{db_workspace}/"
        f"{databricks_institution_name}_silver/silver_volume/{model_run_id}"
    )
    logging.info("Looking for training artifacts in: %s", silver_run_root)

    config_file_path = find_file_in_run_folder(silver_run_root, keyword="config")
    features_table_path = find_file_in_run_folder(
        silver_run_root, keyword="features_table"
    )
    logging.info("Using config file: %s", config_file_path)
    logging.info("Using features table: %s", features_table_path)
    return config_file_path, features_table_path


def set_model_config_task_values(
    config_file_path: str, features_table_path: str
) -> None:
    """Publish resolved config/features_table paths to Databricks task values."""
    dbutils = get_dbutils_or_none()
    if not dbutils:
        logging.warning(
            "dbutils not available - config/features_table task values not set. "
            "This is expected in local/testing environments."
        )
        return
    dbutils.jobs.taskValues.set(key="config_file_path", value=config_file_path)
    dbutils.jobs.taskValues.set(key="features_table_path", value=features_table_path)
    logging.info("Config/features_table task values set for downstream tasks")


def run_legacy_gcp_databricks_ingestion(
    args: argparse.Namespace,
) -> BatchIngestResult:
    if should_skip_batch_ingest(
        is_genai_institution=False,
        validated_blob_paths_json=args.validated_blob_paths_json,
    ):
        logging.info(
            "Batch GCS inference ingest skipped (no validated blob paths); "
            "legacy_preprocessing will fall back to top-level gcs_uploads keyword search."
        )
        result = BatchIngestResult(
            bronze_batch_dir="",
            cohort_dataset_validated_path=None,
            course_dataset_validated_path=None,
            source="existing_bronze_batch",
            copied_count=0,
            skipped=True,
        )
    else:
        result = run_batch_gcs_inference_ingest(
            db_workspace=args.DB_workspace,
            databricks_institution_name=args.databricks_institution_name,
            gcp_bucket_name=args.gcp_bucket_name,
            batch_id=args.batch_id,
            validated_blob_paths_json=args.validated_blob_paths_json,
            db_run_id=args.db_run_id,
            gcs_source_prefix=args.gcs_source_prefix,
            require_at_least_one_file=args.require_at_least_one_file,
            max_objects=args.max_objects,
            is_genai_institution=False,
            bronze_sync_wait_seconds=args.bronze_sync_wait_seconds,
            bronze_sync_poll_interval_seconds=args.bronze_sync_poll_interval_seconds,
        )
    set_batch_ingest_task_values(result)

    # Model/config resolution is independent of the GCS batch ingest above (and its
    # skip status); it always runs so downstream tasks get config_file_path/
    # features_table_path regardless of whether a batch was supplied.
    config_file_path, features_table_path = resolve_model_config_and_features_table(
        model_name=args.model_name,
        db_workspace=args.DB_workspace,
        databricks_institution_name=args.databricks_institution_name,
    )
    set_model_config_task_values(config_file_path, features_table_path)

    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_arguments()
    args.require_at_least_one_file = args.require_at_least_one_file == "true"
    result = run_legacy_gcp_databricks_ingestion(args)
    if not result.skipped:
        logging.info(
            "Legacy data ingestion complete: source=%s bronze_batch_dir=%s copied=%s",
            result.source,
            result.bronze_batch_dir,
            result.copied_count,
        )


if __name__ == "__main__":
    main()
