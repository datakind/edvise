"""
Batch GCS inference ingest for ES and Legacy pipelines.

Polls for upload-time bronze sync, then downloads validated objects from GCS into
``gcs_uploads/{batch_id}/`` only when sync has not landed yet.
"""

from __future__ import annotations

import argparse
import logging
import sys

from edvise.dataio.batch_gcs_inference_ingest import (
    DEFAULT_BRONZE_SYNC_POLL_INTERVAL_SECONDS,
    DEFAULT_BRONZE_SYNC_WAIT_SECONDS,
    BatchIngestResult,
    run_batch_gcs_inference_ingest,
    set_batch_ingest_task_values,
    should_skip_batch_ingest,
)
from edvise.utils.gcs import DEFAULT_GCS_PREFIX


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resolve or download batch-scoped validated GCS inputs for inference."
        )
    )
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    parser.add_argument("--gcp_bucket_name", required=True)
    parser.add_argument(
        "--batch_id",
        default="",
        help="Batch UUID (hex) used for gcs_uploads/{batch_id}/ when bronze sync ran.",
    )
    parser.add_argument(
        "--validated_blob_paths_json",
        default="[]",
        help='JSON array of full GCS object paths, e.g. ["validated/cohort.csv"].',
    )
    parser.add_argument("--db_run_id", required=True)
    parser.add_argument(
        "--gcs_source_prefix",
        default=DEFAULT_GCS_PREFIX,
        help="Prefix inside the bucket (default: validated/).",
    )
    parser.add_argument(
        "--is_genai_institution",
        default="false",
        help="When true, GenAI execute still uses batch ingest when blob paths are set.",
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
        default="true",
        help="If true, fail when zero objects were copied (default: true).",
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    args = parse_arguments()
    args.require_at_least_one_file = args.require_at_least_one_file == "true"

    if should_skip_batch_ingest(
        is_genai_institution=args.is_genai_institution,
        validated_blob_paths_json=args.validated_blob_paths_json,
    ):
        logging.info("Batch GCS inference ingest skipped.")
        set_batch_ingest_task_values(
            BatchIngestResult(
                bronze_batch_dir="",
                cohort_dataset_validated_path=None,
                course_dataset_validated_path=None,
                source="existing_bronze_batch",
                copied_count=0,
                skipped=True,
            )
        )
        return

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
        is_genai_institution=args.is_genai_institution,
        bronze_sync_wait_seconds=args.bronze_sync_wait_seconds,
        bronze_sync_poll_interval_seconds=args.bronze_sync_poll_interval_seconds,
    )
    set_batch_ingest_task_values(result)
    logging.info(
        "Batch ingest complete: source=%s bronze_batch_dir=%s copied=%s",
        result.source,
        result.bronze_batch_dir,
        result.copied_count,
    )


if __name__ == "__main__":
    main()
