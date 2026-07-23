"""Tests for legacy GCP Databricks ingestion (batch GCS ingest task for legacy inference)."""

from __future__ import annotations

import argparse
from unittest import mock

from edvise.dataio.batch_gcs_inference_ingest import BatchIngestResult
from edvise.scripts import legacy_gcp_databricks_ingestion as script


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "DB_workspace": "dev_sst_02",
        "databricks_institution_name": "john_jay_col",
        "gcp_bucket_name": "bucket",
        "db_run_id": "run-1",
        "batch_id": "batch1",
        "validated_blob_paths_json": '["validated/cohort.csv"]',
        "gcs_source_prefix": "validated/",
        "max_objects": 1000,
        "require_at_least_one_file": "false",
        "bronze_sync_wait_seconds": 300.0,
        "bronze_sync_poll_interval_seconds": 10.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_legacy_gcp_databricks_ingestion_skips_when_no_blob_paths() -> None:
    with (
        mock.patch.object(script, "set_batch_ingest_task_values") as set_batch,
        mock.patch.object(script, "run_batch_gcs_inference_ingest") as run_batch,
    ):
        result = script.run_legacy_gcp_databricks_ingestion(
            _args(validated_blob_paths_json="[]")
        )

    run_batch.assert_not_called()
    set_batch.assert_called_once()
    assert result.skipped is True
    assert result.bronze_batch_dir == ""


def test_run_legacy_gcp_databricks_ingestion_runs_batch_ingest() -> None:
    batch_result = BatchIngestResult(
        bronze_batch_dir=(
            "/Volumes/dev_sst_02/john_jay_col_bronze/bronze_volume/gcs_uploads/batch1"
        ),
        cohort_dataset_validated_path=None,
        course_dataset_validated_path=None,
        source="existing_bronze_batch",
        copied_count=0,
        skipped=False,
    )
    with (
        mock.patch.object(
            script, "run_batch_gcs_inference_ingest", return_value=batch_result
        ) as run_batch,
        mock.patch.object(script, "set_batch_ingest_task_values") as set_batch,
    ):
        result = script.run_legacy_gcp_databricks_ingestion(_args())

    run_batch.assert_called_once()
    assert run_batch.call_args.kwargs["is_genai_institution"] is False
    assert run_batch.call_args.kwargs["batch_id"] == "batch1"
    set_batch.assert_called_once_with(batch_result)
    assert result == batch_result
