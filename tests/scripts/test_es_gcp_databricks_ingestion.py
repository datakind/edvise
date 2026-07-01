"""Tests for combined ES GCP Databricks ingestion."""

from __future__ import annotations

import argparse
from unittest import mock

from edvise.dataio.batch_gcs_inference_ingest import BatchIngestResult
from edvise.dataio.inference_model_artifacts import EsInferenceArtifacts
from edvise.scripts import es_gcp_databricks_ingestion as script


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "DB_workspace": "edvise",
        "databricks_institution_name": "acme",
        "gcp_bucket_name": "bucket",
        "model_name": "retention_model",
        "db_run_id": "run-1",
        "batch_id": "batch1",
        "validated_blob_paths_json": '["validated/cohort.csv"]',
        "gcs_source_prefix": "validated/",
        "is_genai_institution": "false",
        "max_objects": 1000,
        "require_at_least_one_file": "true",
        "bronze_sync_wait_seconds": 300.0,
        "bronze_sync_poll_interval_seconds": 10.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_run_es_gcp_databricks_ingestion_skips_when_no_blob_paths() -> None:
    artifacts = EsInferenceArtifacts(
        model_run_id="run-abc",
        config_file_path="/cfg/config.toml",
        silver_run_root="/Volumes/edvise/acme_silver/silver_volume/run-abc",
    )
    with (
        mock.patch.object(
            script, "resolve_es_inference_artifacts", return_value=artifacts
        ),
        mock.patch.object(script, "set_inference_config_task_value") as set_config,
        mock.patch.object(script, "set_batch_ingest_task_values") as set_batch,
        mock.patch.object(script, "run_batch_gcs_inference_ingest") as run_batch,
    ):
        result = script.run_es_gcp_databricks_ingestion(
            _args(validated_blob_paths_json="[]")
        )

    set_config.assert_called_once_with("/cfg/config.toml")
    run_batch.assert_not_called()
    set_batch.assert_called_once()
    assert result is not None
    assert result.skipped is True


def test_run_es_gcp_databricks_ingestion_runs_batch_ingest() -> None:
    artifacts = EsInferenceArtifacts(
        model_run_id="run-abc",
        config_file_path="/cfg/config.toml",
        silver_run_root="/Volumes/edvise/acme_silver/silver_volume/run-abc",
    )
    batch_result = BatchIngestResult(
        bronze_batch_dir="/Volumes/edvise/acme_bronze/bronze_volume/gcs_uploads/batch1",
        cohort_dataset_validated_path=None,
        course_dataset_validated_path=None,
        source="existing_bronze_batch",
        copied_count=0,
        skipped=False,
    )
    with (
        mock.patch.object(
            script, "resolve_es_inference_artifacts", return_value=artifacts
        ),
        mock.patch.object(script, "set_inference_config_task_value"),
        mock.patch.object(
            script, "run_batch_gcs_inference_ingest", return_value=batch_result
        ) as run_batch,
        mock.patch.object(script, "set_batch_ingest_task_values") as set_batch,
    ):
        result = script.run_es_gcp_databricks_ingestion(_args())

    run_batch.assert_called_once()
    assert "raw_cohort_name" not in run_batch.call_args.kwargs
    assert "raw_course_name" not in run_batch.call_args.kwargs
    set_batch.assert_called_once_with(batch_result)
    assert result == batch_result
