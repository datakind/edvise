"""Tests for parent launcher run id resolution and metadata payload."""

from __future__ import annotations

from unittest.mock import patch

from edvise.runtime.versioned_inference.run_metadata import (
    record_versioned_inference_launcher_event,
    resolve_launcher_run_id,
)


def test_resolve_launcher_run_id_prefers_job_parameter() -> None:
    assert resolve_launcher_run_id("439619245566927") == "439619245566927"


def test_resolve_launcher_run_id_ignores_unresolved_template() -> None:
    assert resolve_launcher_run_id("{{job.run_id}}") is None


def test_resolve_launcher_run_id_none_when_missing() -> None:
    assert resolve_launcher_run_id("") is None
    assert resolve_launcher_run_id("{{job.run_id}}") is None


def test_record_launcher_event_uses_archived_pipeline_version() -> None:
    with patch(
        "edvise.shared.dashboard_metadata.pipeline_runs.append_pipeline_run_event",
        return_value=True,
    ) as append:
        ok = record_versioned_inference_launcher_event(
            catalog="dev_sst_02",
            event="completed",
            databricks_institution_name="midway",
            model_name="retention",
            model_run_id="train-123",
            archived_pipeline_version="abc123def456",
            launcher_run_id="439619245566927",
        )
    assert ok is True
    append.assert_called_once()
    kwargs = append.call_args.kwargs
    assert kwargs["pipeline_version"] == "abc123def456"
    assert kwargs["payload"]["archived_pipeline_version"] == "abc123def456"
