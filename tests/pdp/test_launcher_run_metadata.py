"""Tests for parent launcher run id resolution and metadata payload."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from edvise.runtime.versioned_inference.run_metadata import (
    record_launcher_failures,
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


def _launcher_failure_context():
    return record_launcher_failures(
        catalog="dev_sst_02",
        databricks_institution_name="midway",
        model_name="retention",
        launcher_run_id="439619245566927",
        task="versioned_inference_launcher_validate",
    )


def test_record_launcher_failures_reraises_original_exception() -> None:
    with patch(
        "edvise.shared.dashboard_metadata.pipeline_runs.append_pipeline_run_event",
        return_value=True,
    ) as append:
        with pytest.raises(FileNotFoundError, match="missing bundle"):
            with _launcher_failure_context() as event:
                event.model_run_id = "train-123"
                event.archived_pipeline_version = "abc123def456"
                raise FileNotFoundError("missing bundle")

    kwargs = append.call_args.kwargs
    assert kwargs["event"] == "failed"
    assert kwargs["error_message"] == "missing bundle"
    assert kwargs["model_run_id"] == "train-123"
    assert kwargs["pipeline_version"] == "abc123def456"
    assert kwargs["payload"]["task"] == "versioned_inference_launcher_validate"


def test_record_launcher_failures_writes_nothing_on_success() -> None:
    with patch(
        "edvise.shared.dashboard_metadata.pipeline_runs.append_pipeline_run_event",
        return_value=True,
    ) as append:
        with _launcher_failure_context() as event:
            event.model_run_id = "train-123"

    append.assert_not_called()
