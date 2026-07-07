"""Pipeline run metadata for the versioned inference launcher."""

from __future__ import annotations

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

VERSIONED_INFERENCE_LAUNCHER_RUN_TYPE = "versioned_inference_launcher"


def get_databricks_run_id() -> str | None:
    """Best-effort launcher job run id from the Databricks driver environment."""
    for key in ("DATABRICKS_RUN_ID",):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def record_versioned_inference_launcher_event(
    *,
    catalog: str,
    event: str,
    databricks_institution_name: str,
    model_name: str,
    model_run_id: str | None = None,
    pipeline_version: str | None = None,
    launcher_run_id: str | None = None,
    child_inference_run_id: str | int | None = None,
    cohort_dataset_name: str | None = None,
    course_dataset_name: str | None = None,
    error_message: str | None = None,
    payload: dict[str, Any] | None = None,
    logger: logging.Logger = LOGGER,
) -> bool:
    """
    Upsert launcher lifecycle into ``<catalog>.default.pipeline_runs``.

    Best-effort: observability failures must not fail the launcher.
    """
    run_id = launcher_run_id or get_databricks_run_id()
    if not run_id:
        logger.warning(
            "versioned_inference_launcher: skip pipeline_runs write (no launcher run_id)"
        )
        return False

    body: dict[str, Any] = dict(payload or {})
    body.setdefault("launcher_job", "edvise_versioned_inference_launcher")
    body.setdefault("model_name", model_name)
    if child_inference_run_id is not None:
        body["child_inference_run_id"] = str(child_inference_run_id)

    try:
        from edvise.shared.dashboard_metadata.pipeline_runs import (
            append_pipeline_run_event,
        )
    except ImportError as exc:
        logger.warning("versioned_inference_launcher: pipeline_runs import failed: %s", exc)
        return False

    return append_pipeline_run_event(
        catalog=catalog,
        run_id=str(run_id),
        run_type=VERSIONED_INFERENCE_LAUNCHER_RUN_TYPE,
        event=event,
        databricks_institution_name=databricks_institution_name,
        cohort_dataset_name=cohort_dataset_name,
        course_dataset_name=course_dataset_name,
        model_run_id=model_run_id,
        pipeline_version=pipeline_version,
        error_message=error_message,
        payload=body,
    )
