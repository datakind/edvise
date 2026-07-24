"""Pipeline run metadata for the versioned inference launcher."""

from __future__ import annotations

import logging
import os
from typing import Any

LOGGER = logging.getLogger(__name__)

VERSIONED_INFERENCE_LAUNCHER_RUN_TYPE = "versioned_inference_launcher"
_UNRESOLVED_JOB_RUN_ID = "{{job.run_id}}"


def get_databricks_run_id() -> str | None:
    """
    Best-effort launcher job run id from the Databricks driver environment.

    Prefer :func:`resolve_launcher_run_id` with the job parameter
    ``launcher_run_id`` (default ``{{job.run_id}}``) instead of this alone.
    """
    for key in ("DATABRICKS_RUN_ID",):
        raw = os.environ.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return None


def resolve_launcher_run_id(cli_value: str | None = None) -> str | None:
    """
    Resolve the parent launcher job run id.

    Prefer the Databricks job parameter ``{{job.run_id}}`` (passed as
    ``--launcher_run_id``). Fall back to the driver env only when the CLI value
    is missing or still an unresolved template.
    """
    raw = (cli_value or "").strip()
    if raw and raw != _UNRESOLVED_JOB_RUN_ID:
        return raw
    return get_databricks_run_id()


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

    ``run_id`` is the parent launcher job run id. When a child inference run is
    submitted, ``payload`` records the parent → child link
    (``parent_launcher_run_id``, ``child_inference_run_id``, ``db_run_id``).

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
    body.setdefault("parent_launcher_run_id", str(run_id))
    # Child inference silver tables use parent launcher run id as db_run_id.
    body.setdefault("db_run_id", str(run_id))
    if child_inference_run_id is not None:
        body["child_inference_run_id"] = str(child_inference_run_id)

    try:
        from edvise.shared.dashboard_metadata.pipeline_runs import (
            append_pipeline_run_event,
        )
    except ImportError as exc:
        logger.warning(
            "versioned_inference_launcher: pipeline_runs import failed: %s", exc
        )
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
