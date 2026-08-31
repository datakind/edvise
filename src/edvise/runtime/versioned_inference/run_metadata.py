"""Pipeline run metadata for the versioned inference launcher."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)

VERSIONED_INFERENCE_LAUNCHER_RUN_TYPE = "versioned_inference_launcher"
_UNRESOLVED_JOB_RUN_ID = "{{job.run_id}}"


def resolve_launcher_run_id(cli_value: str | None = None) -> str | None:
    """
    Resolve the parent launcher job run id from the Databricks job parameter.

    Expect ``launcher_run_id`` (default ``{{job.run_id}}``) to be passed on the CLI.
    """
    raw = (cli_value or "").strip()
    if raw and raw != _UNRESOLVED_JOB_RUN_ID:
        return raw
    return None


def record_versioned_inference_launcher_event(
    *,
    catalog: str,
    event: str,
    databricks_institution_name: str,
    model_name: str,
    model_run_id: str | None = None,
    archived_pipeline_version: str | None = None,
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

    ``archived_pipeline_version`` is the git SHA or release tag from training
    config (``config*.toml``) — the ref inference runs at, not the launcher deploy version.
    The same value is written to the ``pipeline_version`` column for dashboard
    compatibility.

    Best-effort: observability failures must not fail the launcher.
    """
    run_id = resolve_launcher_run_id(launcher_run_id)
    if not run_id:
        logger.warning(
            "versioned_inference_launcher: skip pipeline_runs write (no launcher run_id)"
        )
        return False

    body: dict[str, Any] = dict(payload or {})
    body.setdefault("launcher_job", "edvise_versioned_inference_launcher")
    body.setdefault("model_name", model_name)
    body.setdefault("parent_launcher_run_id", str(run_id))
    body.setdefault("db_run_id", str(run_id))
    if archived_pipeline_version is not None:
        body.setdefault("archived_pipeline_version", archived_pipeline_version)
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
        pipeline_version=archived_pipeline_version,
        error_message=error_message,
        payload=body,
    )


@dataclass
class LauncherEventContext:
    """Identifiers a launcher task resolves as it runs, for the ``failed`` event."""

    model_run_id: str | None = None
    archived_pipeline_version: str | None = None


@contextmanager
def record_launcher_failures(
    *,
    catalog: str,
    databricks_institution_name: str,
    model_name: str,
    launcher_run_id: str | None,
    task: str,
    logger: logging.Logger = LOGGER,
) -> Iterator[LauncherEventContext]:
    """
    Record a ``failed`` launcher event, then re-raise the original exception.

    Databricks fails a ``spark_python_task`` on any non-zero exit, so re-raising
    keeps the underlying traceback in the run output instead of replacing it with
    an exit code.
    """
    context = LauncherEventContext()
    try:
        yield context
    except Exception as exc:
        logger.error("%s failed: %s", task, exc)
        record_versioned_inference_launcher_event(
            catalog=catalog,
            event="failed",
            databricks_institution_name=databricks_institution_name,
            model_name=model_name,
            model_run_id=context.model_run_id,
            archived_pipeline_version=context.archived_pipeline_version,
            launcher_run_id=launcher_run_id,
            error_message=str(exc),
            payload={"task": task},
            logger=logger,
        )
        raise
