"""Resolve GenAI mapped cohort/course inputs for a webapp batch (Spark-free).

Looks up ``{catalog}.genai_mapping.pipeline_runs`` for an execute run whose
``input_file_paths`` reference ``gcs_uploads/{batch_id}``, then loads that run's
``pipeline_input/cohort.parquet`` and ``course.parquet`` from the silver volume
via the Databricks SDK (no Spark session required).
"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import (
    Disposition,
    ExecuteStatementRequestOnWaitTimeout,
    Format,
    StatementState,
)

from edvise.dataio.batch_gcs_inference_ingest import BRONZE_GCS_UPLOADS_SUBDIR
from edvise.genai.mapping.state._sql import PIPELINE_RUNS, lit, qualified_table
from edvise.runtime.versioned_inference.genai_registry import silver_volume_root
from edvise.utils.institution_naming import databricksify_inst_name

LOGGER = logging.getLogger(__name__)

# Terminal success statuses written by GenAI mapping execute.
_COMPLETED_RUN_STATUSES = frozenset({"complete", "completed"})

# Webapp batch ids are UUID hex strings (with or without hyphens).
_BATCH_ID_RE = re.compile(r"^[0-9a-fA-F-]{32,36}$")


@dataclass(frozen=True)
class GenaiMappedInputs:
    """Mapped cohort/course frames for a completed GenAI execute run."""

    execute_run_id: str
    df_cohort: pd.DataFrame
    df_course: Optional[pd.DataFrame] = None


def _validate_batch_id(batch_id: str) -> str:
    cleaned = batch_id.strip()
    if not _BATCH_ID_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid batch_id for GenAI mapped-input lookup: {batch_id!r}")
    return cleaned


def pipeline_input_parquet_paths(
    *,
    catalog: str,
    institution_id: str,
    execute_run_id: str,
) -> tuple[str, str]:
    """
    Return ``(cohort.parquet, course.parquet)`` UC volume paths for an execute run.

    Layout::

        /Volumes/{catalog}/{institution_id}_silver/silver_volume/
          genai_mapping/runs/execute/{execute_run_id}/pipeline_input/
    """
    run_id = execute_run_id.strip()
    if not run_id:
        raise ValueError("execute_run_id must be a non-empty string")
    silver = silver_volume_root(catalog, institution_id).as_posix()
    root = f"{silver}/genai_mapping/runs/execute/{run_id}/pipeline_input"
    return f"{root}/cohort.parquet", f"{root}/course.parquet"


def _download_volume_bytes(workspace: WorkspaceClient, volume_path: str) -> bytes:
    response = workspace.files.download(volume_path)
    stream = getattr(response, "contents", None)
    if stream is None:
        raise FileNotFoundError(
            f"Databricks download returned no contents: {volume_path}"
        )
    raw = stream.read()
    if isinstance(raw, str):
        return raw.encode("utf-8")
    return bytes(raw)


def _read_volume_parquet(workspace: WorkspaceClient, volume_path: str) -> pd.DataFrame:
    raw = _download_volume_bytes(workspace, volume_path)
    return pd.read_parquet(io.BytesIO(raw))


def _execute_sql(
    workspace: WorkspaceClient,
    *,
    warehouse_id: str,
    statement: str,
) -> list[dict[str, Any]]:
    resp = workspace.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=statement,
        disposition=Disposition.INLINE,
        format=Format.JSON_ARRAY,
        wait_timeout="30s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    if not resp.status or resp.status.state != StatementState.SUCCEEDED:
        raise TimeoutError(
            "Databricks SQL did not succeed for GenAI mapped-input lookup "
            f"(state={getattr(resp.status, 'state', None)!r})"
        )
    cols: list[str] = [
        str(c.name)
        for c in (resp.manifest.schema.columns or [])  # type: ignore[union-attr]
        if c.name is not None
    ]
    rows = resp.result.data_array or []  # type: ignore[union-attr]
    return [dict(zip(cols, row)) for row in rows]


def lookup_genai_execute_runs_for_batch(
    workspace: WorkspaceClient,
    *,
    warehouse_id: str,
    catalog: str,
    institution_id: str,
    batch_id: str,
) -> list[dict[str, Any]]:
    """
    Return execute-mode ``pipeline_runs`` rows tied to ``gcs_uploads/{batch_id}``.

    Newest ``updated_at`` first.
    """
    cleaned_batch = _validate_batch_id(batch_id)
    needle = f"{BRONZE_GCS_UPLOADS_SUBDIR}/{cleaned_batch}"
    table = qualified_table(catalog, PIPELINE_RUNS)
    sql = f"""
SELECT execute_run_id, status, updated_at, input_file_paths
FROM {table}
WHERE institution_id = {lit(institution_id)}
  AND execute_run_id IS NOT NULL
  AND input_file_paths LIKE {lit("%" + needle + "%")}
ORDER BY updated_at DESC
LIMIT 20
"""
    return _execute_sql(workspace, warehouse_id=warehouse_id, statement=sql)


def resolve_genai_mapped_inputs_for_batch(
    workspace: WorkspaceClient,
    *,
    warehouse_id: str,
    catalog: str,
    institution_name: str,
    batch_id: str,
) -> Optional[GenaiMappedInputs]:
    """
    Resolve mapped cohort/course DataFrames for a GenAI webapp batch.

    Returns
    -------
    GenaiMappedInputs
        Latest completed execute run for ``gcs_uploads/{batch_id}``, with parquets loaded.
    None
        No completed mapped run for this batch yet (or cohort file empty).

    Raises
    ------
    ValueError
        Invalid ``batch_id`` / institution / catalog arguments.
    Exception
        Databricks SQL lookup or volume parquet download/load failure.
    """
    inst_slug = databricksify_inst_name(institution_name)
    rows = lookup_genai_execute_runs_for_batch(
        workspace,
        warehouse_id=warehouse_id,
        catalog=catalog,
        institution_id=inst_slug,
        batch_id=batch_id,
    )
    if not rows:
        return None

    complete_rows = [
        r
        for r in rows
        if str(r.get("status") or "").strip().lower() in _COMPLETED_RUN_STATUSES
    ]
    if not complete_rows:
        return None

    execute_run_id = str(complete_rows[0].get("execute_run_id") or "").strip()
    if not execute_run_id:
        return None

    cohort_path, course_path = pipeline_input_parquet_paths(
        catalog=catalog,
        institution_id=inst_slug,
        execute_run_id=execute_run_id,
    )
    LOGGER.info(
        "Loading GenAI mapped EDA inputs "
        "(inst=%s execute_run_id=%s batch_id=%s cohort=%s)",
        inst_slug,
        execute_run_id,
        batch_id.strip(),
        cohort_path,
    )
    df_cohort = _read_volume_parquet(workspace, cohort_path)
    df_course = _read_volume_parquet(workspace, course_path)

    if df_cohort is None or df_cohort.empty:
        return None

    return GenaiMappedInputs(
        execute_run_id=execute_run_id,
        df_cohort=df_cohort,
        df_course=df_course if df_course is not None else None,
    )
