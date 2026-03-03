from __future__ import annotations

import json
import logging
import re
import typing as t
import pathlib
from datetime import datetime, timezone

LOGGER = logging.getLogger(__name__)


def _get_spark_session():
    try:
        from pyspark.sql import SparkSession  # type: ignore

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:
        # Best-effort: never fail the pipeline because observability couldn't init.
        LOGGER.warning("pipeline_runs: unable to create SparkSession (%s)", e)
        return None


def _get_current_catalog(spark) -> t.Optional[str]:
    try:
        # Databricks SQL function
        row = spark.sql("SELECT current_catalog() AS catalog").collect()
        if row and row[0] and row[0][0]:
            return str(row[0][0])
    except Exception:
        pass
    return None


def infer_catalog_from_volume_path(volume_path: t.Optional[str]) -> t.Optional[str]:
    """
    Infer UC catalog from a Unity Catalog Volume path like:
      /Volumes/<catalog>/<schema>/<volume>/<...>
    """
    if not volume_path:
        return None
    try:
        parts = pathlib.PurePosixPath(volume_path).parts
        for i, seg in enumerate(parts):
            if seg == "Volumes" and i + 1 < len(parts):
                return str(parts[i + 1])
    except Exception:
        return None
    return None


def infer_databricks_institution_name_from_volume_path(
    volume_path: t.Optional[str],
    *,
    suffix: str,
) -> t.Optional[str]:
    """
    Infer databricks_institution_name from a volume path segment like "<inst>_silver" or "<inst>_bronze".
    """
    if not volume_path:
        return None
    try:
        parts = pathlib.PurePosixPath(volume_path).parts
        for seg in parts:
            if seg.endswith(suffix):
                return str(seg[: -len(suffix)])
    except Exception:
        return None
    return None


_TS14_RE = re.compile(r"(\d{14})")


def parse_ts14_from_filename(filename: t.Optional[str]) -> t.Optional[datetime]:
    """
    Parse a trailing-ish 14-digit timestamp (YYYYMMDDHHMMSS) embedded in a filename.
    Example: "..._20250723040724.csv" -> 2025-07-23 04:07:24 UTC (naive).
    """
    if not filename:
        return None
    m = _TS14_RE.search(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    except Exception:
        return None


def _json_dumps(payload: t.Optional[dict[str, t.Any]]) -> t.Optional[str]:
    if not payload:
        return None
    try:
        return json.dumps(payload, default=str, sort_keys=True)
    except Exception as e:
        return json.dumps({"_error": f"json_dumps_failed: {e}", "_raw": str(payload)})


def append_pipeline_run_event(
    *,
    catalog: t.Optional[str],
    run_id: t.Optional[str],
    run_type: str,
    task_key: str,
    event: str,
    institution_id: t.Optional[str] = None,
    databricks_institution_name: t.Optional[str] = None,
    cohort_dataset_name: t.Optional[str] = None,
    course_dataset_name: t.Optional[str] = None,
    dataset_ts: t.Optional[datetime] = None,
    term_filter: t.Optional[str] = None,
    model_run_id: t.Optional[str] = None,
    experiment_id: t.Optional[str] = None,
    model_name: t.Optional[str] = None,
    model_card_path: t.Optional[str] = None,
    pipeline_version: t.Optional[str] = None,
    error_message: t.Optional[str] = None,
    payload: t.Optional[dict[str, t.Any]] = None,
    schema: str = "default",
    table: str = "pipeline_runs",
) -> bool:
    """
    Append a single observability event row to a UC Delta table.

    Table target (default): <catalog>.default.pipeline_runs

    This is deliberately best-effort: failures here should never fail the pipeline.
    """
    try:
        if not run_id:
            # If we can't correlate, don't write garbage.
            LOGGER.warning(
                "pipeline_runs: skipping event write because run_id is empty (%s/%s/%s)",
                run_type,
                task_key,
                event,
            )
            return False

        spark = _get_spark_session()
        if spark is None:
            return False

        if not catalog:
            catalog = _get_current_catalog(spark)
        if not catalog:
            LOGGER.warning(
                "pipeline_runs: skipping event write because catalog is unknown (run_id=%s)",
                run_id,
            )
            return False

        table_path = f"{catalog}.{schema}.{table}"

        row: dict[str, t.Any] = {
            "event_ts": datetime.now(timezone.utc),
            "run_id": str(run_id),
            "run_type": str(run_type),
            "task_key": str(task_key),
            "event": str(event),
            "institution_id": institution_id,
            "databricks_institution_name": databricks_institution_name,
            "cohort_dataset_name": cohort_dataset_name,
            "course_dataset_name": course_dataset_name,
            "dataset_ts": dataset_ts,
            "term_filter": term_filter,
            "model_run_id": model_run_id,
            "experiment_id": experiment_id,
            "model_name": model_name,
            "model_card_path": model_card_path,
            "pipeline_version": pipeline_version,
            "error_message": error_message,
            "payload_json": _json_dumps(payload),
        }

        df = spark.createDataFrame([row])
        (
            df.write.format("delta")
            .mode("append")
            .option("mergeSchema", "true")
            .saveAsTable(table_path)
        )
        return True
    except Exception as e:
        LOGGER.warning(
            "pipeline_runs: failed to append event (run_id=%s, task=%s, event=%s): %s",
            run_id,
            task_key,
            event,
            e,
        )
        return False
