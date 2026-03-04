from __future__ import annotations

import json
import logging
import re
import typing as t
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


_TS14_RE = re.compile(r"(\d{14})")
_SST_INST_ID_RE = re.compile(r"^[0-9a-f]{32}$", flags=re.IGNORECASE)


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


def _looks_like_sst_inst_id(value: t.Optional[str]) -> bool:
    if not value:
        return False
    return _SST_INST_ID_RE.fullmatch(str(value)) is not None


def _resolve_inst_id_from_uc(
    *,
    spark,
    catalog: str,
    databricks_institution_name: t.Optional[str],
    schema: str = "default",
    table: str = "institutions",
) -> t.Optional[str]:
    """
    Best-effort lookup of SST WebApp inst_id (institutions.institution_id) by
    databricks_institution_name.

    This must never fail the pipeline. On any error, return None.
    """
    if not databricks_institution_name:
        return None
    try:
        tbl = f"{catalog}.{schema}.{table}"
        # Avoid importing pyspark.sql.functions here so unit tests can run without pyspark.
        escaped = str(databricks_institution_name).replace("'", "''")
        rows = (
            spark.table(tbl)
            .where(f"databricks_institution_name = '{escaped}'")
            .select("institution_id")
            .limit(1)
            .collect()
        )
        if rows:
            # Row access is defensive across spark versions.
            d = rows[0].asDict() if hasattr(rows[0], "asDict") else None
            if isinstance(d, dict) and d.get("institution_id"):
                return str(d["institution_id"])
            # Fallback: assume first column is institution_id
            return str(rows[0][0])
    except Exception as e:
        LOGGER.warning(
            "pipeline_runs: failed to resolve inst_id from %s.%s.%s for databricks_institution_name=%r: %s",
            catalog,
            schema,
            table,
            databricks_institution_name,
            e,
        )
        return None


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
            LOGGER.warning(
                "pipeline_runs: skipping event write because catalog is empty (run_id=%s)",
                run_id,
            )
            return False

        table_path = f"{catalog}.{schema}.{table}"

        # Canonicalize institution_id to SST WebApp inst_id.
        # - If caller already supplied an inst_id, trust it.
        # - Else, resolve from the synced institutions table by databricks_institution_name.
        resolved_inst_id: t.Optional[str] = None
        if _looks_like_sst_inst_id(institution_id):
            resolved_inst_id = str(institution_id)
        else:
            resolved_inst_id = _resolve_inst_id_from_uc(
                spark=spark,
                catalog=catalog,
                databricks_institution_name=databricks_institution_name,
                schema=schema,
                table="institutions",
            )

        # Keep extra institution identifiers in payload_json for debugging/lineage.
        payload2: dict[str, t.Any] = dict(payload) if isinstance(payload, dict) else {}
        if databricks_institution_name and "databricks_institution_name" not in payload2:
            payload2["databricks_institution_name"] = databricks_institution_name
        if (
            institution_id
            and not _looks_like_sst_inst_id(institution_id)
            and "config_institution_id" not in payload2
        ):
            payload2["config_institution_id"] = institution_id

        row: dict[str, t.Any] = {
            "event_ts": datetime.now(timezone.utc),
            "run_id": str(run_id),
            "run_type": str(run_type),
            "task_key": str(task_key),
            "event": str(event),
            "institution_id": resolved_inst_id,
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
            "payload_json": _json_dumps(payload2),
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
