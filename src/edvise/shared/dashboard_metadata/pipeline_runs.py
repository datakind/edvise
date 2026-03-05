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


def parse_timestamp_from_filename(filename: t.Optional[str]) -> t.Optional[datetime]:
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


def parse_ts14_from_filename(filename: t.Optional[str]) -> t.Optional[datetime]:
    """Backwards-compatible alias for parse_timestamp_from_filename()."""
    return parse_timestamp_from_filename(filename)


def _json_dumps(payload: t.Optional[dict[str, t.Any]]) -> t.Optional[str]:
    if not payload:
        return None
    try:
        return json.dumps(payload, default=str, sort_keys=True)
    except Exception as e:
        return json.dumps({"_error": f"json_dumps_failed: {e}", "_raw": str(payload)})


def _coalesce_institution_id(
    *,
    institution_id: t.Optional[str],
    databricks_institution_name: t.Optional[str],
) -> t.Optional[str]:
    # For the dashboard metadata tables, we key pipeline runs by the Databricks institution
    # identifier (the one used in UC schemas/volumes), not the SST WebApp inst_id.
    v = databricks_institution_name or institution_id
    v = str(v).strip() if v else None
    return v or None


def _opt_get(value: t.Any) -> t.Optional[str]:
    """
    Databricks context returns a mix of Scala Options and plain strings.
    Best-effort unwrap to a string.
    """
    if value is None:
        return None
    try:
        # Scala Option-like
        if hasattr(value, "get"):
            v = value.get()
            return None if v is None else str(v)
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return None


def _normalize_dbx_base_url(url: t.Optional[str]) -> t.Optional[str]:
    if not url:
        return None
    u = str(url).strip()
    if not u:
        return None
    # Some contexts return ".../api/2.0"; convert to UI base.
    if "/api/" in u:
        u = u.split("/api/")[0]
    return u.rstrip("/")


def _best_effort_databricks_run_url(*, run_id: str) -> t.Optional[str]:
    """
    Build a clickable Jobs UI URL for this run.

    Best-effort only: returns None if we can't read Databricks context (e.g. local tests).
    """
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore

        # databricks.sdk.runtime.dbutils is a dynamically-typed proxy; mypy stubs don't
        # expose the full notebook context surface area.
        dbx = t.cast(t.Any, dbutils)
        ctx = dbx.notebook.entry_point.getDbutils().notebook().getContext()

        base_url = None
        try:
            base_url = _normalize_dbx_base_url(_opt_get(ctx.apiUrl()))
        except Exception:
            base_url = None
        if not base_url:
            try:
                host = _opt_get(ctx.browserHostName())
                if host:
                    base_url = f"https://{host}".rstrip("/")
            except Exception:
                base_url = None

        workspace_id = None
        try:
            workspace_id = _opt_get(ctx.workspaceId())
        except Exception:
            workspace_id = None

        job_id = None
        try:
            tags = ctx.tags()
            # tags.get(key) returns an Option-like value in many runtimes.
            job_id = _opt_get(tags.get("jobId"))
            if not job_id:
                job_id = _opt_get(tags.get("job_id"))
            if not job_id:
                # Some runtimes expose a dict-like apply().
                try:
                    job_id = str(tags.apply("jobId"))
                except Exception:
                    job_id = None
        except Exception:
            job_id = None

        if not base_url or not job_id:
            return None

        # Common Databricks Jobs UI pattern.
        if workspace_id:
            return f"{base_url}/?o={workspace_id}#job/{job_id}/run/{run_id}"
        return f"{base_url}/#job/{job_id}/run/{run_id}"
    except Exception:
        return None


def append_pipeline_run_event(
    *,
    catalog: t.Optional[str],
    run_id: t.Optional[str],
    run_type: str,
    event: str,
    institution_id: t.Optional[str] = None,
    databricks_institution_name: t.Optional[str] = None,
    cohort_dataset_name: t.Optional[str] = None,
    course_dataset_name: t.Optional[str] = None,
    dataset_ts: t.Optional[datetime] = None,
    term_filter: t.Optional[str] = None,
    model_run_id: t.Optional[str] = None,
    experiment_id: t.Optional[str] = None,
    pipeline_version: t.Optional[str] = None,
    error_message: t.Optional[str] = None,
    payload: t.Optional[dict[str, t.Any]] = None,
    schema: str = "default",
    table: str = "pipeline_runs",
) -> bool:
    """
    Upsert (MERGE) a run-level row to a UC Delta table.

    Table target (default): <catalog>.default.pipeline_runs

    This is deliberately best-effort: failures here should never fail the pipeline.
    """
    try:
        if not run_id:
            # If we can't correlate, don't write garbage.
            LOGGER.warning(
                "pipeline_runs: skipping event write because run_id is empty (%s/%s)",
                run_type,
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

        resolved_institution_id = _coalesce_institution_id(
            institution_id=institution_id,
            databricks_institution_name=databricks_institution_name,
        )

        # Keep extra institution identifiers in payload_json for debugging/lineage.
        payload2: dict[str, t.Any] = dict(payload) if isinstance(payload, dict) else {}
        if (
            institution_id
            and databricks_institution_name
            and institution_id != databricks_institution_name
            and "config_institution_id" not in payload2
        ):
            payload2["config_institution_id"] = institution_id

        # Infer dataset timestamp from filenames when not provided (helps training runs).
        if dataset_ts is None:
            dataset_ts = parse_timestamp_from_filename(cohort_dataset_name) or parse_timestamp_from_filename(
                course_dataset_name
            )

        now = datetime.now(timezone.utc)
        event_norm = str(event or "").strip().lower()
        status = None
        started_at = None
        finished_at = None
        if event_norm == "started":
            status = "running"
            started_at = now
        elif event_norm == "completed":
            status = "completed"
            finished_at = now
        elif event_norm == "failed":
            status = "failed"
            finished_at = now

        # Dict insertion order is used below to enforce a stable column order on first table creation.
        row: dict[str, t.Any] = {
            "institution_id": resolved_institution_id,
            "run_id": str(run_id),
            "run_type": str(run_type),
            "status": status,
            "started_at": started_at,
            "finished_at": finished_at,
            "updated_at": now,
            "run_url": _best_effort_databricks_run_url(run_id=str(run_id)),
            "cohort_dataset_name": cohort_dataset_name,
            "course_dataset_name": course_dataset_name,
            "dataset_ts": dataset_ts,
            "term_filter": term_filter,
            "model_run_id": model_run_id,
            "experiment_id": experiment_id,
            "pipeline_version": pipeline_version,
            "error_message": error_message,
            "payload_json": _json_dumps(payload2),
        }

        df = spark.createDataFrame([row])
        try:
            # Reorder columns explicitly (Spark may reorder dict fields when inferring schema).
            if hasattr(df, "select"):
                df = df.select(*list(row.keys()))
        except Exception:
            pass

        # Prefer an idempotent upsert via Delta merge when available.
        try:
            from delta.tables import DeltaTable  # type: ignore
            from pyspark.sql import functions as F  # type: ignore

            # Allow merge schema evolution on Databricks if enabled.
            try:
                spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
            except Exception:
                pass

            try:
                dt = DeltaTable.forName(spark, table_path)
            except Exception:
                (
                    df.write.format("delta")
                    .mode("append")
                    .option("mergeSchema", "true")
                    .saveAsTable(table_path)
                )
                return True

            # finished_at: keep the latest terminal timestamp
            finished_at_expr = F.when(
                F.col("s.finished_at").isNull(),
                F.col("t.finished_at"),
            ).when(
                F.col("t.finished_at").isNull(),
                F.col("s.finished_at"),
            ).when(
                F.col("s.finished_at") > F.col("t.finished_at"),
                F.col("s.finished_at"),
            ).otherwise(F.col("t.finished_at"))

            # status precedence: failed > completed > running
            status_expr = F.expr(
                """
                CASE
                  WHEN t.status = 'failed' OR s.status = 'failed' THEN 'failed'
                  WHEN t.status = 'completed' OR s.status = 'completed' THEN 'completed'
                  WHEN t.status = 'running' OR s.status = 'running' THEN 'running'
                  ELSE COALESCE(s.status, t.status)
                END
                """
            )

            (
                dt.alias("t")
                .merge(
                    df.alias("s"),
                    "t.run_id = s.run_id AND t.run_type = s.run_type",
                )
                .whenMatchedUpdate(
                    set={
                        "updated_at": F.col("s.updated_at"),
                        "run_url": F.coalesce(F.col("s.run_url"), F.col("t.run_url")),
                        "institution_id": F.coalesce(
                            F.col("s.institution_id"), F.col("t.institution_id")
                        ),
                        "status": status_expr,
                        "started_at": F.coalesce(F.col("t.started_at"), F.col("s.started_at")),
                        "finished_at": finished_at_expr,
                        "cohort_dataset_name": F.coalesce(
                            F.col("s.cohort_dataset_name"), F.col("t.cohort_dataset_name")
                        ),
                        "course_dataset_name": F.coalesce(
                            F.col("s.course_dataset_name"), F.col("t.course_dataset_name")
                        ),
                        "dataset_ts": F.coalesce(F.col("s.dataset_ts"), F.col("t.dataset_ts")),
                        "term_filter": F.coalesce(F.col("s.term_filter"), F.col("t.term_filter")),
                        "model_run_id": F.coalesce(F.col("s.model_run_id"), F.col("t.model_run_id")),
                        "experiment_id": F.coalesce(F.col("s.experiment_id"), F.col("t.experiment_id")),
                        "pipeline_version": F.coalesce(F.col("s.pipeline_version"), F.col("t.pipeline_version")),
                        "error_message": F.coalesce(F.col("s.error_message"), F.col("t.error_message")),
                        "payload_json": F.coalesce(F.col("s.payload_json"), F.col("t.payload_json")),
                    }
                )
                .whenNotMatchedInsertAll()
                .execute()
            )
            return True
        except Exception:
            # Fallback: append-only (may create duplicates if Delta merge isn't available).
            (
                df.write.format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(table_path)
            )
            return True
    except Exception as e:
        LOGGER.warning(
            "pipeline_runs: failed to write run record (run_id=%s, run_type=%s, event=%s): %s",
            run_id,
            run_type,
            event,
            e,
        )
        return False
