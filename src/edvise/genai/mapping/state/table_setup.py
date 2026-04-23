"""
One-time (or idempotent) Delta table creation for the GenAI mapping pipeline state layer.

State tables live under ``{catalog}.genai_mapping`` (Unity Catalog + Delta).
"""

from __future__ import annotations

import logging
import sys
from typing import Any
from uuid import uuid4

from edvise.genai.mapping.state._sql import (
    HITL_REVIEWS,
    PIPELINE_PHASES,
    PIPELINE_RUNS,
    get_spark_session,
    qualified_schema,
    qualified_table,
)

LOGGER = logging.getLogger(__name__)


def _describe_table_col_names(spark: Any, table_fqn: str) -> list[str]:
    """Return base column names from ``DESCRIBE TABLE`` (stops at partition / metadata section)."""
    out: list[str] = []
    for row in spark.sql(f"DESCRIBE TABLE {table_fqn}").collect():
        name: str
        try:
            name = str(row["col_name"]).strip()
        except Exception:
            try:
                name = str(row.col_name).strip()  # type: ignore[attr-defined]
            except Exception:
                name = str(row[0]).strip()
        if not name:
            continue
        if name.startswith("#"):
            break
        out.append(name)
    return out


def _rebuild_table_pipeline_run_id_to_onboard(
    spark: Any,
    catalog: str,
    table_short: str,
    fqn: str,
) -> None:
    """
    Replace a table that still has ``pipeline_run_id`` when ``RENAME COLUMN`` is unsupported
    (e.g. legacy Parquet) with a Delta table using ``onboard_run_id``, preserving rows.
    """
    mig = f"{table_short}__edvise_mig_{uuid4().hex[:12]}"
    tmp = qualified_table(catalog, mig)
    spark.sql(f"DROP TABLE IF EXISTS {tmp}")
    if table_short == PIPELINE_RUNS:
        ctas = f"""
        CREATE TABLE {tmp} USING DELTA AS
        SELECT
          institution_id,
          pipeline_run_id AS onboard_run_id,
          `catalog`,
          status,
          db_run_id,
          execute_run_id,
          input_file_paths,
          created_at,
          updated_at
        FROM {fqn}
        """
    elif table_short == PIPELINE_PHASES:
        ctas = f"""
        CREATE TABLE {tmp} USING DELTA AS
        SELECT
          pipeline_run_id AS onboard_run_id,
          phase,
          status,
          started_at,
          completed_at
        FROM {fqn}
        """
    elif table_short == HITL_REVIEWS:
        ctas = f"""
        CREATE TABLE {tmp} USING DELTA AS
        SELECT
          pipeline_run_id AS onboard_run_id,
          phase,
          artifact_type,
          artifact_path,
          status,
          reviewer,
          reviewed_at
        FROM {fqn}
        """
    else:
        raise ValueError(f"unknown state table {table_short!r}")
    spark.sql(ctas)
    spark.sql(f"DROP TABLE {fqn}")
    spark.sql(f"ALTER TABLE {tmp} RENAME TO {fqn}")


def _ensure_onboard_run_id_column(spark: Any, catalog: str, table_short: str) -> None:
    """Older deployments used ``pipeline_run_id``; normalize to ``onboard_run_id`` (rename or rebuild)."""
    fqn = qualified_table(catalog, table_short)
    cols = set(_describe_table_col_names(spark, fqn))
    if "onboard_run_id" in cols:
        return
    # Empty ``cols`` (e.g. test doubles): still attempt rename so behavior matches unknown schemas.
    if cols and "pipeline_run_id" not in cols:
        return
    try:
        spark.sql(f"ALTER TABLE {fqn} RENAME COLUMN pipeline_run_id TO onboard_run_id")
        LOGGER.info("Renamed pipeline_run_id -> onboard_run_id on %s", fqn)
    except Exception as e:  # noqa: BLE001 — Parquet / older catalogs often reject RENAME COLUMN
        LOGGER.warning(
            "ALTER RENAME COLUMN failed for %s (%s); rebuilding as Delta with onboard_run_id",
            fqn,
            e,
        )
        _rebuild_table_pipeline_run_id_to_onboard(spark, catalog, table_short, fqn)


def _add_column_if_missing(spark: Any, table_fqn: str, col_name: str, col_type: str) -> None:
    """Add a column when missing; older runtimes reject ``ADD COLUMN IF NOT EXISTS`` syntax."""
    try:
        spark.sql(f"ALTER TABLE {table_fqn} ADD COLUMN {col_name} {col_type}")
    except Exception:  # noqa: BLE001 — ok if column already exists (e.g. new CREATE TABLE)
        LOGGER.debug("Skipped ADD COLUMN %s on %s (likely already present)", col_name, table_fqn)


def create_state_tables(catalog: str, spark: Any | None = None) -> None:
    """
    Create the ``genai_mapping`` schema and all state tables if they do not exist.

    Tables:

    * ``pipeline_runs`` — top-level run tracking
    * ``pipeline_phases`` — phase transition audit log
    * ``hitl_reviews`` — HITL artifact path + review status

    Pass ``spark`` when the caller already holds the active session (e.g. tests that
    monkeypatch :func:`edvise.genai.mapping.state._sql.get_spark_session`).
    """
    if not (catalog and str(catalog).strip()):
        raise ValueError("catalog must be non-empty")

    spark = spark if spark is not None else get_spark_session()
    if spark is None:
        raise RuntimeError("No active Spark session; run on Databricks with Spark available")

    c = str(catalog).strip()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {qualified_schema(c)}")

    pr = qualified_table(c, PIPELINE_RUNS)
    pp = qualified_table(c, PIPELINE_PHASES)
    hr = qualified_table(c, HITL_REVIEWS)

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {pr} (
          institution_id STRING,
          onboard_run_id STRING,
          `catalog` STRING,
          status STRING,
          db_run_id STRING,
          execute_run_id STRING,
          input_file_paths STRING,
          created_at TIMESTAMP,
          updated_at TIMESTAMP
        ) USING DELTA
        """
    )
    # Existing deployments created before ``db_run_id`` / ``execute_run_id`` — add idempotently.
    for _col, _typ in (
        ("db_run_id", "STRING"),
        ("execute_run_id", "STRING"),
        ("input_file_paths", "STRING"),
    ):
        _add_column_if_missing(spark, pr, _col, _typ)
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {pp} (
          onboard_run_id STRING,
          phase STRING,
          status STRING,
          started_at TIMESTAMP,
          completed_at TIMESTAMP
        ) USING DELTA
        """
    )
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {hr} (
          onboard_run_id STRING,
          phase STRING,
          artifact_type STRING,
          artifact_path STRING,
          status STRING,
          reviewer STRING,
          reviewed_at TIMESTAMP
        ) USING DELTA
        """
    )
    for short in (PIPELINE_RUNS, PIPELINE_PHASES, HITL_REVIEWS):
        _ensure_onboard_run_id_column(spark, c, short)
    LOGGER.info("genai state tables ensured under %s", qualified_schema(c))


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m edvise.genai.mapping.state.table_setup <catalog>", file=sys.stderr)
        sys.exit(1)
    create_state_tables(sys.argv[1])


if __name__ == "__main__":
    _main()
