"""
One-time (or idempotent) Delta table creation for the GenAI mapping pipeline state layer.

State tables live under ``{catalog}.genai_mapping`` (Unity Catalog + Delta).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from edvise.genai.mapping.state._sql import (
    HITL_REVIEWS,
    PIPELINE_PHASES,
    PIPELINE_RUNS,
    get_spark_session,
    qualified_schema,
    qualified_table,
)

LOGGER = logging.getLogger(__name__)


def _rename_legacy_pipeline_run_id_column(spark: Any, table_fqn: str) -> None:
    """Older deployments used ``pipeline_run_id``; normalize to ``onboard_run_id`` idempotently."""
    try:
        spark.sql(f"ALTER TABLE {table_fqn} RENAME COLUMN pipeline_run_id TO onboard_run_id")
    except Exception:  # noqa: BLE001 — ok if column already named onboard_run_id
        LOGGER.debug("Skipped pipeline_run_id -> onboard_run_id rename for %s", table_fqn)


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
    spark.sql(f"ALTER TABLE {pr} ADD COLUMN IF NOT EXISTS db_run_id STRING")
    spark.sql(f"ALTER TABLE {pr} ADD COLUMN IF NOT EXISTS execute_run_id STRING")
    spark.sql(f"ALTER TABLE {pr} ADD COLUMN IF NOT EXISTS input_file_paths STRING")
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
    for tbl in (pr, pp, hr):
        _rename_legacy_pipeline_run_id_column(spark, tbl)
    LOGGER.info("genai state tables ensured under %s", qualified_schema(c))


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m edvise.genai.mapping.state.table_setup <catalog>", file=sys.stderr)
        sys.exit(1)
    create_state_tables(sys.argv[1])


if __name__ == "__main__":
    _main()
