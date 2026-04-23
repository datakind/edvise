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


def create_state_tables(catalog: str) -> None:
    """
    Create the ``genai_mapping`` schema and all state tables if they do not exist.

    Tables:

    * ``pipeline_runs`` — top-level run tracking
    * ``pipeline_phases`` — phase transition audit log
    * ``hitl_reviews`` — HITL artifact path + review status
    """
    if not (catalog and str(catalog).strip()):
        raise ValueError("catalog must be non-empty")

    spark: Any = get_spark_session()
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
          pipeline_run_id STRING,
          `catalog` STRING,
          status STRING,
          db_run_id STRING,
          created_at TIMESTAMP,
          updated_at TIMESTAMP
        ) USING DELTA
        """
    )
    # Existing deployments created before ``db_run_id`` — add column idempotently.
    spark.sql(f"ALTER TABLE {pr} ADD COLUMN IF NOT EXISTS db_run_id STRING")
    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {pp} (
          pipeline_run_id STRING,
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
          pipeline_run_id STRING,
          phase STRING,
          artifact_type STRING,
          artifact_path STRING,
          status STRING,
          reviewer STRING,
          reviewed_at TIMESTAMP
        ) USING DELTA
        """
    )
    LOGGER.info("genai state tables ensured under %s", qualified_schema(c))


def _main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m edvise.genai.mapping.state.table_setup <catalog>", file=sys.stderr)
        sys.exit(1)
    create_state_tables(sys.argv[1])


if __name__ == "__main__":
    _main()
