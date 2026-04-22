"""Internal helpers: Spark session and safe SQL identifier / string literals for UC."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

GENAI_MAPPING_SCHEMA: str = "genai_mapping"
PIPELINE_RUNS: str = "pipeline_runs"
PIPELINE_PHASES: str = "pipeline_phases"
HITL_REVIEWS: str = "hitl_reviews"


def get_spark_session() -> Any:
    try:
        from pyspark.sql import SparkSession  # type: ignore

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception as e:
        LOGGER.warning("genai state: unable to create SparkSession (%s)", e)
        return None


def _ident(s: str) -> str:
    if not s or not s.strip():
        raise ValueError("Identifier must be non-empty")
    return f"`{s.replace('`', '``')}`"


def lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def qualified_schema(catalog: str) -> str:
    """Return ``{catalog}`.`genai_mapping`` (schema) for use in SQL."""
    c = str(catalog).strip()
    return f"{_ident(c)}.{_ident(GENAI_MAPPING_SCHEMA)}"


def qualified_table(catalog: str, table: str) -> str:
    """Return ``{catalog}`.`genai_mapping`.`table`` for use in SQL."""
    t = str(table).strip()
    c = str(catalog).strip()
    return f"{_ident(c)}.{_ident(GENAI_MAPPING_SCHEMA)}.{_ident(t)}"
