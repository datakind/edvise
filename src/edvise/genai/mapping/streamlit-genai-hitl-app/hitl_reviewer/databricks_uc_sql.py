"""Databricks SQL warehouse helpers for GenAI HITL ``hitl_reviews`` and related queries."""

from __future__ import annotations

import os

import pandas as pd
from databricks import sql as databricks_sql  # type: ignore[attr-defined]
from databricks.sdk.core import Config


def sql_str(value: str) -> str:
    """Single-quote SQL literal for UC identifier-style values."""
    return "'" + str(value).replace("'", "''") + "'"


def _sql_ident(part: str) -> str:
    if not str(part).strip():
        raise ValueError("SQL identifier must be non-empty")
    return "`" + str(part).replace("`", "``") + "`"


def hitl_reviews_fqn(catalog: str) -> str:
    c = str(catalog).strip()
    return f"{_sql_ident(c)}.{_sql_ident('genai_mapping')}.{_sql_ident('hitl_reviews')}"


def pipeline_runs_fqn(catalog: str) -> str:
    c = str(catalog).strip()
    return f"{_sql_ident(c)}.{_sql_ident('genai_mapping')}.{_sql_ident('pipeline_runs')}"


def get_warehouse_id() -> str:
    warehouse_id = (os.getenv("DATABRICKS_WAREHOUSE_ID") or "").strip()
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID must be set (SQL warehouse used to query UC)."
        )
    return warehouse_id


def _connection():
    cfg = Config()
    return databricks_sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{get_warehouse_id()}",
        credentials_provider=lambda: cfg.authenticate,
    )


def run_query(query: str) -> pd.DataFrame:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


def execute_statement(sql: str) -> None:
    with _connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql)


def approve_or_reject(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    reviewer: str,
    decision: str,
) -> None:
    if decision not in ("approved", "rejected"):
        raise ValueError("decision must be approved or rejected")
    t = hitl_reviews_fqn(catalog)
    rev_sql = "NULL" if not (reviewer or "").strip() else sql_str(reviewer.strip())
    q = f"""
    UPDATE {t}
    SET
      status = {sql_str(decision)},
      reviewer = {rev_sql},
      reviewed_at = current_timestamp()
    WHERE onboard_run_id = {sql_str(onboard_run_id)}
      AND phase = {sql_str(phase)}
      AND artifact_type = {sql_str(artifact_type)}
    """
    execute_statement(q)
