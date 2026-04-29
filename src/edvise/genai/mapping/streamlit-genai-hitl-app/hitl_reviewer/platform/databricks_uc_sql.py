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
    return (
        f"{_sql_ident(c)}.{_sql_ident('genai_mapping')}.{_sql_ident('pipeline_runs')}"
    )


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


def hitl_group_identity_where_sql(
    *,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    table_alias: str | None = None,
) -> tuple[str, str, str, str]:
    """
    Normalize identifiers and build a WHERE clause that tolerates stray whitespace in table columns.

    ``table_alias`` prefixes columns (e.g. ``h`` for ``FROM hitl_reviews h``).

    Returns ``(oid, ph, at, where_sql)`` where ``where_sql`` uses trim(cast(... AS STRING)).
    """
    oid = str(onboard_run_id).strip()
    ph = str(phase).strip()
    at = str(artifact_type).strip()
    if not oid or not ph or not at:
        raise ValueError(
            "onboard_run_id, phase, and artifact_type must be non-empty after stripping"
        )
    p = f"{table_alias.strip()}." if (table_alias or "").strip() else ""
    w = f"""
    trim(cast({p}onboard_run_id AS STRING)) = {sql_str(oid)}
      AND trim(cast({p}phase AS STRING)) = {sql_str(ph)}
      AND trim(cast({p}artifact_type AS STRING)) = {sql_str(at)}
    """.strip()
    return oid, ph, at, w


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
    oid, ph, at, where_match = hitl_group_identity_where_sql(
        onboard_run_id=onboard_run_id,
        phase=phase,
        artifact_type=artifact_type,
    )
    count_q = f"SELECT COUNT(*) AS c FROM {t} WHERE {where_match}"
    n_before = int(run_query(count_q)["c"].iloc[0])
    if n_before < 1:
        raise RuntimeError(
            "No ``hitl_reviews`` row matches this group in the sidebar catalog. "
            f"Confirm **Unity Catalog** is `{str(catalog).strip()}`, then check "
            f"``onboard_run_id={oid!r}``, ``phase={ph!r}``, ``artifact_type={at!r}`` "
            "against the table (filters / SQL). Whitespace or catalog mismatch often causes this."
        )
    rev_sql = "NULL" if not (reviewer or "").strip() else sql_str(reviewer.strip())
    q = f"""
    UPDATE {t}
    SET
      status = {sql_str(decision)},
      reviewer = {rev_sql},
      reviewed_at = current_timestamp()
    WHERE {where_match}
    """
    execute_statement(q)
