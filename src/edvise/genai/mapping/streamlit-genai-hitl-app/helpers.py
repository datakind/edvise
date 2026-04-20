from __future__ import annotations

import os

from databricks import sql as databricks_sql  # type: ignore[attr-defined]
from databricks.sdk.core import Config
import pandas as pd


DEFAULT_DB_WORKSPACE = "dev_sst_02"


def get_databricks_warehouse_id() -> str:
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    if not warehouse_id:
        raise RuntimeError(
            "DATABRICKS_WAREHOUSE_ID must be set in the app configuration."
        )
    return warehouse_id


def run_sql_query(query: str) -> pd.DataFrame:
    """Run a SQL statement against the configured SQL warehouse and return a DataFrame."""
    cfg = Config()

    with databricks_sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{get_databricks_warehouse_id()}",
        credentials_provider=lambda: cfg.authenticate,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()


def quote_uc_three_part_name(fqn: str) -> str:
    """Quote ``catalog.schema.table`` for use in SQL."""
    parts = [p.strip() for p in fqn.split(".") if p.strip()]
    if len(parts) != 3:
        raise ValueError(
            "Expected a three-part Unity Catalog name (catalog.schema.table), "
            f"got {fqn!r}"
        )
    return ".".join(f"`{p}`" for p in parts)


def get_genai_pipeline_artifacts_table_fqn() -> str:
    """
    Fully qualified name for the GenAI pipeline artifact registry Delta table.

    Override with ``GENAI_ARTIFACTS_UC_SCHEMA`` / ``GENAI_ARTIFACTS_UC_TABLE`` if the table
    lives outside ``{DB_workspace}.genai.genai_pipeline_artifacts``.
    """
    catalog = os.getenv("DB_workspace", DEFAULT_DB_WORKSPACE).strip() or DEFAULT_DB_WORKSPACE
    schema = os.getenv("GENAI_ARTIFACTS_UC_SCHEMA", "genai").strip() or "genai"
    table = os.getenv("GENAI_ARTIFACTS_UC_TABLE", "genai_pipeline_artifacts").strip()
    if not table:
        table = "genai_pipeline_artifacts"
    return f"{catalog}.{schema}.{table}"


def build_genai_pipeline_artifacts_list_query(table_fqn: str) -> str:
    """SELECT all registered GenAI JSON artifact rows (most recent first)."""
    t = quote_uc_three_part_name(table_fqn)
    return f"""
    SELECT *
    FROM {t}
    ORDER BY registered_at DESC NULLS LAST, pipeline_run_id DESC, artifact_kind ASC
    """


def filter_artifacts_dataframe(
    df: pd.DataFrame,
    *,
    institution_id: str | None,
    artifact_kinds: list[str] | None,
) -> pd.DataFrame:
    """Apply sidebar filters in memory (full scan was already loaded from UC)."""
    out = df
    if institution_id:
        out = out[out["institution_id"].astype(str) == institution_id]
    if artifact_kinds:
        out = out[out["artifact_kind"].astype(str).isin(artifact_kinds)]
    return out
