"""
GenAI mapping — Unity Catalog ``hitl_reviews`` reviewer UI.

**Local run** (from this directory, with Databricks auth configured, e.g. ``databricks auth login``)::

    export DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
    export GENAI_HITL_CATALOG=dev_sst_02   # optional; default in sidebar
    streamlit run app.py

**Databricks Apps (dev):** run ``databricks bundle deploy`` / ``databricks bundle run`` from
this directory (see ``databricks.yml``) or trigger CI
``.github/workflows/deploy-genai-hitl-app.yml`` (Actions → “Deploy GenAI HITL app (dev)”).
Use the same ``--var sql_warehouse_id=…`` and ``--var datakind_group_to_manage_workflow=…``
as the metadata dashboard app for dev.
"""

from __future__ import annotations

import os
import re
import pandas as pd
import streamlit as st
from databricks import sql as databricks_sql  # type: ignore[attr-defined]
from databricks.sdk.core import Config


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sql_ident(part: str) -> str:
    if not str(part).strip():
        raise ValueError("SQL identifier must be non-empty")
    return "`" + str(part).replace("`", "``") + "`"


def hitl_reviews_fqn(catalog: str) -> str:
    c = str(catalog).strip()
    return f"{_sql_ident(c)}.{_sql_ident('genai_mapping')}.{_sql_ident('hitl_reviews')}"


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


def load_hitl_rows(
    catalog: str,
    *,
    onboard_run_id: str | None,
    phase: str | None,
    status: str | None,
    limit: int,
) -> pd.DataFrame:
    t = hitl_reviews_fqn(catalog)
    where: list[str] = []
    if (onboard_run_id or "").strip():
        where.append(f"onboard_run_id = {_sql_str(onboard_run_id.strip())}")
    if (phase or "").strip():
        where.append(f"phase = {_sql_str(phase.strip())}")
    if (status or "").strip():
        where.append(f"status = {_sql_str(status.strip())}")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    lim = max(1, min(int(limit), 5000))
    q = f"""
    SELECT
      onboard_run_id, phase, artifact_type, artifact_path, status,
      reviewer, reviewed_at
    FROM {t}
    {w}
    ORDER BY reviewed_at DESC NULLS FIRST, onboard_run_id, phase, artifact_type, artifact_path
    LIMIT {lim}
    """
    return run_query(q)


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
    rev_sql = "NULL" if not (reviewer or "").strip() else _sql_str(reviewer.strip())
    q = f"""
    UPDATE {t}
    SET
      status = {_sql_str(decision)},
      reviewer = {rev_sql},
      reviewed_at = current_timestamp()
    WHERE onboard_run_id = {_sql_str(onboard_run_id)}
      AND phase = {_sql_str(phase)}
      AND artifact_type = {_sql_str(artifact_type)}
    """
    execute_statement(q)


def _default_catalog() -> str:
    for key in ("GENAI_HITL_CATALOG", "DB_workspace"):
        v = (os.getenv(key) or "").strip()
        if v:
            return v
    return "dev_sst_02"


_CATALOG_SAFE = re.compile(r"^[a-zA-Z0-9_]+$")


def _validate_catalog(catalog: str) -> str:
    c = str(catalog).strip()
    if not c or not _CATALOG_SAFE.match(c):
        raise ValueError("Catalog must be a simple identifier (letters, digits, underscore).")
    return c


st.set_page_config(page_title="GenAI HITL reviews", layout="wide")
st.title("GenAI mapping — UC HITL reviews")
st.caption(
    "Approve or reject by ``(onboard_run_id, phase, artifact_type)`` — same scope as "
    "``pipeline_state.resolve_hitl`` (all paths for that type are updated together)."
)

if "reviewer" not in st.session_state:
    st.session_state["reviewer"] = os.getenv("GENAI_HITL_REVIEWER", "").strip() or (
        os.getenv("USER", "") or os.getenv("USERNAME", "") or "reviewer"
    ).strip()

with st.sidebar:
    st.subheader("Connection")
    try:
        get_warehouse_id()
        st.success("DATABRICKS_WAREHOUSE_ID is set")
    except RuntimeError as e:
        st.error(str(e))
    catalog_in = st.text_input("Unity Catalog", value=_default_catalog())
    try:
        catalog = _validate_catalog(catalog_in)
    except ValueError as e:
        st.warning(str(e))
        catalog = _default_catalog()
    st.session_state["reviewer"] = st.text_input(
        "Reviewer name (stored on approve/reject)",
        value=st.session_state["reviewer"],
    )
    limit = st.number_input("Row limit", min_value=50, max_value=5000, value=500, step=50)
    st.divider()
    st.subheader("Filters")
    f_run = st.text_input("onboard_run_id contains", value="")
    f_phase = st.text_input("phase equals", value="")
    f_status = st.selectbox(
        "status",
        options=["(any)", "pending", "approved", "rejected"],
        index=0,
    )
    st.button("Refresh data", type="primary", help="Re-runs the query with current filters.")

try:
    get_warehouse_id()
except RuntimeError:
    st.stop()

# Client-side filter for "contains" on run id: load broader slice when using contains
use_contains = bool((f_run or "").strip())
onboard_exact = None if use_contains else ((f_run or "").strip() or None)
phase_f = (f_phase or "").strip() or None
status_f = None if f_status == "(any)" else f_status

try:
    df = load_hitl_rows(
        catalog,
        onboard_run_id=onboard_exact,
        phase=phase_f,
        status=status_f,
        limit=limit if not use_contains else min(limit, 5000),
    )
    if use_contains and (f_run or "").strip():
        needle = (f_run or "").strip().lower()
        df = df[df["onboard_run_id"].astype(str).str.lower().str.contains(needle, na=False)]
except Exception as e:  # noqa: BLE001 — show in UI
    st.exception(e)
    st.stop()

st.dataframe(df, use_container_width=True, hide_index=True)

if df.empty:
    st.info("No rows match. Clear filters or raise the row limit.")
    st.stop()

st.subheader("Actions")
st.caption(
    "Pick a pending group and click Approve or Reject. "
    "Ensure JSON HITL files on the volume are reviewed (``choice``) before approving if your process requires it."
)

pending = df[df["status"].astype(str).str.lower() == "pending"].copy()
if pending.empty:
    st.success("No pending rows in the current result set.")
    st.stop()

groups = (
    pending[["onboard_run_id", "phase", "artifact_type"]]
    .drop_duplicates()
    .sort_values(["onboard_run_id", "phase", "artifact_type"])
    .itertuples(index=False)
)

for onboard_run_id, phase, artifact_type in groups:
    sub = pending[
        (pending["onboard_run_id"] == onboard_run_id)
        & (pending["phase"] == phase)
        & (pending["artifact_type"] == artifact_type)
    ]
    paths_preview = "; ".join(sub["artifact_path"].astype(str).head(3).tolist())
    if len(sub) > 3:
        paths_preview += f" … (+{len(sub) - 3} more)"
    with st.expander(f"{onboard_run_id} | {phase} | {artifact_type} ({len(sub)} path(s))"):
        st.text(paths_preview)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(
                "Approve",
                key=f"a-{onboard_run_id}-{phase}-{artifact_type}",
                type="primary",
            ):
                try:
                    approve_or_reject(
                        catalog,
                        str(onboard_run_id),
                        str(phase),
                        str(artifact_type),
                        st.session_state["reviewer"],
                        "approved",
                    )
                    st.toast("Approved.", icon="✅")
                    st.rerun()
                except Exception as ex:  # noqa: BLE001
                    st.error(str(ex))
        with c2:
            if st.button("Reject", key=f"r-{onboard_run_id}-{phase}-{artifact_type}"):
                try:
                    approve_or_reject(
                        catalog,
                        str(onboard_run_id),
                        str(phase),
                        str(artifact_type),
                        st.session_state["reviewer"],
                        "rejected",
                    )
                    st.toast("Rejected.", icon="⛔")
                    st.rerun()
                except Exception as ex:  # noqa: BLE001
                    st.error(str(ex))
        with c3:
            st.caption("Updates all rows for this type (UC merge semantics).")
