"""
GenAI mapping — Unity Catalog ``hitl_reviews`` reviewer UI.

HITL **choice** values are edited in JSON on the **silver** volume under
``{silver_genai_mapping_root}/runs/...``, matching
``edvise_genai_ia.resolve_run_paths`` (``identity_agent`` grain/term HITL files) and
``edvise_genai_sma.resolve_run_paths`` (``schema_mapping_agent`` cohort/course HITL manifests).
The file path is the one stored in ``hitl_reviews.artifact_path`` (and overridable in the UI).
Unity Catalog *approval* is still ``{catalog}.genai_mapping.hitl_reviews``.

**Local run** (from this directory, with Databricks auth configured, e.g. ``databricks auth login``)::

    export DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
    export GENAI_HITL_CATALOG=dev_sst_02   # optional; default in sidebar
    streamlit run app.py

**Databricks Apps (dev):** run ``databricks bundle deploy`` / ``databricks bundle run`` from
this directory (see ``databricks.yml``) or trigger CI
``.github/workflows/deploy-genai-hitl-app.yml`` (Actions → “Deploy GenAI HITL app (dev)”).
Use the same ``--var sql_warehouse_id=…`` and ``--var datakind_group_to_manage_workflow=…``
as the metadata dashboard app for dev.

The Databricks app identity (service principal) needs **read/write** on the silver
``/Volumes/.../..._silver/.../genai_mapping/...`` HITL paths, in addition to SQL access.
"""

from __future__ import annotations

import json
import os
import re
import pandas as pd
import streamlit as st
from databricks import sql as databricks_sql  # type: ignore[attr-defined]
from databricks.sdk.core import Config

from hitl_silver_paths import (
    artifact_path_contains_onboard_run_id,
    set_item_choice,
)
from uc_files import read_unity_file_text, write_unity_file_text


def _sql_str(value: str) -> str:
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


def load_hitl_rows(
    catalog: str,
    *,
    onboard_run_id: str | None,
    phase: str | None,
    status: str | None,
    limit: int,
) -> pd.DataFrame:
    t_h = hitl_reviews_fqn(catalog)
    t_p = pipeline_runs_fqn(catalog)
    where: list[str] = []
    c_sql = _sql_str(str(catalog).strip())
    if (onboard_run_id or "").strip():
        where.append(f"h.onboard_run_id = {_sql_str(onboard_run_id.strip())}")
    if (phase or "").strip():
        where.append(f"h.phase = {_sql_str(phase.strip())}")
    if (status or "").strip():
        where.append(f"h.status = {_sql_str(status.strip())}")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    lim = max(1, min(int(limit), 5000))
    q = f"""
    SELECT
      h.onboard_run_id,
      h.phase,
      h.artifact_type,
      h.artifact_path,
      h.status,
      h.reviewer,
      h.reviewed_at,
      p.institution_id
    FROM {t_h} h
    LEFT JOIN {t_p} p
      ON h.onboard_run_id = p.onboard_run_id
     AND p.`catalog` = {c_sql}
    {w}
    ORDER BY h.reviewed_at DESC NULLS FIRST, h.onboard_run_id, h.phase, h.artifact_type, h.artifact_path
    LIMIT {lim}
    """
    return run_query(q)


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s))[:80]


def _hitl_option_label(options: list, j: int) -> str:
    o = options[j]
    if isinstance(o, dict):
        lab = o.get("label")
        if lab is not None:
            return f"{j + 1}. {lab}"
    return f"{j + 1}. {o!r}"


def render_silver_hitl_editor(
    *,
    default_artifact_path: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
) -> None:
    st.caption(
        f"**artifact_type** in UC: ``{artifact_type}`` — **onboard_run_id** (this expander): "
        f"``{onboard_run_id}`` (in standard onboard layout it appears in the path under "
        f"``…/genai_mapping/runs/onboard/{{onboard_run_id}}/``)."
    )
    st.markdown(
        "**HITL JSON (silver volume)** — paths match ``edvise_genai_ia`` / ``edvise_genai_sma`` "
        "``resolve_run_paths``: ``{silver}/genai_mapping/runs/…/identity_agent/`` or "
        "``…/schema_mapping_agent/`` with the HITL filenames. "
        "The ``onboard_run_id`` is the run folder in ``runs/onboard/…/``. "
        "Default file path = ``hitl_reviews.artifact_path`` (full string already includes that segment)."
    )
    sk = f"{_safe_key(onboard_run_id)}-{_safe_key(phase)}-{_safe_key(artifact_type)}"
    pkey = f"path-{sk}"
    path_in = st.text_input(
        "UC file path to read/write (absolute ``/Volumes/{catalog}/…_silver/…``)",
        value=default_artifact_path,
        key=pkey,
    )
    silver_path = (path_in or "").strip()
    if not silver_path:
        st.caption("Set the file path to load HITL JSON (typically the registered `artifact_path`).")
        return
    if not silver_path.startswith("/Volumes/"):
        st.warning("Path should be an absolute Unity Catalog volume path starting with ``/Volumes/``.")
    if not artifact_path_contains_onboard_run_id(silver_path, str(onboard_run_id)):
        st.warning(
            "This path string does not contain the **onboard_run_id** for this review row. "
            "Onboard HITL files usually include it under "
            f"``…/genai_mapping/runs/onboard/{onboard_run_id}/…``. "
            "Confirm the path, or keep going only if you intend another file (e.g. an execute run path). "
        )
    st.code(silver_path, language="text")
    try:
        raw = read_unity_file_text(silver_path)
    except Exception as e:  # noqa: BLE001 — show in UI
        st.error(f"Could not read file: {e}")
        return
    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return
    items = data.get("items")
    if not isinstance(items, list) or not items:
        st.info("No `items` in this HITL JSON, or the list is empty — nothing to select.")
        return

    choice_updates: list[tuple[int, int]] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
        options = item.get("options")
        if not options or not isinstance(options, list):
            st.caption(f"**{q}** — no `options` (direct manifest edit may be required).")
            continue
        n = len(options)
        if n < 1:
            continue
        c_raw = item.get("choice")
        if c_raw is None:
            default_ix = 0
        else:
            try:
                c_int = int(c_raw)
            except (TypeError, ValueError):
                c_int = 1
            default_ix = max(0, min(n - 1, c_int - 1))

        sel = st.radio(
            q,
            list(range(n)),
            index=default_ix,
            format_func=lambda j, opts=options: _hitl_option_label(opts, j),
            key=f"sv{sk}item{i}{item.get('item_id', i)}",
        )
        choice_updates.append((i, int(sel) + 1))

    if st.button(
        "Save JSON to silver volume (writes `choice` for items with options above)",
        key=f"ssave{sk}",
        type="secondary",
    ):
        try:
            fresh = json.loads(read_unity_file_text(silver_path))
        except Exception as e:  # noqa: BLE001
            st.error(f"Re-read failed: {e}")
            return
        for i, c in choice_updates:
            set_item_choice(fresh, i, c)
        try:
            out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
            write_unity_file_text(silver_path, out, overwrite=True)
        except Exception as e:  # noqa: BLE001
            st.error(f"Write failed: {e}")
        else:
            st.success("Saved to silver volume.")


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
    "Approve or reject in ``hitl_reviews`` by "
    "``(onboard_run_id, phase, artifact_type)``. Set ``choice`` in HITL JSON on the **silver** "
    "volume (same path the pipeline registered) when your process requires it."
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

# Show ``artifact_path`` (silver) and optional ``institution_id`` from ``pipeline_runs``
_display_cols = [
    c
    for c in (
        "institution_id",
        "onboard_run_id",
        "phase",
        "artifact_type",
        "artifact_path",
        "status",
        "reviewer",
        "reviewed_at",
    )
    if c in df.columns
]
st.dataframe(
    df[_display_cols] if _display_cols else df,
    use_container_width=True,
    hide_index=True,
)

if df.empty:
    st.info("No rows match. Clear filters or raise the row limit.")
    st.stop()

st.subheader("Actions")
st.caption(
    "1) For pending groups, set ``choice`` in the HITL JSON on the **silver** path below "
    "(``artifact_path`` from the run). 2) Approve or Reject in Unity Catalog when ready."
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
        st.text(
            "Registered ``artifact_path``(s) on silver — includes ``onboard_run_id`` in "
            f"``…/runs/onboard/{onboard_run_id}/…`` for standard onboard: " + paths_preview
        )
        raw_s = sub["artifact_path"].dropna().astype(str).str.strip()
        raw_paths = [p for p in raw_s.tolist() if p]
        if not raw_paths:
            st.warning(
                "No ``artifact_path`` on these rows; the pipeline has nothing to point the editor at. "
                "Re-run registration from ``edvise_genai_ia`` / ``edvise_genai_sma`` onboard."
            )
        else:
            default_path = raw_paths[0]
            if len(set(raw_paths)) > 1:
                st.caption("Several paths are registered; defaulting the editor to the first. Adjust the path field if needed.")
            render_silver_hitl_editor(
                default_artifact_path=default_path,
                onboard_run_id=str(onboard_run_id),
                phase=str(phase),
                artifact_type=str(artifact_type),
            )
        st.divider()
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
            st.caption("Updates ``hitl_reviews`` only; JSON edits target the silver ``artifact_path`` above.")
