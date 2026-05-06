"""
Maps & outputs: browse silver artifacts for onboard runs with **no pending** HITL rows, plus ``active/``.

Uses ``hitl_reviews`` aggregates and the same ``…/genai_mapping/`` layout as the pipeline jobs.
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.platform.databricks_uc_sql import (
    get_warehouse_id,
    load_onboard_runs_hitl_complete,
)
from hitl_reviewer.platform.volume_path_utils import institution_id_from_silver_volume_path
from hitl_reviewer.ui.hitl_streamlit import (
    init_reviewer_in_session,
    init_sidebar_form_state,
    render_connection_sidebar,
    validate_catalog,
)
from hitl_reviewer.ui.run_artifacts_browser import (
    known_active_artifact_paths,
    known_onboard_run_artifact_paths,
    render_artifact_sections,
)

st.set_page_config(page_title="Maps & outputs", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()

st.title("Maps & outputs")
st.caption(
    "Pick an onboard run where **all** ``hitl_reviews`` rows are finalized (no ``pending``). "
    "Then open expanders to preview maps and outputs on silver, and compare with **active/** "
    "(promoted execute artifacts for the same institution)."
)

catalog, _sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=False,
    page_heading="Maps & outputs",
    page_caption=(
        "Unity Catalog for ``hitl_reviews`` / ``pipeline_runs``. "
        "Use **HITL Review History** in the sidebar to edit pending groups."
    ),
)
if not warehouse_ok:
    st.stop()

try:
    catalog = validate_catalog(catalog)
except ValueError as e:
    st.error(str(e))
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    lim = st.number_input(
        "Max runs to load",
        min_value=20,
        max_value=2000,
        value=200,
        step=20,
        help="SQL LIMIT on distinct onboard_run_id groups.",
    )
with c2:
    onboard_only = st.checkbox(
        "Only paths under …/runs/onboard/…",
        value=True,
        help="Filters out groups whose sample HITL path looks like an execute run or other layout.",
    )
with c3:
    inst_needle = st.text_input(
        "institution_id contains (optional)",
        value="",
        help="Case-insensitive substring filter on resolved institution id.",
    )

if st.button("Load HITL-complete runs", type="primary"):
    st.session_state["maps_hitl_complete_df"] = None
    st.session_state["maps_hitl_complete_err"] = None
    try:
        df = load_onboard_runs_hitl_complete(catalog, limit=int(lim))
        if onboard_only and not df.empty and "sample_artifact_path" in df.columns:
            sub = (
                df["sample_artifact_path"]
                .astype(str)
                .str.contains("/runs/onboard/", case=False, na=False)
            )
            df = df.loc[sub].copy()
        st.session_state["maps_hitl_complete_df"] = df
    except Exception as e:  # noqa: BLE001
        st.session_state["maps_hitl_complete_err"] = str(e)

err = st.session_state.get("maps_hitl_complete_err")
if err:
    st.error(err)

df = st.session_state.get("maps_hitl_complete_df")
if df is None:
    st.info('Set filters and click **Load HITL-complete runs**.')
    st.stop()

if df.empty:
    st.warning("No runs match the current filters.")
    st.stop()

view = df
if (inst_needle or "").strip():
    needle = inst_needle.strip().lower()

    def _row_inst(r) -> str:
        explicit = str(r.get("institution_id") or "").strip()
        if explicit:
            return explicit
        p = str(r.get("sample_artifact_path") or "").strip()
        return institution_id_from_silver_volume_path(p) or ""

    inst_col = df.apply(_row_inst, axis=1)
    view = df.loc[inst_col.astype(str).str.lower().str.contains(needle, na=False)].copy()
    if view.empty:
        st.warning("No rows match the institution filter.")
        st.stop()

st.subheader("Runs with HITL complete (no pending rows)")
st.dataframe(
    view,
    use_container_width=True,
    hide_index=True,
    height=min(360, 60 + len(view) * 36),
)

opts = view["onboard_run_id"].astype(str).tolist()
pick = st.selectbox("Choose onboard_run_id", options=opts, index=0)
row = view.loc[view["onboard_run_id"].astype(str) == pick].iloc[0]

resolved_inst = str(row.get("institution_id") or "").strip()
if not resolved_inst:
    resolved_inst = institution_id_from_silver_volume_path(
        str(row.get("sample_artifact_path") or "")
    ) or ""

inst_final = st.text_input(
    "Institution id (edit if missing or wrong)",
    value=resolved_inst,
    help="Drives ``/Volumes/…/<institution_id>_silver/…/genai_mapping/`` paths.",
)
inst_final = (inst_final or "").strip()
if not inst_final:
    st.warning("Institution id is required to build volume paths.")
    st.stop()

genai_root, onboard_paths = known_onboard_run_artifact_paths(
    inst_final, catalog, str(pick).strip()
)
active_paths = known_active_artifact_paths(inst_final, catalog)

st.divider()
st.markdown(f"**Silver genai_mapping root:** `{genai_root}`")
c_left, c_right = st.columns(2)
with c_left:
    render_artifact_sections(
        title=f"Onboard run `{pick}`",
        paths=onboard_paths,
    )
with c_right:
    st.caption(
        "Promoted hook packages may also live under ``active/identity_hooks/…`` — browse that folder "
        "in Unity Catalog when your onboard run materialized IA hooks."
    )
    render_artifact_sections(
        title=f"active/ (institution `{inst_final}`)",
        paths=active_paths,
    )
