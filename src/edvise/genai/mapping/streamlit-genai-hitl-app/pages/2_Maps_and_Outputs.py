"""
Maps & outputs: browse silver **active/** or an **onboard run** (HITL-complete) for an institution.

Catalog comes from ``GENAI_HITL_CATALOG`` / default (not a sidebar control). Uses ``hitl_reviews`` aggregates
and the same ``…/genai_mapping/`` layout as the pipeline jobs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from hitl_reviewer.platform.databricks_uc_sql import (
    load_onboard_runs_hitl_complete,
)
from hitl_reviewer.platform.volume_path_utils import institution_id_from_silver_volume_path
from hitl_reviewer.ui.hitl_streamlit import (
    default_catalog,
    init_reviewer_in_session,
    init_sidebar_form_state,
    render_warehouse_sidebar,
    validate_catalog,
)
from hitl_reviewer.ui.run_artifacts_browser import (
    genai_mapping_root_uc,
    known_active_artifact_paths,
    known_onboard_run_artifact_paths,
    render_artifact_sections,
)

st.set_page_config(page_title="Maps & outputs", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()

st.title("Maps & outputs")
st.caption(
    "Choose an **institution**, then browse **active/** (promoted execute artifacts) or an **onboard run** "
    "whose HITL rows are all finalized (no ``pending``). Unity Catalog for queries is taken from the "
    "environment, same as **HITL Review History** when that sidebar field is left at its default."
)

warehouse_ok = render_warehouse_sidebar(
    page_heading="Maps & outputs",
    page_caption="Warehouse must be configured (``DATABRICKS_WAREHOUSE_ID``). Catalog is not edited here.",
)
if not warehouse_ok:
    st.stop()

try:
    catalog = validate_catalog(default_catalog())
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"**Queries use catalog:** `{catalog}` (`GENAI_HITL_CATALOG` / default).")

if st.button("Refresh run list from Unity Catalog"):
    for k in ("maps_outputs_df", "maps_outputs_err"):
        st.session_state.pop(k, None)
    st.rerun()

if "maps_outputs_df" not in st.session_state:
    try:
        raw = load_onboard_runs_hitl_complete(catalog)
        if raw.empty:
            st.session_state["maps_outputs_df"] = raw
        else:
            ap = raw["sample_artifact_path"].astype(str)
            onboard_mask = ap.str.contains("/runs/onboard/", case=False, na=False)
            work = raw.loc[onboard_mask].copy()

            def _resolve_institution(row: pd.Series) -> str:
                explicit = str(row.get("institution_id") or "").strip()
                if explicit:
                    return explicit
                p = str(row.get("sample_artifact_path") or "").strip()
                return institution_id_from_silver_volume_path(p) or ""

            work["_institution_id"] = work.apply(_resolve_institution, axis=1)
            work = work[work["_institution_id"].astype(str).str.len() > 0]
            st.session_state["maps_outputs_df"] = work
        st.session_state["maps_outputs_err"] = None
    except Exception as e:  # noqa: BLE001
        st.session_state["maps_outputs_df"] = pd.DataFrame()
        st.session_state["maps_outputs_err"] = str(e)

if st.session_state.get("maps_outputs_err"):
    st.error(st.session_state["maps_outputs_err"])
    st.stop()

df: pd.DataFrame = st.session_state["maps_outputs_df"]
if df.empty:
    st.warning(
        "No onboard runs with HITL complete (and resolvable institution) found. "
        "Try **Refresh run list** after more reviews finish, or confirm catalog and warehouse access."
    )
    st.stop()

institutions = sorted(df["_institution_id"].astype(str).unique().tolist())
inst_pick = st.selectbox("Institution", options=institutions, index=0)

source = st.radio(
    "Browse",
    options=["Onboard run (HITL complete)", "active/"],
    horizontal=True,
)

if source == "active/":
    genai_root = genai_mapping_root_uc(inst_pick, catalog)
    st.divider()
    st.markdown(f"**Silver genai_mapping root:** `{genai_root}`")
    st.caption(
        "Promoted hook packages may also live under ``active/identity_hooks/…`` — browse that folder "
        "in Unity Catalog when your onboard run materialized IA hooks."
    )
    render_artifact_sections(
        title=f"active/ — `{inst_pick}`",
        paths=known_active_artifact_paths(inst_pick, catalog),
    )
else:
    sub = df.loc[df["_institution_id"].astype(str) == inst_pick].copy()
    sub = sub.sort_values(
        "last_reviewed_at",
        ascending=False,
        na_position="last",
    )
    run_ids = (
        sub.drop_duplicates(subset=["onboard_run_id"], keep="first")["onboard_run_id"]
        .astype(str)
        .tolist()
    )
    if not run_ids:
        st.warning("No HITL-complete onboard runs found for this institution.")
        st.stop()
    run_pick = st.selectbox("Onboard run", options=run_ids, index=0)

    genai_root, onboard_paths = known_onboard_run_artifact_paths(
        inst_pick, catalog, str(run_pick).strip()
    )
    st.divider()
    st.markdown(f"**Silver genai_mapping root:** `{genai_root}`")
    render_artifact_sections(
        title=f"Onboard run `{run_pick}` — `{inst_pick}`",
        paths=onboard_paths,
    )
