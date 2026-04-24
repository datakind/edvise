"""HITL items / silver JSON and UC actions for a single group — opened from the reviews table or URL."""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.databricks_uc_sql import get_warehouse_id
from hitl_reviewer.hitl_streamlit import (
    get_nav_from_session_or_url,
    init_reviewer_in_session,
    load_hitl_group_rows,
    maybe_hydrate_sidebar_from_nav,
    render_connection_sidebar,
    render_group_loop,
)

st.set_page_config(page_title="HITL — items & JSON", layout="wide")
st.title("HITL — items and JSON")
st.caption(
    "One ``(onboard_run_id, phase, artifact_type)`` group. Use the reviews table to pick a run, or share "
    "a URL with ``catalog``, ``onboard_run_id``, ``phase``, and ``artifact_type`` in the query string."
)

init_reviewer_in_session()
maybe_hydrate_sidebar_from_nav()
catalog, _sidebar = render_connection_sidebar()
try:
    get_warehouse_id()
except RuntimeError:
    st.stop()

c, o, ph, at = get_nav_from_session_or_url()
if not (c and o and ph and at):
    st.info(
        "No review group selected. On the **app** (home) page, use **Open in HITL items page** after a "
        "row appears in the table, or set query parameters: ``?catalog=…&onboard_run_id=…&phase=…&artifact_type=…``"
    )
    st.page_link("app.py", label="Back to HITL reviews table")
    st.stop()

st.caption(f"Group: ``{o}`` · ``{ph}`` · ``{at}`` · catalog **{catalog}**")
try:
    df = load_hitl_group_rows(catalog, o, ph, at)
except Exception as e:  # noqa: BLE001
    st.exception(e)
    st.stop()
if df.empty:
    st.warning("No ``hitl_reviews`` rows for this group in the current catalog. Check the catalog in the sidebar.")
    st.page_link("app.py", label="Back to HITL reviews table")
    st.stop()

st.subheader("Actions")
st.caption(
    "**IA grain**, **IA term**, and **SMA** (cohort/course manifest): while the UC row is **pending**, "
    "**Save JSON & approve UC** (or **Approve** on grain/term) writes silver JSON and approves UC. "
    "After UC finalizes, the editor is read-only. Other HITL JSON uses the same **Save JSON** button with "
    "the same rules."
)
render_group_loop(df, catalog=catalog)

st.page_link("app.py", label="Back to HITL reviews table")
