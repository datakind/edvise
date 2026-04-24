"""HITL items / silver JSON and UC actions for a single group — opened from the reviews table or URL."""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.hitl_streamlit import (
    HITL_ITEMS_SIDEBAR_CAPTION,
    get_nav_from_session_or_url,
    init_reviewer_in_session,
    init_sidebar_form_state,
    load_hitl_group_rows,
    maybe_hydrate_sidebar_from_nav,
    render_connection_sidebar,
    render_group_loop,
)

st.set_page_config(page_title="HITL — items & JSON", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
maybe_hydrate_sidebar_from_nav()

c, o, ph, at = get_nav_from_session_or_url()
nav_group_line: str | None = None
if c and o and ph and at:
    nav_group_line = f"**Group** · ``{o}`` · ``{ph}`` · ``{at}`` (URL / nav catalog: **{c}**)"

catalog, _sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=False,
    page_heading="HITL — items and JSON",
    page_caption=HITL_ITEMS_SIDEBAR_CAPTION,
    nav_group_line=nav_group_line,
)
if not warehouse_ok:
    st.stop()

if not (c and o and ph and at):
    st.info(
        "No review group selected. On the **HITL — reviews table** (home) page, use **Open in HITL items page** "
        "in the **sidebar** after a row appears, or set query parameters: "
        "``?catalog=…&onboard_run_id=…&phase=…&artifact_type=…``"
    )
    st.page_link("app.py", label="HITL reviews table (home)")
    st.stop()

try:
    df = load_hitl_group_rows(catalog, o, ph, at)
except Exception as e:  # noqa: BLE001
    st.exception(e)
    st.stop()
if df.empty:
    st.warning("No ``hitl_reviews`` rows for this group. Check **Unity Catalog** in the **sidebar**.")
    st.page_link("app.py", label="HITL reviews table (home)")
    st.stop()

st.subheader("Actions")
st.caption(
    "**IA grain**, **IA term**, and **SMA** (cohort/course manifest): while the UC row is **pending**, "
    "**Save JSON & approve UC** (or **Approve** on grain/term) writes silver JSON and approves UC. "
    "After UC finalizes, the editor is read-only. Other HITL JSON uses the same **Save JSON** button with "
    "the same rules."
)
render_group_loop(df, catalog=catalog)

with st.sidebar:
    st.divider()
    st.caption("Navigation")
    st.page_link("app.py", label="HITL reviews table (home)")
