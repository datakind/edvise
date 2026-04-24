"""
``hitl_reviews`` table, filters, and “open in HITL items” (see :mod:`hitl_reviewer.hitl_streamlit`).
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.hitl_streamlit import (
    HITL_REVIEW_HISTORY_SIDEBAR_CAPTION,
    display_columns,
    init_reviewer_in_session,
    init_sidebar_form_state,
    load_dataframe_for_sidebar,
    render_connection_sidebar,
    render_open_group_in_sidebar,
)

st.set_page_config(page_title="HITL Review History", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
catalog, sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=True,
    page_heading="HITL Review History",
    page_caption=HITL_REVIEW_HISTORY_SIDEBAR_CAPTION,
)
if not warehouse_ok:
    st.stop()

try:
    df = load_dataframe_for_sidebar(catalog, sidebar)
except Exception as e:  # noqa: BLE001
    st.exception(e)
    st.stop()

cols = display_columns(df)
st.subheader("Results")
st.dataframe(
    df[cols] if cols else df,
    use_container_width=True,
    hide_index=True,
)
if df.empty:
    st.info(
        "No rows match your current filters. If you only care about the **pending** queue, check **Unity Catalog** "
        "in the **sidebar**; to search older decisions, set **status** to **(any)** or a specific value, and raise "
        "the **row limit** if needed."
    )
    st.stop()

gdf = (
    df[["onboard_run_id", "phase", "artifact_type"]]
    .drop_duplicates()
    .sort_values(["onboard_run_id", "phase", "artifact_type"], na_position="last")
    .reset_index(drop=True)
)
render_open_group_in_sidebar(gdf, catalog)
