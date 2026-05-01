"""
``hitl_reviews`` table, filters, and per-group HITL JSON/UC in one place (``hitl_reviewer.ui.hitl_streamlit``).
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.ui._shared import render_hitl_flash_banner_if_any
from hitl_reviewer.ui.hitl_streamlit import (
    HITL_RESULTS_DF_KEY,
    HITL_WORKBENCH_CAPTION,
    apply_nav_from_results_dataframe_event,
    display_columns,
    get_nav_from_session_or_url,
    init_reviewer_in_session,
    init_sidebar_form_state,
    load_dataframe_for_sidebar,
    load_hitl_group_rows,
    maybe_hydrate_sidebar_from_nav,
    render_connection_sidebar,
    render_group_loop,
)

st.set_page_config(page_title="HITL Review History", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
maybe_hydrate_sidebar_from_nav()
render_hitl_flash_banner_if_any()

c0, o0, ph0, at0 = get_nav_from_session_or_url()
nav_group_line: str | None = None
if c0 and o0 and ph0 and at0:
    nav_group_line = (
        f"**Selected group** · ``{o0}`` · ``{ph0}`` · ``{at0}`` (nav catalog **{c0}**)"
    )

catalog, sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=True,
    page_heading="HITL Review History",
    page_caption=HITL_WORKBENCH_CAPTION,
    nav_group_line=nav_group_line,
)
if not warehouse_ok:
    st.stop()

try:
    df = load_dataframe_for_sidebar(catalog, sidebar)
except Exception as e:  # noqa: BLE001
    st.exception(e)
    st.stop()

st.subheader("Review History")
st.caption(
    "Select a **row** to open the JSON/UC editor below. **Save** (and **Approve** where shown) writes silver "
    "and updates pending Unity Catalog rows; after UC is finalized, the editor is read-only. "
    "A **shared link** with ``catalog``, ``onboard_run_id``, ``phase``, and ``artifact_type`` also opens a group."
)
if not df.empty:
    cols = display_columns(df)
    display_df = df[cols] if cols else df
    _results_event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=HITL_RESULTS_DF_KEY,
    )
    apply_nav_from_results_dataframe_event(
        full_df=df, catalog=catalog, event=_results_event
    )
else:
    st.info(
        "No rows match your current table filters. Set **status** to **(any)**, clear run/phase filters, "
        "or raise the **row limit** in the **sidebar**. You can still load a group from a **shared URL** below."
    )

# Editor: uses sidebar **Unity Catalog** + (onboard_run_id, phase, artifact_type) from table selection or URL
c, o, ph, at = get_nav_from_session_or_url()
if c and o and ph and at:
    st.divider()
    try:
        dgrp = load_hitl_group_rows(catalog, o, ph, at)
    except Exception as e:  # noqa: BLE001
        st.exception(e)
    else:
        if dgrp.empty:
            st.warning(
                "No ``hitl_reviews`` rows for this group in the catalog selected in the **sidebar**. "
                "Check **Unity Catalog** and the run id / phase, or the shared link’s ``catalog`` query parameter."
            )
        else:
            render_group_loop(dgrp, catalog=catalog)
