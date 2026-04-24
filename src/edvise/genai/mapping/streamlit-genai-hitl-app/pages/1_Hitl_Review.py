"""
``hitl_reviews`` table, filters, and per-group HITL JSON/UC in one place (``hitl_reviewer.hitl_streamlit``).
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.hitl_streamlit import (
    HITL_WORKBENCH_CAPTION,
    display_columns,
    get_nav_from_session_or_url,
    init_reviewer_in_session,
    init_sidebar_form_state,
    load_dataframe_for_sidebar,
    load_hitl_group_rows,
    maybe_hydrate_sidebar_from_nav,
    render_connection_sidebar,
    render_group_loop,
    render_group_picker_in_sidebar,
)

st.set_page_config(page_title="HITL Review", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
maybe_hydrate_sidebar_from_nav()

c0, o0, ph0, at0 = get_nav_from_session_or_url()
nav_group_line: str | None = None
if c0 and o0 and ph0 and at0:
    nav_group_line = f"**Selected group** · ``{o0}`` · ``{ph0}`` · ``{at0}`` (nav catalog **{c0}**)"

catalog, sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=True,
    page_heading="HITL Review",
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

if not df.empty:
    cols = display_columns(df)
    st.subheader("Results")
    st.dataframe(
        df[cols] if cols else df,
        use_container_width=True,
        hide_index=True,
    )
    gdf = (
        df[["onboard_run_id", "phase", "artifact_type"]]
        .drop_duplicates()
        .sort_values(["onboard_run_id", "phase", "artifact_type"], na_position="last")
        .reset_index(drop=True)
    )
    render_group_picker_in_sidebar(gdf, catalog)
else:
    st.info(
        "No rows match your current table filters. Set **status** to **(any)**, clear run/phase filters, "
        "or raise the **row limit** in the **sidebar**. You can still load a group from a **shared URL** below."
    )

# Editor: uses sidebar **Unity Catalog** + (onboard_run_id, phase, artifact_type) from nav / “Load this group”
c, o, ph, at = get_nav_from_session_or_url()
if c and o and ph and at:
    st.divider()
    st.subheader("HITL JSON and UC (this group)")
    st.caption(
        "**IA grain**, **IA term**, and **SMA** (cohort/course manifest): while the UC row is **pending**, "
        "**Save JSON & approve UC** (or **Approve** on grain/term) writes silver JSON and approves UC. "
        "After UC finalizes, the editor is read-only. Other HITL JSON uses the same **Save JSON** button with "
        "the same rules."
    )
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
else:
    st.caption(
        "Pick a **Group to review** in the **sidebar** (or open a **shared link** with ``catalog``, "
        "``onboard_run_id``, ``phase``, ``artifact_type``) to load the editor here."
    )
