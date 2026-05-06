"""
GenAI mapping — Unity Catalog ``hitl_reviews`` HITL reviewer UI (multipage).

* **Home** (this file): short landing. Main work: **HITL Review** — ``pages/1_HITL_Review.py``
  (``hitl_reviews`` table, filters, and per-group silver JSON/UC on one page).
* **Maps & outputs** — ``pages/2_Maps_and_Outputs.py``: institution dropdown, then **active/** or a **HITL-complete onboard run**,
  with catalog from the environment (not a sidebar field on that page).

HITL **choice** values live in JSON on the **silver** volume. **IA** / **SMA** write silver JSON
while the UC group is **pending**; use **HITL Review** for the full flow. See module doc in previous
revisions of this app for Databricks, paths, and deploy.

**Local run** (this directory)::

    export DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
    export GENAI_HITL_CATALOG=dev_sst_02
    streamlit run Home.py

**Databricks:** ``databricks bundle deploy``; app command uses ``streamlit run Home.py`` in ``databricks.yml``.
Library layout: ``hitl_reviewer/{platform,persistence,ui}/`` next to this file.
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.platform.databricks_uc_sql import get_warehouse_id
from hitl_reviewer.ui.hitl_streamlit import (
    HITL_HOME_PENDING_DF_KEY,
    HITL_WORKBENCH_PAGE,
    MAPS_AND_OUTPUTS_PAGE,
    apply_nav_from_home_pending_dataframe_event,
    clear_hitl_workbench_group_nav,
    default_catalog,
    display_columns,
    load_hitl_rows,
    validate_catalog,
)

_HITL_BENCH = "HITL Review"

st.set_page_config(page_title="HITL — home", layout="wide")
clear_hitl_workbench_group_nav()
st.title("GenAI HITL")
st.caption(
    f"**How to use:** open **{_HITL_BENCH}** from the sidebar to work the queue "
    f"(``hitl_reviews`` table, filters, JSON/UC editor), or **select a row** below to jump there. "
    f"This page is a home snapshot; add more app pages under ``pages/`` as needed."
)
st.page_link(
    HITL_WORKBENCH_PAGE,
    label=f"Open {_HITL_BENCH}",
    use_container_width=True,
)
st.page_link(
    MAPS_AND_OUTPUTS_PAGE,
    label="Open Maps & outputs (HITL-complete runs + active/)",
    use_container_width=True,
)

st.divider()
st.subheader("What’s still pending?")
_PENDING_BLURB = (
    f"``status = pending`` in ``hitl_reviews`` for the default catalog; preview is capped at 50 rows "
    f"and matches the default queue in **{_HITL_BENCH}**. **Select a row** to open that group in **{_HITL_BENCH}**."
)
st.caption(_PENDING_BLURB)
try:
    get_warehouse_id()
    _cat = validate_catalog(default_catalog())
    _pend = load_hitl_rows(
        _cat, onboard_run_id=None, phase=None, status="pending", limit=50
    )
    st.metric("Pending UC rows (up to 50 shown)", len(_pend))
    if not _pend.empty:
        _cols = display_columns(_pend)
        _display = _pend[_cols] if _cols else _pend
        _ev = st.dataframe(
            _display,
            use_container_width=True,
            hide_index=True,
            height=260,
            on_select="rerun",
            selection_mode="single-row",
            key=HITL_HOME_PENDING_DF_KEY,
        )
        apply_nav_from_home_pending_dataframe_event(
            full_df=_pend, catalog=_cat, event=_ev
        )
    else:
        st.info("No matching rows in this snapshot.")
except RuntimeError as e:
    st.caption(
        f"**Preview unavailable:** {e} After the warehouse is configured, open **{_HITL_BENCH}** from the sidebar."
    )
except Exception as e:  # noqa: BLE001
    st.warning(str(e))
