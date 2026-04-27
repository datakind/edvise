"""
GenAI mapping — Unity Catalog ``hitl_reviews`` HITL reviewer UI (multipage).

* **Home** (this file): short landing. Main work: **HITL Review History** — ``pages/1_HITL_Review_History.py``
  (``hitl_reviews`` table, filters, and per-group silver JSON/UC on one page). Add more ``pages/`` as needed.

HITL **choice** values live in JSON on the **silver** volume. **IA** / **SMA** write silver JSON
while the UC group is **pending**; use **HITL Review History** for the full flow. See module doc in previous
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
    HITL_WORKBENCH_PAGE,
    default_catalog,
    display_columns,
    load_hitl_rows,
    validate_catalog,
)

st.set_page_config(page_title="HITL — home", layout="wide")
st.title("GenAI HITL")
st.caption(
    "The **HITL Review History** page combines the ``hitl_reviews`` table, search filters, and the JSON/UC editor in one place."
)
st.page_link(
    HITL_WORKBENCH_PAGE,
    label="HITL Review History (table + editor)",
    use_container_width=True,
)
st.caption("More pages can be added under ``pages/`` later. Use the app sidebar to switch between **Home** and **HITL Review History**.")

st.divider()
st.subheader("What’s still pending?")
st.caption("``status = pending`` in ``hitl_reviews`` for the default catalog (up to 50 rows). The **HITL Review History** table defaults to the same work queue with full search.")
try:
    get_warehouse_id()
    _cat = validate_catalog(default_catalog())
    _pend = load_hitl_rows(_cat, onboard_run_id=None, phase=None, status="pending", limit=50)
    st.metric("Pending UC rows (up to 50 shown)", len(_pend))
    if not _pend.empty:
        _cols = display_columns(_pend)
        st.dataframe(
            _pend[_cols] if _cols else _pend,
            use_container_width=True,
            hide_index=True,
            height=260,
        )
    else:
        st.info("No pending HITL gates in this catalog (or none in the first 50 rows for this query).")
except RuntimeError as e:
    st.caption(f"**Preview unavailable:** {e} After the warehouse is configured, open **HITL Review History** in the sidebar.")
except Exception as e:  # noqa: BLE001
    st.warning(str(e))
st.page_link(HITL_WORKBENCH_PAGE, label="Open HITL Review History (full workbench)", use_container_width=True)
