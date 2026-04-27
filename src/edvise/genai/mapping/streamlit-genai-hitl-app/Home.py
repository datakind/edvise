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

_HITL_BENCH = "HITL Review History"

st.set_page_config(page_title="HITL — home", layout="wide")
st.title("GenAI HITL")
st.caption(
    f"**How to use:** open **{_HITL_BENCH}** from the sidebar to work the queue "
    f"(``hitl_reviews`` table, filters, JSON/UC editor). This page is a home snapshot; add more app pages under ``pages/`` as needed."
)
st.page_link(
    HITL_WORKBENCH_PAGE,
    label=f"Open {_HITL_BENCH}",
    use_container_width=True,
)

st.divider()
st.subheader("What’s still pending?")
_PENDING_BLURB = (
    f"``status = pending`` in ``hitl_reviews`` for the default catalog; preview is capped at 50 rows "
    f"and matches the default queue in **{_HITL_BENCH}**."
)
st.caption(_PENDING_BLURB)
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
        st.info("No matching rows in this snapshot.")
except RuntimeError as e:
    st.caption(
        f"**Preview unavailable:** {e} After the warehouse is configured, open **{_HITL_BENCH}** from the sidebar."
    )
except Exception as e:  # noqa: BLE001
    st.warning(str(e))
