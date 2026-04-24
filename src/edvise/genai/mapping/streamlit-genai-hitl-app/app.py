"""
GenAI mapping — Unity Catalog ``hitl_reviews`` reviewer UI (multipage).

* **app** (this file): home / landing.
* **pages/1_Hitl_Review_History.py** — **HITL Review History** (``hitl_reviews`` table, filters, open a group to review).
* **pages/2_Hitl_Items.py** — HITL items, silver JSON, and UC for one
  ``(onboard_run_id, phase, artifact_type)`` group.

HITL **choice** values live in JSON on the **silver** volume under
``{silver_genai_mapping_root}/runs/...`` (see ``edvise_genai_ia`` / ``edvise_genai_sma``
``resolve_run_paths``). Paths come from ``hitl_reviews.artifact_path``. **IA grain**, **IA term**,
and **SMA** editors write silver JSON **only** while that UC group is **pending** (via **Save JSON &
approve UC** or **Approve**); once the gate is approved or rejected, the UI is read-only. Other phases
keep **Approve UC** / **Reject UC** under each run group. Unity Catalog: ``{catalog}.genai_mapping.hitl_reviews``.

**Local run** (repo root: ``uv pip install -e .`` or regenerate ``requirements.txt`` with
``python generate_requirements.py`` so ``edvise`` is installable; then from this directory,
with Databricks auth configured, e.g. ``databricks auth login``). Shared code lives in
``hitl_reviewer/`` next to ``app.py``::

    export DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id>
    export GENAI_HITL_CATALOG=dev_sst_02   # optional; default in the sidebar on data pages
    streamlit run app.py

**Databricks Apps (dev):** run ``databricks bundle deploy`` / ``databricks bundle run`` from
this directory (see ``databricks.yml``) or trigger CI
``.github/workflows/deploy-genai-hitl-app.yml`` (Actions → “Deploy GenAI HITL app (dev)”).
Use the same ``--var sql_warehouse_id=…`` and ``--var datakind_group_to_manage_workflow=…``
as the metadata dashboard app for dev.

The Databricks app identity (service principal) needs **read/write** on the silver
``/Volumes/.../..._silver/.../genai_mapping/...`` HITL paths, in addition to SQL access.
"""

from __future__ import annotations

import streamlit as st
from hitl_reviewer.hitl_streamlit import HITL_REVIEW_HISTORY_PAGE, HITL_ITEMS_PAGE, display_columns, default_catalog, load_hitl_rows, validate_catalog
from hitl_reviewer.databricks_uc_sql import get_warehouse_id

st.set_page_config(page_title="HITL — home", layout="wide")
st.title("GenAI HITL")
st.caption(
    "Unity Catalog ``hitl_reviews`` — on **HITL Review History** the table **defaults to pending** (your queue). "
    "Use filters there when you need to **search** full history."
)
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("HITL Review History")
    st.markdown(
        "Search and filter the full table (all statuses, run id, phase). The page opens with **pending** rows by default."
    )
    st.page_link(
        HITL_REVIEW_HISTORY_PAGE, label="Open HITL Review History", use_container_width=True
    )
with c2:
    st.subheader("HITL items")
    st.markdown(
        "Silver HITL JSON and **Save / approve UC** for **one** group. Choose the run on Review History, or open a link with query parameters."
    )
    st.page_link(HITL_ITEMS_PAGE, label="Open HITL items", use_container_width=True)

st.divider()
st.subheader("What’s still pending?")
st.caption(
    "``status = pending`` in ``hitl_reviews`` for this catalog (no search UI here). "
    "Use **HITL Review History** to filter by run, change status, or raise the row limit."
)
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
    st.caption(f"**Pending preview unavailable:** {e} Open **HITL Review History** after the warehouse is configured.")
except Exception as e:  # noqa: BLE001
    st.warning(str(e))
st.page_link(
    HITL_REVIEW_HISTORY_PAGE,
    label="Open HITL Review History (full search & filters)",
    use_container_width=True,
)

st.caption(
    "App navigation in the **sidebar** includes bookmarked **HITL items** URLs (``catalog``, ``onboard_run_id``, ``phase``, ``artifact_type``)."
)
