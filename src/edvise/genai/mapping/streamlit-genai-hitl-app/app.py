"""
GenAI mapping — Unity Catalog ``hitl_reviews`` reviewer UI (multipage).

* **app** (this file): filter and browse ``hitl_reviews``; pick a group in the **sidebar** to open the
  JSON/UC view on **1_Hitl_Items**.

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
    export GENAI_HITL_CATALOG=dev_sst_02   # optional; default in sidebar
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
from hitl_reviewer.hitl_streamlit import (
    HITL_HOME_SIDEBAR_CAPTION,
    display_columns,
    init_reviewer_in_session,
    init_sidebar_form_state,
    load_dataframe_for_sidebar,
    render_connection_sidebar,
    render_open_group_in_sidebar,
)

st.set_page_config(page_title="HITL — reviews table", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
catalog, sidebar, warehouse_ok = render_connection_sidebar(
    show_table_query_filters=True,
    page_heading="HITL — reviews table",
    page_caption=HITL_HOME_SIDEBAR_CAPTION,
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
    st.info("No rows match. Clear filters or raise the row limit in the **sidebar**.")
    st.stop()

gdf = (
    df[["onboard_run_id", "phase", "artifact_type"]]
    .drop_duplicates()
    .sort_values(["onboard_run_id", "phase", "artifact_type"], na_position="last")
    .reset_index(drop=True)
)
render_open_group_in_sidebar(gdf, catalog)
