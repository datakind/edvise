"""
Streamlit UI: browse Unity Catalog rows for registered GenAI pipeline JSON artifacts.

Each row points at a JSON file on the institution bronze volume (paths only in UC).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from helpers import (
    build_genai_pipeline_artifacts_list_query,
    filter_artifacts_dataframe,
    get_genai_pipeline_artifacts_table_fqn,
    run_sql_query,
)

DISPLAY_COLUMNS_DEFAULT = [
    "institution_id",
    "pipeline_run_id",
    "pipeline_version",
    "artifact_kind",
    "uc_catalog",
    "relative_path",
    "absolute_path",
    "content_sha256",
    "registered_at",
]

st.set_page_config(
    page_title="GenAI pipeline artifacts",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=60)
def load_all_artifact_rows(table_fqn: str) -> pd.DataFrame:
    q = build_genai_pipeline_artifacts_list_query(table_fqn)
    return run_sql_query(q)


def main() -> None:
    st.title("GenAI pipeline artifacts (Unity Catalog)")
    table_fqn = get_genai_pipeline_artifacts_table_fqn()

    with st.sidebar:
        st.caption("Registry table")
        st.code(table_fqn, language="text")
        st.caption("Env: `DB_workspace`, optional `GENAI_ARTIFACTS_UC_*`")
        refresh = st.button("Refresh from UC")

    if refresh:
        st.cache_data.clear()

    try:
        df = load_all_artifact_rows(table_fqn)
    except Exception as exc:
        st.error(f"Could not load `{table_fqn}`: {exc}")
        st.stop()

    st.metric("Registered JSON artifacts (rows)", len(df))

    if df.empty:
        st.info("No rows in the registry yet.")
        return

    institutions = sorted(
        {str(x) for x in df["institution_id"].dropna().unique() if str(x).strip()}
    )
    kinds = sorted(
        {str(x) for x in df["artifact_kind"].dropna().unique() if str(x).strip()}
    )

    with st.sidebar:
        institution = st.selectbox(
            "Institution",
            options=[""] + institutions,
            format_func=lambda x: "(all)" if x == "" else x,
            help="Filter rows after loading the full table scan.",
        )
        kind_filter = st.multiselect(
            "Artifact kinds",
            options=kinds,
            help="Empty = all kinds.",
        )

    filtered = filter_artifacts_dataframe(
        df,
        institution_id=institution or None,
        artifact_kinds=kind_filter if kind_filter else None,
    )

    st.caption(f"Showing {len(filtered)} of {len(df)} rows.")

    show_cols = [c for c in DISPLAY_COLUMNS_DEFAULT if c in filtered.columns]
    other = [c for c in filtered.columns if c not in show_cols]
    column_order = show_cols + other

    st.dataframe(
        filtered.reindex(columns=column_order),
        use_container_width=True,
        hide_index=True,
    )


main()
