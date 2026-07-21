"""
Manifest Explorer: one consolidated table joining the Step 2a field mapping manifest
(``manifest_map.json``), per-column stats from the enriched schema contract, and HITL
review status — so reviewing a manifest doesn't require opening several separate JSON
files (see ``pages/2_Maps_and_Outputs.py`` for the raw per-file browser this complements).

Catalog comes from ``GENAI_HITL_CATALOG`` / default, same as **Maps & outputs**.
"""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from hitl_reviewer.platform.databricks_uc_sql import load_onboard_runs_hitl_complete
from hitl_reviewer.platform.volume_path_utils import institution_id_from_silver_volume_path
from hitl_reviewer.ui._shared import inject_hitl_css, render_institution_line
from hitl_reviewer.ui.hitl_streamlit import (
    default_catalog,
    init_reviewer_in_session,
    init_sidebar_form_state,
    render_warehouse_sidebar,
    validate_catalog,
)
from hitl_reviewer.ui.manifest_explorer import (
    attach_column_stats,
    attach_hitl_status,
    flatten_manifest_envelope,
    full_column_sample_values,
    load_json_object_or_none,
    resolve_explorer_paths,
)
from hitl_reviewer.ui.sma.enriched_schema_contract import visualize_value_whitespace
from hitl_reviewer.utils.institution_naming import format_institution_display_name

st.set_page_config(page_title="Manifest explorer", layout="wide")
init_reviewer_in_session()
init_sidebar_form_state()
inject_hitl_css()

st.title("Manifest explorer")
st.caption(
    "One table for the Step 2a field mapping manifest: source table/column, join + row "
    "selection, confidence, HITL status, and source-column stats (null rate, unique count, "
    "sample values) from the enriched schema contract — joined by target field so you don't "
    "have to open ``manifest_map.json``, ``enriched_schema_contract.json``, and the HITL "
    "manifests separately."
)

warehouse_ok = render_warehouse_sidebar(
    page_heading="Manifest explorer",
    page_caption="Warehouse must be configured (``DATABRICKS_WAREHOUSE_ID``). Catalog is not edited here.",
)
if not warehouse_ok:
    st.stop()

try:
    catalog = validate_catalog(default_catalog())
except ValueError as e:
    st.error(str(e))
    st.stop()

st.caption(f"**Queries use catalog:** `{catalog}` (`GENAI_HITL_CATALOG` / default).")

if st.button("Refresh run list from Unity Catalog"):
    for k in ("manifest_explorer_runs_df", "manifest_explorer_runs_err"):
        st.session_state.pop(k, None)
    st.rerun()

if "manifest_explorer_runs_df" not in st.session_state:
    try:
        raw = load_onboard_runs_hitl_complete(catalog)
        if raw.empty:
            st.session_state["manifest_explorer_runs_df"] = raw
        else:
            ap = raw["sample_artifact_path"].astype(str)
            onboard_mask = ap.str.contains("/runs/onboard/", case=False, na=False)
            work = raw.loc[onboard_mask].copy()

            def _resolve_institution(row: pd.Series) -> str:
                explicit = str(row.get("institution_id") or "").strip()
                if explicit:
                    return explicit
                p = str(row.get("sample_artifact_path") or "").strip()
                return institution_id_from_silver_volume_path(p) or ""

            work["_institution_id"] = work.apply(_resolve_institution, axis=1)
            work = work[work["_institution_id"].astype(str).str.len() > 0]
            st.session_state["manifest_explorer_runs_df"] = work
        st.session_state["manifest_explorer_runs_err"] = None
    except Exception as e:  # noqa: BLE001
        st.session_state["manifest_explorer_runs_df"] = pd.DataFrame()
        st.session_state["manifest_explorer_runs_err"] = str(e)

if st.session_state.get("manifest_explorer_runs_err"):
    st.error(st.session_state["manifest_explorer_runs_err"])
    st.stop()

runs_df: pd.DataFrame = st.session_state["manifest_explorer_runs_df"]
if runs_df.empty:
    st.warning(
        "No onboard runs with HITL complete (and resolvable institution) found. "
        "Try **Refresh run list** after more reviews finish, or confirm catalog and warehouse access."
    )
    st.stop()

institutions = sorted(runs_df["_institution_id"].astype(str).unique().tolist())
inst_pick = st.selectbox("Institution", options=institutions, index=0)
render_institution_line(inst_raw=inst_pick, format_fn=format_institution_display_name)

source = st.radio(
    "Manifest source",
    options=["Onboard run (HITL complete)", "Active"],
    horizontal=True,
)

run_pick: str | None = None
if source == "Active":
    paths = resolve_explorer_paths(inst_pick, catalog, onboard_run_id=None)
else:
    sub = runs_df.loc[runs_df["_institution_id"].astype(str) == inst_pick].copy()
    sub = sub.sort_values("last_reviewed_at", ascending=False, na_position="last")
    run_ids = (
        sub.drop_duplicates(subset=["onboard_run_id"], keep="first")["onboard_run_id"]
        .astype(str)
        .tolist()
    )
    if not run_ids:
        st.warning("No HITL-complete onboard runs found for this institution.")
        st.stop()
    run_pick = st.selectbox("Onboard run", options=run_ids, index=0)
    paths = resolve_explorer_paths(inst_pick, catalog, onboard_run_id=str(run_pick))

st.divider()
st.caption(f"**Manifest map:** `{paths['manifest_map']}`")


def _cached_json_object(path: str) -> dict | None:
    """Session-cached read; keyed the same way as the per-item SMA HITL panel's contract cache
    (``esc-json-{path}``) so switching between this page and HITL Review reuses one fetch."""
    if not path:
        return None
    ck = f"esc-json-{path}"
    if ck not in st.session_state:
        st.session_state[ck] = {"ok": False, "obj": None}
        obj = load_json_object_or_none(path)
        st.session_state[ck] = {"ok": obj is not None, "obj": obj}
    cached = st.session_state[ck]
    return cached.get("obj") if cached.get("ok") else None


if st.button("Reload manifest/contract from volume"):
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith("esc-json-"):
            del st.session_state[key]
    st.rerun()

envelope = _cached_json_object(paths["manifest_map"])
contract = _cached_json_object(paths["enriched_schema_contract"])
cohort_hitl = _cached_json_object(paths["cohort_hitl_manifest"]) if paths["cohort_hitl_manifest"] else None
course_hitl = _cached_json_object(paths["course_hitl_manifest"]) if paths["course_hitl_manifest"] else None

df = flatten_manifest_envelope(envelope)
df = attach_hitl_status(df, cohort_hitl=cohort_hitl, course_hitl=course_hitl)
df = attach_column_stats(df, contract)

availability = {
    "manifest_map": envelope is not None,
    "enriched_schema_contract": contract is not None,
    "cohort_hitl_manifest": bool(paths["cohort_hitl_manifest"]) and cohort_hitl is not None,
    "course_hitl_manifest": bool(paths["course_hitl_manifest"]) and course_hitl is not None,
}

missing = []
if not availability["manifest_map"]:
    missing.append("manifest_map.json")
if not availability["enriched_schema_contract"]:
    missing.append("enriched_schema_contract.json (column stats will be blank)")
if source != "Active":
    if not availability["cohort_hitl_manifest"]:
        missing.append("cohort_hitl_manifest.json (HITL status will be blank for cohort fields)")
    if not availability["course_hitl_manifest"]:
        missing.append("course_hitl_manifest.json (HITL status will be blank for course fields)")
if missing:
    st.warning("Could not read: " + "; ".join(missing))

if df.empty:
    st.info("No field mappings found at this path.")
    st.stop()

st.divider()
st.subheader("Filters")
f_cols = st.columns([1, 1, 1, 2])
with f_cols[0]:
    entity_pick = st.multiselect(
        "Entity type",
        options=sorted(df["entity_type"].dropna().unique().tolist()),
        default=sorted(df["entity_type"].dropna().unique().tolist()),
    )
with f_cols[1]:
    status_options = sorted([s for s in df["review_status"].dropna().unique().tolist() if s])
    status_pick = st.multiselect("Review status", options=status_options, default=[])
with f_cols[2]:
    flagged_only = st.checkbox("Flagged for HITL only", value=False)
with f_cols[3]:
    search = st.text_input(
        "Search (target field / source table / source column)", value=""
    )

filtered = df.copy()
if entity_pick:
    filtered = filtered[filtered["entity_type"].isin(entity_pick)]
if status_pick:
    filtered = filtered[filtered["review_status"].isin(status_pick)]
if flagged_only:
    filtered = filtered[filtered["flagged_for_hitl"] == True]  # noqa: E712
if search.strip():
    s = search.strip().lower()
    mask = (
        filtered["target_field"].astype(str).str.lower().str.contains(s, na=False)
        | filtered["source_table"].astype(str).str.lower().str.contains(s, na=False)
        | filtered["source_column"].astype(str).str.lower().str.contains(s, na=False)
    )
    filtered = filtered[mask]

st.caption(f"**{len(filtered)} of {len(df)}** field mappings shown.")

# Core columns fit on-screen without horizontal scroll; `join` / `row_selection` / values
# preview are intentionally left out by default since they're already shown in full in the
# detail panel below when you select a row — add them back here only if you want them visible
# across every row at once.
_CORE_COLS = [
    "entity_type",
    "target_field",
    "source_table",
    "source_column",
    "confidence",
    "review_status",
    "null_percentage",
    "sample_values",
]
_OPTIONAL_COLS = [
    "join",
    "row_selection",
    "flagged_for_hitl",
    "hitl_failure_mode",
    "dtype",
    "unique_count",
]
core_cols = [c for c in _CORE_COLS if c in filtered.columns]
optional_cols = [c for c in _OPTIONAL_COLS if c in filtered.columns]
extra_pick = st.multiselect(
    "+ more columns",
    options=optional_cols,
    default=[],
    help="Add columns to the table below. Full detail (join, row selection, all values) is "
    "always available in the detail panel after selecting a row, regardless of this picker.",
)
display_cols = core_cols + [c for c in extra_pick if c not in core_cols]

column_config = {
    "sample_values": st.column_config.TextColumn(
        "values (preview)",
        help=(
            "Unique values when the column has <=50 distinct values (per the enriched schema "
            "contract), else the 5 most frequent values. Truncated to 12 here — see the detail "
            "panel below for the full list. Leading/trailing whitespace in a value (e.g. a "
            "padded fixed-width source column) is shown as \u00b7 rather than stripped."
        ),
    ),
}

event = st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
    height=460,
    on_select="rerun",
    selection_mode="single-row",
    key="manifest_explorer_table",
    column_config=column_config,
)

st.download_button(
    "Download filtered table as CSV",
    data=filtered[display_cols].to_csv(index=False).encode("utf-8"),
    file_name=f"manifest_explorer_{inst_pick}_{run_pick or 'active'}.csv",
    mime="text/csv",
)

sel_rows = (event.get("selection") or {}).get("rows") or []
if sel_rows:
    ridx = filtered.index[sel_rows[0]]
    row = filtered.loc[ridx]
    st.divider()
    st.subheader(f"Detail — `{row.get('target_field')}` ({row.get('entity_type')})")

    d_cols = st.columns(2)
    with d_cols[0]:
        st.markdown("**Sourcing**")
        st.markdown(
            f"- Source: `{row.get('source_table')}` . `{row.get('source_column')}`\n"
            f"- Confidence: `{row.get('confidence')}`\n"
            f"- Review status: `{row.get('review_status') or '—'}`\n"
            f"- Flagged for HITL: `{row.get('flagged_for_hitl')}`"
            + (
                f" (`{row.get('hitl_failure_mode')}`)"
                if row.get("flagged_for_hitl")
                else ""
            )
        )
        if row.get("hitl_question"):
            st.caption(f"HITL question: {row.get('hitl_question')}")
        if row.get("_join_raw"):
            st.markdown("Join (raw)")
            st.json(row.get("_join_raw"))
        if row.get("_row_selection_raw"):
            st.markdown("Row selection (raw)")
            st.json(row.get("_row_selection_raw"))

    with d_cols[1]:
        st.markdown("**Source column detail**")
        chips, mode = full_column_sample_values(
            contract,
            source_table=row.get("source_table"),
            source_column=row.get("source_column"),
        )
        st.caption(
            f"dtype `{row.get('dtype') or '—'}` · null rate `{row.get('null_percentage')}"
            f"%` ({row.get('null_count')} rows) · unique `{row.get('unique_count')}`"
        )
        label = "Unique values" if mode == "unique" else "Sample values"
        st.caption(label)
        if chips:
            any_padded = False
            spans: list[str] = []
            for v in chips:
                display, had_padding = visualize_value_whitespace(v)
                any_padded = any_padded or had_padding
                cls = "hitl-chip hitl-chip-padded" if had_padding else "hitl-chip"
                title = (
                    "Source value has leading/trailing whitespace (e.g. a fixed-width "
                    "source column) — shown here as \u00b7 so it isn't confused with the "
                    "trimmed value."
                    if had_padding
                    else ""
                )
                spans.append(
                    f'<span class="{cls}" title="{html.escape(title)}">'
                    f"{html.escape(display)[:200]}</span>"
                )
            st.markdown(
                f'<div class="hitl-chip-row">{"".join(spans)}</div>',
                unsafe_allow_html=True,
            )
            if any_padded:
                st.caption(
                    "\u00b7 marks leading/trailing whitespace present in the raw source "
                    "value (not stripped — shown as-is)."
                )
        else:
            st.caption("—")

    if row.get("rationale"):
        st.markdown("**Model rationale**")
        st.caption(row.get("rationale"))
    if row.get("validation_notes"):
        st.markdown("**Predicted validation notes**")
        st.caption(row.get("validation_notes"))
    if row.get("reviewer_notes"):
        st.markdown("**Reviewer notes**")
        st.caption(row.get("reviewer_notes"))
else:
    st.caption("Select a row above to see full sourcing detail and source-column values.")
