"""
Streamlit HITL app: shared data loading, sidebar, and review-group rendering.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from hitl_reviewer.databricks_uc_sql import (
    approve_or_reject,
    get_warehouse_id,
    hitl_reviews_fqn,
    pipeline_runs_fqn,
    run_query,
    sql_str,
)
from hitl_reviewer.hitl_json_batch_commit import (
    persist_hitl_choice_radios_from_session,
    try_approve_uc_after_json_write,
)
from hitl_reviewer.ia.grain_review_ui import is_ia_grain_phase, render_ia_grain_hitl_cards
from hitl_reviewer.ia.term_review_ui import is_ia_term_phase, render_ia_term_hitl_cards
from hitl_reviewer.sma.review_ui import is_sma_phase, render_sma_hitl_cards
from hitl_reviewer.silver_hitl_paths import artifact_path_contains_onboard_run_id
from hitl_reviewer.sma.enriched_schema_contract import silver_relative_path
from hitl_reviewer.unity_volume_files import read_unity_file_text

# Path passed to st.switch_page — relative to app entrypoint (app.py) directory.
HITL_ITEMS_PAGE = "pages/1_Hitl_Items.py"

# Sidebar copy (kept in one place for the two Streamlit entrypoints).
HITL_HOME_SIDEBAR_CAPTION = (
    "Browse ``hitl_reviews`` with the table filters, then open a run’s HITL JSON and UC actions on the **1_Hitl_Items** page."
)
HITL_ITEMS_SIDEBAR_CAPTION = (
    "One ``(onboard_run_id, phase, artifact_type)`` group. Pick a run from the home table, or open a link with "
    "``catalog``, ``onboard_run_id``, ``phase``, and ``artifact_type`` in the query string."
)

# Session state keys for navigating to the HITL items page.
KEY_NAV_CATALOG = "hitl_nav_catalog"
KEY_NAV_ONBOARD = "hitl_nav_onboard_run_id"
KEY_NAV_PHASE = "hitl_nav_phase"
KEY_NAV_ARTIFACT_TYPE = "hitl_nav_artifact_type"
KEY_HYDRATE_SIG = "hitl_sidebar_hydrate_sig"

_DISPLAY_COLS: tuple[str, ...] = (
    "institution_id",
    "onboard_run_id",
    "phase",
    "artifact_type",
    "artifact_path",
    "status",
    "reviewer",
    "reviewed_at",
)

_CATALOG_SAFE = re.compile(r"^[a-zA-Z0-9_]+$")


@dataclass(frozen=True)
class SidebarState:
    catalog: str
    f_run: str
    f_phase: str
    f_status: str
    limit: int
    refresh_clicked: bool


def default_catalog() -> str:
    for key in ("GENAI_HITL_CATALOG", "DB_workspace"):
        v = (os.getenv(key) or "").strip()
        if v:
            return v
    return "dev_sst_02"


def validate_catalog(catalog: str) -> str:
    c = str(catalog).strip()
    if not c or not _CATALOG_SAFE.match(c):
        raise ValueError("Catalog must be a simple identifier (letters, digits, underscore).")
    return c


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s))[:80]


def init_reviewer_in_session() -> None:
    if "reviewer" not in st.session_state:
        st.session_state["reviewer"] = os.getenv("GENAI_HITL_REVIEWER", "").strip() or (
            os.getenv("USER", "") or os.getenv("USERNAME", "") or "reviewer"
        ).strip()


def init_sidebar_form_state() -> None:
    """
    Seeds session keys used by `st.*(..., key=...)` *before* widgets are drawn.

    Avoids the Streamlit warning that appears when a widget is given `value=...` while
    the same `key` is also set via the Session State API (e.g. from `maybe_hydrate_sidebar_from_nav`).
    """
    if "sidebar_catalog" not in st.session_state:
        st.session_state["sidebar_catalog"] = default_catalog()
    if "sidebar_limit" not in st.session_state:
        st.session_state["sidebar_limit"] = 500
    if "sidebar_f_run" not in st.session_state:
        st.session_state["sidebar_f_run"] = ""
    if "sidebar_f_phase" not in st.session_state:
        st.session_state["sidebar_f_phase"] = ""
    if "sidebar_f_status" not in st.session_state:
        st.session_state["sidebar_f_status"] = "(any)"


def render_connection_sidebar(
    *,
    show_table_query_filters: bool = True,
    page_heading: str = "HITL",
    page_caption: str = "",
    nav_group_line: str | None = None,
) -> tuple[str, SidebarState, bool]:
    """
    Renders ``st.sidebar`` with Databricks, Unity Catalog, reviewer, and (on the home page) table filters.

    **Do not** pass ``value=`` to the Unity Catalog field; it uses ``key="sidebar_catalog"`` only, with
    the initial value coming from :func:`init_sidebar_form_state` and navigation from
    :func:`maybe_hydrate_sidebar_from_nav`.
    """
    warehouse_ok = False
    refresh_clicked = False
    with st.sidebar:
        st.subheader(page_heading)
        if (page_caption or "").strip():
            st.caption(page_caption.strip())
        if (nav_group_line or "").strip():
            st.caption(nav_group_line.strip())
        st.divider()
        try:
            get_warehouse_id()
            st.success("DATABRICKS_WAREHOUSE_ID is set")
            warehouse_ok = True
        except RuntimeError as e:
            st.error(str(e))
        st.subheader("Connection & filters")
        st.text_input("Unity Catalog", key="sidebar_catalog", help="UC for ``genai_mapping.hitl_reviews``.")
        catalog_in = (st.session_state.get("sidebar_catalog") or default_catalog() or "").strip()
        try:
            catalog = validate_catalog(catalog_in)
        except ValueError as e:
            st.warning(str(e))
            catalog = default_catalog()
        st.text_input(
            "Reviewer name (stored on approve/reject)",
            key="reviewer",
            help="Written to UC on approve or reject.",
        )
        if show_table_query_filters:
            st.number_input("Row limit", min_value=50, max_value=5000, step=50, key="sidebar_limit")
            st.caption("**Table filters** — for the home **HITL reviews** query only.")
            st.text_input("onboard_run_id contains", key="sidebar_f_run")
            st.text_input("phase equals", key="sidebar_f_phase")
            st.selectbox(
                "status",
                options=["(any)", "pending", "approved", "rejected"],
                key="sidebar_f_status",
            )
            refresh_clicked = st.button(
                "Refresh data",
                type="primary",
                help="Re-runs the home table query with the current filters.",
                key="sidebar_refresh",
            )
        else:
            st.caption(
                "**Table filters and row limit** are on the home page. This view loads a single HITL group; "
                "**Unity Catalog** and **reviewer** above still apply to this page."
            )

    limit = int(st.session_state.get("sidebar_limit", 500) or 500)
    f_run = str(st.session_state.get("sidebar_f_run", "") or "")
    f_phase = str(st.session_state.get("sidebar_f_phase", "") or "")
    f_status = str(st.session_state.get("sidebar_f_status", "(any)") or "(any)")
    state = SidebarState(
        catalog=catalog,
        f_run=f_run,
        f_phase=f_phase,
        f_status=f_status,
        limit=limit,
        refresh_clicked=refresh_clicked,
    )
    return catalog, state, warehouse_ok


def render_open_group_in_sidebar(
    gdf: pd.DataFrame,
    catalog: str,
) -> None:
    """Appends the “open in HITL items” block to :func:`st.sidebar` (call after the main table has loaded)."""
    with st.sidebar:
        st.divider()
        st.caption("**Open a HITL group** (JSON + UC on the next page)")
        cands = [
            f"{gdf['onboard_run_id'].iat[i]}  |  {gdf['phase'].iat[i]}  |  {gdf['artifact_type'].iat[i]}"
            for i in range(len(gdf))
        ]
        ix = st.selectbox(
            "``(onboard_run_id, phase, artifact_type)``",
            options=range(len(gdf)),
            format_func=lambda j: cands[j],
            key="home_group_ix",
        )
        if st.button("Open in HITL items page", type="primary", key="home_open_items"):
            row = gdf.iloc[int(ix)]
            set_nav_selection(
                str(catalog),
                str(row["onboard_run_id"]),
                str(row["phase"]),
                str(row["artifact_type"]),
            )
            st.switch_page(HITL_ITEMS_PAGE)


def load_hitl_rows(
    catalog: str,
    *,
    onboard_run_id: str | None,
    phase: str | None,
    status: str | None,
    limit: int,
) -> pd.DataFrame:
    t_h = hitl_reviews_fqn(catalog)
    t_p = pipeline_runs_fqn(catalog)
    where: list[str] = []
    c_sql = sql_str(str(catalog).strip())
    if (onboard_run_id or "").strip():
        where.append(f"h.onboard_run_id = {sql_str(onboard_run_id.strip())}")
    if (phase or "").strip():
        where.append(f"h.phase = {sql_str(phase.strip())}")
    if (status or "").strip():
        where.append(f"h.status = {sql_str(status.strip())}")
    w = f"WHERE {' AND '.join(where)}" if where else ""
    lim = max(1, min(int(limit), 5000))
    q = f"""
    SELECT
      h.onboard_run_id,
      h.phase,
      h.artifact_type,
      h.artifact_path,
      h.status,
      h.reviewer,
      h.reviewed_at,
      p.institution_id
    FROM {t_h} h
    LEFT JOIN {t_p} p
      ON h.onboard_run_id = p.onboard_run_id
     AND p.`catalog` = {c_sql}
    {w}
    ORDER BY h.reviewed_at DESC NULLS FIRST, h.onboard_run_id, h.phase, h.artifact_type, h.artifact_path
    LIMIT {lim}
    """
    return run_query(q)


def load_hitl_group_rows(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
) -> pd.DataFrame:
    t_h = hitl_reviews_fqn(catalog)
    t_p = pipeline_runs_fqn(catalog)
    c_sql = sql_str(str(catalog).strip())
    q = f"""
    SELECT
      h.onboard_run_id,
      h.phase,
      h.artifact_type,
      h.artifact_path,
      h.status,
      h.reviewer,
      h.reviewed_at,
      p.institution_id
    FROM {t_h} h
    LEFT JOIN {t_p} p
      ON h.onboard_run_id = p.onboard_run_id
     AND p.`catalog` = {c_sql}
    WHERE h.onboard_run_id = {sql_str(str(onboard_run_id).strip())}
      AND h.phase = {sql_str(str(phase).strip())}
      AND h.artifact_type = {sql_str(str(artifact_type).strip())}
    ORDER BY h.artifact_path
    """
    return run_query(q)


def apply_contains_filter(
    df: pd.DataFrame,
    f_run: str,
) -> pd.DataFrame:
    if not (f_run or "").strip():
        return df
    needle = f_run.strip().lower()
    return df[df["onboard_run_id"].astype(str).str.lower().str.contains(needle, na=False)]


def _tuple_unpack_filters(sidebar: SidebarState) -> tuple[str | None, str | None, str | None, bool, int]:
    use_contains = bool((sidebar.f_run or "").strip())
    onboard_exact = None if use_contains else ((sidebar.f_run or "").strip() or None)
    phase_f = (sidebar.f_phase or "").strip() or None
    status_f = None if sidebar.f_status == "(any)" else sidebar.f_status
    lim = sidebar.limit if not use_contains else min(sidebar.limit, 5000)
    return onboard_exact, phase_f, status_f, use_contains, lim


def load_dataframe_for_sidebar(catalog: str, sidebar: SidebarState) -> pd.DataFrame:
    onboard_exact, phase_f, status_f, use_contains, lim = _tuple_unpack_filters(sidebar)
    df = load_hitl_rows(
        catalog,
        onboard_run_id=onboard_exact,
        phase=phase_f,
        status=status_f,
        limit=lim,
    )
    if use_contains and (sidebar.f_run or "").strip():
        df = apply_contains_filter(df, sidebar.f_run)
    return df


def display_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in _DISPLAY_COLS if c in df.columns]


def render_silver_hitl_editor(
    *,
    catalog: str,
    default_artifact_path: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    pending_df: pd.DataFrame | None = None,
    uc_group_pending: bool = False,
) -> None:
    is_sma = is_sma_phase(phase, artifact_type)
    is_ia_grain = is_ia_grain_phase(phase, artifact_type)
    is_ia_term = is_ia_term_phase(phase, artifact_type)

    compact_chrome = is_ia_grain or is_ia_term or is_sma
    if not compact_chrome:
        st.caption(
            f"**artifact_type** in UC: ``{artifact_type}`` — **onboard_run_id** (this review block): "
            f"``{onboard_run_id}`` (in standard onboard layout it appears in the path under "
            f"``…/genai_mapping/runs/onboard/{{onboard_run_id}}/``)."
        )
        st.markdown(
            "**HITL JSON (silver volume)** — paths match ``edvise_genai_ia`` / ``edvise_genai_sma`` "
            "``resolve_run_paths``: ``{silver}/genai_mapping/runs/…/identity_agent/`` (IA) or "
            "``…/schema_mapping_agent/`` (SMA) with the HITL filenames. "
            "The ``onboard_run_id`` is the run folder in ``runs/onboard/…/``. "
            "Default file path = ``hitl_reviews.artifact_path`` (full string already includes that segment)."
        )
    sk = f"{_safe_key(onboard_run_id)}-{_safe_key(phase)}-{_safe_key(artifact_type)}"
    pkey = f"path-{sk}"
    silver_path = (st.session_state.get(pkey, default_artifact_path) or default_artifact_path or "").strip()

    if not silver_path:
        st.caption("Set the file path to load HITL JSON (typically the registered `artifact_path`).")
        return
    if not silver_path.startswith("/Volumes/"):
        st.warning("Path should be an absolute Unity Catalog volume path starting with ``/Volumes/``.")
    if not artifact_path_contains_onboard_run_id(silver_path, str(onboard_run_id)):
        st.warning(
            "This path string does not contain the **onboard_run_id** for this review row. "
            "Onboard HITL files usually include it under "
            f"``…/genai_mapping/runs/onboard/{onboard_run_id}/…``. "
            "Confirm the path, or keep going only if you intend another file (e.g. an execute run path). "
        )
    try:
        raw = read_unity_file_text(silver_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read file: {e}")
        return
    try:
        data: dict = json.loads(raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        return
    items = data.get("items")
    if not isinstance(items, list) or not items:
        st.info("No `items` in this HITL JSON, or the list is empty — nothing to select.")
        return

    if is_ia_grain:
        render_ia_grain_hitl_cards(
            data=data,
            items=items,
            silver_path=silver_path,
            sk=sk,
            onboard_run_id=str(onboard_run_id),
            pending_df=pending_df,
            uc_group_pending=uc_group_pending,
            approve_uc_if_complete=lambda: approve_or_reject(
                catalog,
                str(onboard_run_id),
                str(phase),
                str(artifact_type),
                st.session_state["reviewer"],
                "approved",
            ),
        )
        return

    if is_ia_term:
        render_ia_term_hitl_cards(
            data=data,
            items=items,
            silver_path=silver_path,
            sk=sk,
            onboard_run_id=str(onboard_run_id),
            pending_df=pending_df,
            uc_group_pending=uc_group_pending,
            approve_uc_if_complete=lambda: approve_or_reject(
                catalog,
                str(onboard_run_id),
                str(phase),
                str(artifact_type),
                st.session_state["reviewer"],
                "approved",
            ),
        )
        return

    if is_sma:
        render_sma_hitl_cards(
            data=data,
            items=items,
            silver_path=silver_path,
            sk=sk,
            onboard_run_id=str(onboard_run_id),
            pending_df=pending_df,
            uc_group_pending=uc_group_pending,
            approve_uc_if_complete=lambda: approve_or_reject(
                catalog,
                str(onboard_run_id),
                str(phase),
                str(artifact_type),
                st.session_state["reviewer"],
                "approved",
            ),
        )
        return

    def _default_choice_index(item: dict, n_opts: int) -> int:
        c_raw = item.get("choice")
        if c_raw is None:
            return 0
        try:
            c_int = int(c_raw)
        except (TypeError, ValueError):
            c_int = 1
        return max(0, min(n_opts - 1, c_int - 1))

    def _hitl_option_label(options: list, j: int) -> str:
        o = options[j]
        if isinstance(o, dict):
            lab = o.get("label")
            if lab is not None:
                return f"{j + 1}. {lab}"
        return f"{j + 1}. {o!r}"

    option_item_indices: list[int] = []
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        opts = item.get("options")
        if not opts or not isinstance(opts, list) or len(opts) < 1:
            continue
        option_item_indices.append(i)
        rk = f"sv{sk}item{i}{item.get('item_id', i)}"
        if rk not in st.session_state:
            st.session_state[rk] = _default_choice_index(item, len(opts))

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
        options = item.get("options")
        if not options or not isinstance(options, list):
            st.caption(f"**{q}** — no `options` (direct manifest edit may be required).")
            continue
        n = len(options)
        if n < 1:
            continue
        default_ix = _default_choice_index(item, n)
        rk = f"sv{sk}item{i}{item.get('item_id', i)}"
        st.radio(
            q,
            list(range(n)),
            index=default_ix,
            format_func=lambda j, opts=options: _hitl_option_label(opts, j),
            key=rk,
            disabled=not uc_group_pending,
        )

    def _approve_uc() -> None:
        approve_or_reject(
            catalog,
            str(onboard_run_id),
            str(phase),
            str(artifact_type),
            st.session_state["reviewer"],
            "approved",
        )

    if st.button(
        "Save JSON & approve UC",
        key=f"ssave{sk}",
        type="primary",
        disabled=not uc_group_pending,
        help="Writes every ``choice`` from the radios into this manifest, then approves the UC row when pending.",
    ):
        ok, err = persist_hitl_choice_radios_from_session(
            silver_path=silver_path,
            sk=sk,
            option_item_indices=option_item_indices,
            default_choice_index=_default_choice_index,
            allow_silver_write=uc_group_pending,
        )
        if not ok:
            st.error(err)
        else:
            ap_ok, ap_err = try_approve_uc_after_json_write(
                uc_group_pending=uc_group_pending,
                approve_uc_if_complete=_approve_uc,
            )
            if not ap_ok:
                st.warning(f"JSON saved, but UC approve failed: {ap_err}")
            elif uc_group_pending:
                st.success("Saved manifest JSON and approved the UC row.")
                st.toast("JSON + UC complete.", icon="✅")
            else:
                st.success("Saved manifest JSON. UC was not pending, so UC approve was skipped.")
            st.rerun()


def set_nav_selection(
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
) -> None:
    st.session_state[KEY_NAV_CATALOG] = catalog
    st.session_state[KEY_NAV_ONBOARD] = str(onboard_run_id)
    st.session_state[KEY_NAV_PHASE] = str(phase)
    st.session_state[KEY_NAV_ARTIFACT_TYPE] = str(artifact_type)
    p = st.query_params
    p["catalog"] = st.session_state[KEY_NAV_CATALOG]
    p["onboard_run_id"] = st.session_state[KEY_NAV_ONBOARD]
    p["phase"] = st.session_state[KEY_NAV_PHASE]
    p["artifact_type"] = st.session_state[KEY_NAV_ARTIFACT_TYPE]


def _one_query_value(key: str) -> str | None:
    v = st.query_params.get(key)
    if v is None:
        return None
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v) if v else None


def get_nav_from_session_or_url() -> tuple[str | None, str | None, str | None, str | None]:
    """(catalog, onboard, phase, artifact_type) or Nones if any piece is missing."""
    c = (_one_query_value("catalog") or (st.session_state.get(KEY_NAV_CATALOG) or "")).strip() or None
    o = (_one_query_value("onboard_run_id") or (st.session_state.get(KEY_NAV_ONBOARD) or "")).strip() or None
    ph = (_one_query_value("phase") or (st.session_state.get(KEY_NAV_PHASE) or "")).strip() or None
    at = (
        _one_query_value("artifact_type") or (st.session_state.get(KEY_NAV_ARTIFACT_TYPE) or "")
    ).strip() or None
    if c and o and ph and at:
        return c, o, ph, at
    c2 = (st.session_state.get(KEY_NAV_CATALOG) or "").strip() or None
    o2 = (st.session_state.get(KEY_NAV_ONBOARD) or "").strip() or None
    ph2 = (st.session_state.get(KEY_NAV_PHASE) or "").strip() or None
    at2 = (st.session_state.get(KEY_NAV_ARTIFACT_TYPE) or "").strip() or None
    if c2 and o2 and ph2 and at2:
        return c2, o2, ph2, at2
    return None, None, None, None


def maybe_hydrate_sidebar_from_nav() -> None:
    """Set ``sidebar_catalog`` from navigation once per target group so the SQL catalog matches the table page."""
    c, o, ph, at = get_nav_from_session_or_url()
    if not (c and o and ph and at):
        return
    sig = f"{c}|{o}|{ph}|{at}"
    if st.session_state.get(KEY_HYDRATE_SIG) == sig:
        return
    st.session_state["sidebar_catalog"] = c
    st.session_state[KEY_HYDRATE_SIG] = sig


def render_one_hitl_group(
    *,
    catalog: str,
    onboard_run_id: str,
    phase: str,
    artifact_type: str,
    sub: pd.DataFrame,
    sub_pending: pd.DataFrame,
    pending: pd.DataFrame,
    action_df: pd.DataFrame,
) -> None:
    # Bordered container: nested expanders are forbidden in Streamlit in some cases; HITL editor uses expanders.
    with st.container(border=True):
        _n_paths = len(sub)
        _hdr = f"`{onboard_run_id}` · `{phase}` · `{artifact_type}`"
        if _n_paths != 1:
            _hdr += f" · {_n_paths} paths"
        st.markdown(_hdr)
        raw_s = sub["artifact_path"].dropna().astype(str).str.strip()
        raw_paths = [p for p in raw_s.tolist() if p]
        if not raw_paths:
            st.warning(
                "No ``artifact_path`` on these rows; the pipeline has nothing to point the editor at. "
                "Re-run registration from ``edvise_genai_ia`` / ``edvise_genai_sma`` onboard."
            )
        else:
            default_path = raw_paths[0]
            if sub_pending.empty:
                st.warning(
                    "This `hitl_reviews` group is **not pending** (already approved or rejected). "
                    "Silver JSON is **read-only** here so UC and volume state cannot diverge."
                )
            if len(set(raw_paths)) > 1:
                with st.expander("Multiple artifact paths for this group", expanded=False):
                    st.caption("Defaulting the editor to the first path. Pick another in the path field if needed.")
                    st.code("\n".join(sorted(set(raw_paths))), language="text")
            sk = f"{_safe_key(str(onboard_run_id))}-{_safe_key(str(phase))}-{_safe_key(str(artifact_type))}"
            _is_sma = is_sma_phase(str(phase), str(artifact_type))
            _is_ia_g = is_ia_grain_phase(str(phase), str(artifact_type))
            _is_ia_t = is_ia_term_phase(str(phase), str(artifact_type))
            if _is_sma or _is_ia_g or _is_ia_t:
                _pl = "Silver JSON path" + (
                    " (read-only — UC gate not pending)" if sub_pending.empty else ""
                )
            elif sub_pending.empty:
                _pl = "UC file path (read-only — UC gate not pending)"
            else:
                _pl = "UC file path to read/write (absolute ``/Volumes/{catalog}/…_silver/…``)"
            with st.expander("📁 Path details", expanded=False):
                path_in = st.text_input(
                    _pl,
                    value=default_path,
                    key=f"path-{sk}",
                    disabled=sub_pending.empty,
                )
                _rel = silver_relative_path(path_in or "")
                if _rel:
                    st.caption(f"Volume-relative: `{_rel}`")
            render_silver_hitl_editor(
                catalog=catalog,
                default_artifact_path=default_path,
                onboard_run_id=str(onboard_run_id),
                phase=str(phase),
                artifact_type=str(artifact_type),
                pending_df=pending if not pending.empty else action_df,
                uc_group_pending=not sub_pending.empty,
            )
        st.divider()
        is_ia_grain_row = is_ia_grain_phase(str(phase), str(artifact_type))
        is_ia_term_row = is_ia_term_phase(str(phase), str(artifact_type))
        is_sma_row = is_sma_phase(str(phase), str(artifact_type))
        if sub_pending.empty:
            st.caption(
                "This UC group is not **pending**. The editor above is **read-only**; silver JSON "
                "cannot be changed from this app after the gate is approved or rejected."
            )
        elif is_ia_grain_row or is_ia_term_row or is_sma_row:
            st.caption(
                "IA / SMA: **Save JSON & approve UC** (or **Approve** on grain/term cards) approves this pending row. "
                "Use **Reject UC** only if you intend to block the gate."
            )
            if st.button("Reject UC", key=f"r-{onboard_run_id}-{phase}-{artifact_type}"):
                try:
                    approve_or_reject(
                        catalog,
                        str(onboard_run_id),
                        str(phase),
                        str(artifact_type),
                        st.session_state["reviewer"],
                        "rejected",
                    )
                    st.toast("UC row rejected.", icon="⛔")
                    st.rerun()
                except Exception as ex:  # noqa: BLE001
                    st.error(str(ex))
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Approve UC", key=f"a-{onboard_run_id}-{phase}-{artifact_type}", type="primary"):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "approved",
                        )
                        st.toast("UC row approved.", icon="✅")
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(str(ex))
            with c2:
                if st.button("Reject UC", key=f"r-{onboard_run_id}-{phase}-{artifact_type}-sma"):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "rejected",
                        )
                        st.toast("UC row rejected.", icon="⛔")
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(str(ex))
            with c3:
                st.caption("Updates ``hitl_reviews`` only (not the JSON file).")


def render_group_loop(
    df: pd.DataFrame,
    *,
    catalog: str,
) -> None:
    pending = df[df["status"].astype(str).str.lower() == "pending"].copy()
    action_df = pending if not pending.empty else df
    if pending.empty and not df.empty:
        n_all = len(df)
        st.info(
            f"**{n_all}** ``hitl_reviews`` row(s) match your filters, and **0** are ``status = pending``. "
            "That usually means these runs were already approved or rejected in Unity Catalog—**not** "
            "that the JSON is empty. The editor under each group is **read-only** until that group is "
            "``pending`` again (silver JSON cannot be changed from this app after UC finalizes). "
            "To list only gates awaiting UC, set sidebar **status** to **pending**."
        )
    elif not pending.empty:
        st.success(f"{len(pending)} pending UC row(s) in the current result set.")

    groups = (
        action_df[["onboard_run_id", "phase", "artifact_type"]]
        .drop_duplicates()
        .sort_values(["onboard_run_id", "phase", "artifact_type"])
        .itertuples(index=False)
    )

    for onboard_run_id, phase, artifact_type in groups:
        sub = action_df[
            (action_df["onboard_run_id"] == onboard_run_id)
            & (action_df["phase"] == phase)
            & (action_df["artifact_type"] == artifact_type)
        ]
        sub_pending = pending[
            (pending["onboard_run_id"] == onboard_run_id)
            & (pending["phase"] == phase)
            & (pending["artifact_type"] == artifact_type)
        ]
        render_one_hitl_group(
            catalog=catalog,
            onboard_run_id=str(onboard_run_id),
            phase=str(phase),
            artifact_type=str(artifact_type),
            sub=sub,
            sub_pending=sub_pending,
            pending=pending,
            action_df=action_df,
        )
