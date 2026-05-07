"""
Streamlit HITL app: shared data loading, sidebar, and review-group rendering.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from hitl_reviewer.platform.databricks_uc_sql import (
    approve_or_reject,
    get_warehouse_id,
    hitl_group_identity_where_sql,
    hitl_reviews_fqn,
    pipeline_runs_fqn,
    run_query,
    sql_str,
)
from hitl_reviewer.persistence.hitl_json_batch_commit import (
    persist_hitl_choice_radios_from_session,
    try_approve_uc_after_json_write,
)
from hitl_reviewer.ui._shared import HITL_FLASH_HINT_AFTER_UC, set_hitl_flash_banner
from hitl_reviewer.ui.ia.grain_review_ui import (
    is_ia_grain_phase,
    render_ia_grain_hitl_cards,
)
from hitl_reviewer.ui.ia.term_review_ui import (
    is_ia_term_phase,
    render_ia_term_hitl_cards,
)
from hitl_reviewer.ui.ia.hook_preview_ui import (
    is_ia_hook_preview_phase,
    is_sma_transform_hook_preview_phase,
    render_ia_hook_preview_cards,
)
from hitl_reviewer.ui.sma.manifest_review_ui import (
    is_sma_phase,
    render_sma_hitl_cards,
)
from hitl_reviewer.ui.sma.transformation_review_ui import (
    is_sma_transformation_review_phase,
    render_sma_transformation_review_cards,
)
from hitl_reviewer.persistence.silver_hitl_paths import (
    artifact_path_contains_onboard_run_id,
)
from hitl_reviewer.ui.sma.enriched_schema_contract import silver_relative_path
from hitl_reviewer.platform.unity_volume_files import read_unity_file_text

# Primary HITL workbench: ``hitl_reviews`` table and per-group JSON/UC on the **same** page.
# Paths for ``st.page_link`` (relative to the main script, e.g. :file:`Home.py` for this app).
HITL_WORKBENCH_PAGE = "pages/1_HITL_Review.py"
HITL_REVIEW_HISTORY_PAGE = HITL_WORKBENCH_PAGE
HITL_ITEMS_PAGE = HITL_WORKBENCH_PAGE
MAPS_AND_OUTPUTS_PAGE = "pages/2_Maps_and_Outputs.py"

HITL_WORKBENCH_CAPTION = (
    "The workbench table **defaults to ``status = pending``**. Set **status** to **(any)** and add run/phase "
    "filters to search full history. **Select a row** to open the editor on the main page; **URLs** with "
    "``catalog``, ``onboard_run_id``, ``phase``, and ``artifact_type`` do the same."
)
HITL_REVIEW_HISTORY_SIDEBAR_CAPTION = HITL_WORKBENCH_CAPTION
HITL_ITEMS_SIDEBAR_CAPTION = HITL_WORKBENCH_CAPTION

# Session state keys for the selected (onboard_run_id, phase, artifact_type) + catalog.
KEY_NAV_CATALOG = "hitl_nav_catalog"
KEY_NAV_ONBOARD = "hitl_nav_onboard_run_id"
KEY_NAV_PHASE = "hitl_nav_phase"
KEY_NAV_ARTIFACT_TYPE = "hitl_nav_artifact_type"
KEY_HYDRATE_SIG = "hitl_sidebar_hydrate_sig"
# st.dataframe row selection in workbench: must match :file:`pages/1_HITL_Review.py`
HITL_RESULTS_DF_KEY = "hitl_workbench_results_df"
# Home “What’s still pending?” table: selecting a row switches to the workbench page.
HITL_HOME_PENDING_DF_KEY = "hitl_home_pending_results_df"

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
        raise ValueError(
            "Catalog must be a simple identifier (letters, digits, underscore)."
        )
    return c


def _safe_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", str(s))[:80]


def _df_pending_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose ``status`` is pending (trimmed, case-insensitive)."""
    if df.empty or "status" not in df.columns:
        return df.iloc[0:0].copy()
    ok = df["status"].astype(str).str.strip().str.lower() == "pending"
    return df.loc[ok].copy()


def _mask_same_uc_group(
    df: pd.DataFrame,
    onboard_run_id: object,
    phase: object,
    artifact_type: object,
) -> pd.Series:
    """Align with SQL ``trim(cast(... AS STRING))`` / manual edits: strip UC join keys."""
    oid = str(onboard_run_id).strip()
    ph = str(phase).strip().lower()
    at = str(artifact_type).strip().lower()
    return (
        (df["onboard_run_id"].astype(str).str.strip() == oid)
        & (df["phase"].astype(str).str.strip().str.lower() == ph)
        & (df["artifact_type"].astype(str).str.strip().str.lower() == at)
    )


def _uc_gate_button_key(
    kind: str, onboard_run_id: str, phase: str, artifact_type: str
) -> str:
    """Sanitized Streamlit widget keys for UC approve/reject (phase/at may contain odd characters)."""
    return (
        f"{kind}-{_safe_key(str(onboard_run_id))}-{_safe_key(str(phase))}-"
        f"{_safe_key(str(artifact_type))}"
    )


def init_reviewer_in_session() -> None:
    """Default ``session_state['reviewer']`` for UC approve/reject (``GENAI_HITL_REVIEWER`` or ``USER``). No UI field."""
    if "reviewer" not in st.session_state:
        st.session_state["reviewer"] = (
            os.getenv("GENAI_HITL_REVIEWER", "").strip()
            or (
                os.getenv("USER", "") or os.getenv("USERNAME", "") or "reviewer"
            ).strip()
        )


def render_warehouse_sidebar(
    *,
    page_heading: str,
    page_caption: str = "",
) -> bool:
    """
    Minimal sidebar: SQL warehouse check only (no Unity Catalog or table filters).

    Use when the page reads catalog from :func:`default_catalog` / environment instead of user input.
    """
    warehouse_ok = False
    with st.sidebar:
        st.subheader(page_heading)
        if (page_caption or "").strip():
            st.caption(page_caption.strip())
        st.divider()
        try:
            get_warehouse_id()
            warehouse_ok = True
        except RuntimeError as e:
            st.error(str(e))
    return warehouse_ok


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
        # Default the work queue to pending; use "(any)" on the History page when you need to search
        # approved/rejected or all states.
        st.session_state["sidebar_f_status"] = "pending"


def render_connection_sidebar(
    *,
    show_table_query_filters: bool = True,
    page_heading: str = "HITL",
    page_caption: str = "",
    nav_group_line: str | None = None,
) -> tuple[str, SidebarState, bool]:
    """
    Renders ``st.sidebar`` with Databricks, Unity Catalog, and (on the home page) table filters.

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
            warehouse_ok = True
        except RuntimeError as e:
            st.error(str(e))
        st.subheader("Connection & filters")
        st.text_input(
            "Unity Catalog",
            key="sidebar_catalog",
            help="UC for ``genai_mapping.hitl_reviews``.",
        )
        catalog_in = (
            st.session_state.get("sidebar_catalog") or default_catalog() or ""
        ).strip()
        try:
            catalog = validate_catalog(catalog_in)
        except ValueError as e:
            st.warning(str(e))
            catalog = default_catalog()
        if show_table_query_filters:
            st.number_input(
                "Row limit", min_value=50, max_value=5000, step=50, key="sidebar_limit"
            )
            st.caption(
                "**Table filters** — for the **hitl_reviews** result set above (defaults to **pending**)."
            )
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
                help="Re-runs the table query with the current filters.",
                key="sidebar_refresh",
            )
        else:
            st.caption(
                "**Table filters and row limit** are on the **HITL Review** workbench. "
                "Use **Unity Catalog** there for the HITL editor below."
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
        where.append(
            f"trim(cast(h.onboard_run_id AS STRING)) = trim({sql_str(onboard_run_id.strip())})"
        )
    if (phase or "").strip():
        where.append(
            f"lower(trim(cast(h.phase AS STRING))) = lower({sql_str(phase.strip())})"
        )
    if (status or "").strip():
        where.append(
            f"lower(trim(cast(h.status AS STRING))) = lower({sql_str(status.strip())})"
        )
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
    _, _, _, where_h = hitl_group_identity_where_sql(
        onboard_run_id=onboard_run_id,
        phase=phase,
        artifact_type=artifact_type,
        table_alias="h",
    )
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
      ON trim(cast(h.onboard_run_id AS STRING)) = trim(cast(p.onboard_run_id AS STRING))
     AND p.`catalog` = {c_sql}
    WHERE {where_h}
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
    return df[
        df["onboard_run_id"].astype(str).str.lower().str.contains(needle, na=False)
    ]


def _tuple_unpack_filters(
    sidebar: SidebarState,
) -> tuple[str | None, str | None, str | None, bool, int]:
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
    after_uc_approve_success: Callable[[], None] | None = None,
) -> None:
    is_sma = is_sma_phase(phase, artifact_type)
    is_ia_grain = is_ia_grain_phase(phase, artifact_type)
    is_ia_term = is_ia_term_phase(phase, artifact_type)
    is_hook_preview = is_ia_hook_preview_phase(
        phase, artifact_type
    ) or is_sma_transform_hook_preview_phase(phase, artifact_type)
    is_tr_review = is_sma_transformation_review_phase(str(phase), str(artifact_type))

    sk = f"{_safe_key(onboard_run_id)}-{_safe_key(phase)}-{_safe_key(artifact_type)}"
    reject_uc_key = _uc_gate_button_key(
        "r", str(onboard_run_id), str(phase), str(artifact_type)
    )

    def _reject_uc_row() -> None:
        approve_or_reject(
            catalog,
            str(onboard_run_id),
            str(phase),
            str(artifact_type),
            st.session_state["reviewer"],
            "rejected",
        )
        advance_to_next_pending_group(
            catalog=str(catalog),
            current_onboard_run_id=str(onboard_run_id),
            current_phase=str(phase),
            current_artifact_type=str(artifact_type),
        )
        set_hitl_flash_banner(
            "warning",
            "UC row rejected. " + HITL_FLASH_HINT_AFTER_UC,
        )
        st.toast("UC row rejected.", icon="⛔")
        st.rerun()

    # Same session key as ``st.text_input(..., key=f"path-{sk}")`` in ``render_one_hitl_group`` (Path details).
    silver_path = (
        st.session_state.get(f"path-{sk}") or default_artifact_path or ""
    ).strip()

    if not silver_path:
        return
    if not silver_path.startswith("/Volumes/"):
        st.warning(
            "Path should be an absolute Unity Catalog volume path starting with ``/Volumes/``."
        )
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

    if is_hook_preview:
        render_ia_hook_preview_cards(
            data=data,
            silver_path=silver_path,
            sk=sk,
            uc_group_pending=uc_group_pending,
            approve_uc_if_complete=lambda: approve_or_reject(
                catalog,
                str(onboard_run_id),
                str(phase),
                str(artifact_type),
                st.session_state["reviewer"],
                "approved",
            ),
            after_uc_approve_success=after_uc_approve_success,
            reject_uc_fn=_reject_uc_row,
            reject_uc_button_key=reject_uc_key,
        )
        return

    items = data.get("items")
    if not isinstance(items, list):
        items = []

    if is_tr_review:
        if not items:
            st.info(
                "No `items` in this transformation review JSON (empty file — nothing to review)."
            )
            return
        render_sma_transformation_review_cards(
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
            after_uc_approve_success=after_uc_approve_success,
            reject_uc_fn=_reject_uc_row,
            reject_uc_button_key=reject_uc_key,
        )
        return

    if not items:
        st.info(
            "No `items` in this HITL JSON, or the list is empty — nothing to select."
        )
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
            after_uc_approve_success=after_uc_approve_success,
            reject_uc_fn=_reject_uc_row,
            reject_uc_button_key=reject_uc_key,
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
            after_uc_approve_success=after_uc_approve_success,
            reject_uc_fn=_reject_uc_row,
            reject_uc_button_key=reject_uc_key,
        )
        return

    if is_sma:
        ph_lc = str(phase).strip().lower()
        if ph_lc == "sma_gate_2_hook_required":
            pend_ph = ph_lc
            pend_types = (
                "cohort_transformation_hook_hitl",
                "course_transformation_hook_hitl",
            )
        else:
            pend_ph = "sma_gate_1"
            pend_types = ("cohort_manifest", "course_manifest")
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
            after_uc_approve_success=after_uc_approve_success,
            pending_pair_phase=pend_ph,
            pending_pair_artifact_types=pend_types,
            reject_uc_fn=_reject_uc_row,
            reject_uc_button_key=reject_uc_key,
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
            st.caption(
                f"**{q}** — no `options` (direct manifest edit may be required)."
            )
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
                _uc_fail = (ap_err or "").strip() or "Unknown error."
                set_hitl_flash_banner(
                    "warning",
                    "JSON was saved, but the Unity Catalog gate did not finalize: "
                    f"{_uc_fail} Use **Refresh data** after fixing; you can retry **Save** if silver already looks correct.",
                )
                st.rerun()
            elif uc_group_pending:
                if after_uc_approve_success is not None:
                    after_uc_approve_success()
                st.success("Saved HITL JSON and approved the UC row.")
                st.toast("JSON + UC complete.", icon="✅")
                set_hitl_flash_banner(
                    "success",
                    "Saved HITL JSON and approved the UC row. "
                    + HITL_FLASH_HINT_AFTER_UC,
                )
            else:
                st.success(
                    "Saved HITL JSON. UC was not pending, so UC approve was skipped."
                )
                set_hitl_flash_banner(
                    "info",
                    "Saved HITL JSON. UC approve was skipped (row was not pending).",
                )
            st.rerun()


def apply_nav_from_results_dataframe_event(
    *,
    full_df: pd.DataFrame,
    catalog: str,
    event: object,
) -> None:
    """
    When the workbench table uses ``on_select="rerun"`` with row selection, set the HITL workbench
    group (session + query params) from the first selected row and :func:`st.rerun` if it changed.
    """
    if full_df is None or full_df.empty or event is None:
        return
    sel = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not sel:
        return
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    if not rows:
        return
    ri = int(rows[0])
    if ri < 0 or ri >= len(full_df):
        return
    row = full_df.iloc[ri]
    o = str(row.get("onboard_run_id", "") or "").strip()
    ph = str(row.get("phase", "") or "").strip()
    at = str(row.get("artifact_type", "") or "").strip()
    if not o or not ph or not at:
        return
    c_use = str(catalog).strip()
    c_cur, o_cur, ph_cur, at_cur = get_nav_from_session_or_url()
    if (
        c_cur
        and o_cur
        and ph_cur
        and at_cur
        and c_use == c_cur
        and o == o_cur
        and ph == ph_cur
        and at == at_cur
    ):
        return
    set_nav_selection(c_use, o, ph, at)
    st.rerun()


def apply_nav_from_home_pending_dataframe_event(
    *,
    full_df: pd.DataFrame,
    catalog: str,
    event: object,
) -> None:
    """
    Home pending snapshot: selecting a row sets the HITL workbench group and navigates to
    :data:`HITL_WORKBENCH_PAGE` (same keys as the workbench table, but uses ``st.switch_page``).
    """
    if full_df is None or full_df.empty or event is None:
        return
    sel = (
        event.get("selection")
        if isinstance(event, dict)
        else getattr(event, "selection", None)
    )
    if not sel:
        return
    rows = sel.get("rows") if isinstance(sel, dict) else getattr(sel, "rows", None)
    if not rows:
        return
    ri = int(rows[0])
    if ri < 0 or ri >= len(full_df):
        return
    row = full_df.iloc[ri]
    o = str(row.get("onboard_run_id", "") or "").strip()
    ph = str(row.get("phase", "") or "").strip()
    at = str(row.get("artifact_type", "") or "").strip()
    if not o or not ph or not at:
        return
    c_use = str(catalog).strip()
    c_cur, o_cur, ph_cur, at_cur = get_nav_from_session_or_url()
    if (
        c_cur
        and o_cur
        and ph_cur
        and at_cur
        and c_use == c_cur
        and o == o_cur
        and ph == ph_cur
        and at == at_cur
    ):
        return
    set_nav_selection(c_use, o, ph, at)
    st.switch_page(HITL_WORKBENCH_PAGE)


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


_NAV_QUERY_KEYS = ("catalog", "onboard_run_id", "phase", "artifact_type")


def clear_hitl_workbench_group_nav() -> None:
    """
    Drop the HITL workbench “selected group” from session and from URL query params.

    Call from the Home page (or any non-workbench page) so a previous table selection or
    in-app navigation does not keep showing after the user leaves and returns to Review History.
    Also clears the workbench table selection widget key so the grid does not keep a row highlighted.
    """
    for k in (
        KEY_NAV_CATALOG,
        KEY_NAV_ONBOARD,
        KEY_NAV_PHASE,
        KEY_NAV_ARTIFACT_TYPE,
        KEY_HYDRATE_SIG,
    ):
        st.session_state.pop(k, None)
    st.session_state.pop(HITL_RESULTS_DF_KEY, None)
    st.session_state.pop(HITL_HOME_PENDING_DF_KEY, None)
    qp = st.query_params
    for qk in _NAV_QUERY_KEYS:
        if qk in qp:
            del qp[qk]


def advance_to_next_pending_group(
    *,
    catalog: str,
    current_onboard_run_id: str,
    current_phase: str,
    current_artifact_type: str,
) -> bool:
    """
    Move editor navigation to the next pending group in the current workbench context.

    Preference order:
    1) Pending rows from the current sidebar-filtered result set.
    2) Fallback to a broader pending query.
    If no other pending group exists, clear selection.
    """
    try:
        sidebar = SidebarState(
            catalog=str(catalog).strip(),
            f_run=str(st.session_state.get("sidebar_f_run", "") or ""),
            f_phase=str(st.session_state.get("sidebar_f_phase", "") or ""),
            f_status=str(st.session_state.get("sidebar_f_status", "(any)") or "(any)"),
            limit=int(st.session_state.get("sidebar_limit", 500) or 500),
            refresh_clicked=False,
        )
        base_df = load_dataframe_for_sidebar(str(catalog).strip(), sidebar)
    except Exception:
        base_df = load_hitl_rows(
            str(catalog).strip(),
            onboard_run_id=None,
            phase=None,
            status="pending",
            limit=5000,
        )

    if base_df is None or base_df.empty:
        clear_hitl_workbench_group_nav()
        return False

    pending_df = _df_pending_rows(base_df)
    if pending_df.empty:
        clear_hitl_workbench_group_nav()
        return False

    groups = pending_df[["onboard_run_id", "phase", "artifact_type"]].drop_duplicates()
    if groups.empty:
        clear_hitl_workbench_group_nav()
        return False

    o_cur = str(current_onboard_run_id).strip()
    ph_cur = str(current_phase).strip()
    at_cur = str(current_artifact_type).strip()
    mask_current = (
        (groups["onboard_run_id"].astype(str).str.strip() == o_cur)
        & (groups["phase"].astype(str).str.strip() == ph_cur)
        & (groups["artifact_type"].astype(str).str.strip() == at_cur)
    )
    next_groups = groups[~mask_current]
    if next_groups.empty:
        clear_hitl_workbench_group_nav()
        return False

    nxt = next_groups.iloc[0]
    set_nav_selection(
        str(catalog).strip(),
        str(nxt.get("onboard_run_id", "")).strip(),
        str(nxt.get("phase", "")).strip(),
        str(nxt.get("artifact_type", "")).strip(),
    )
    return True


def _one_query_value(key: str) -> str | None:
    v = st.query_params.get(key)
    if v is None:
        return None
    if isinstance(v, list):
        return str(v[0]) if v else None
    return str(v) if v else None


def get_nav_from_session_or_url() -> tuple[
    str | None, str | None, str | None, str | None
]:
    """(catalog, onboard, phase, artifact_type) or Nones if any piece is missing."""
    c = (
        _one_query_value("catalog") or (st.session_state.get(KEY_NAV_CATALOG) or "")
    ).strip() or None
    o = (
        _one_query_value("onboard_run_id")
        or (st.session_state.get(KEY_NAV_ONBOARD) or "")
    ).strip() or None
    ph = (
        _one_query_value("phase") or (st.session_state.get(KEY_NAV_PHASE) or "")
    ).strip() or None
    at = (
        _one_query_value("artifact_type")
        or (st.session_state.get(KEY_NAV_ARTIFACT_TYPE) or "")
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
                with st.expander(
                    "Multiple artifact paths for this group", expanded=False
                ):
                    st.caption(
                        "Defaulting the editor to the first path. Pick another in the path field if needed."
                    )
                    st.code("\n".join(sorted(set(raw_paths))), language="text")
            sk = f"{_safe_key(str(onboard_run_id))}-{_safe_key(str(phase))}-{_safe_key(str(artifact_type))}"
            _is_sma = is_sma_phase(str(phase), str(artifact_type))
            _is_sma_tr = is_sma_transformation_review_phase(
                str(phase), str(artifact_type)
            )
            _is_ia_g = is_ia_grain_phase(str(phase), str(artifact_type))
            _is_ia_t = is_ia_term_phase(str(phase), str(artifact_type))
            _is_hook_pv = is_ia_hook_preview_phase(
                str(phase), str(artifact_type)
            ) or is_sma_transform_hook_preview_phase(str(phase), str(artifact_type))
            if _is_sma or _is_sma_tr or _is_ia_g or _is_ia_t or _is_hook_pv:
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
                after_uc_approve_success=lambda: advance_to_next_pending_group(
                    catalog=str(catalog),
                    current_onboard_run_id=str(onboard_run_id),
                    current_phase=str(phase),
                    current_artifact_type=str(artifact_type),
                ),
            )
        st.divider()
        is_ia_grain_row = is_ia_grain_phase(str(phase), str(artifact_type))
        is_ia_term_row = is_ia_term_phase(str(phase), str(artifact_type))
        is_hook_pv_row = is_ia_hook_preview_phase(
            str(phase), str(artifact_type)
        ) or is_sma_transform_hook_preview_phase(str(phase), str(artifact_type))
        is_sma_row = is_sma_phase(str(phase), str(artifact_type))
        is_sma_tr_row = is_sma_transformation_review_phase(
            str(phase), str(artifact_type)
        )
        if sub_pending.empty:
            st.caption(
                "This UC group is not **pending**. The editor above is **read-only**; silver JSON "
                "cannot be changed from this app after the gate is approved or rejected."
            )
        elif not (
            is_ia_grain_row
            or is_ia_term_row
            or is_hook_pv_row
            or is_sma_row
            or is_sma_tr_row
        ):
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button(
                    "Approve UC",
                    key=_uc_gate_button_key("a", onboard_run_id, phase, artifact_type),
                    type="primary",
                ):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "approved",
                        )
                        advance_to_next_pending_group(
                            catalog=str(catalog),
                            current_onboard_run_id=str(onboard_run_id),
                            current_phase=str(phase),
                            current_artifact_type=str(artifact_type),
                        )
                        set_hitl_flash_banner(
                            "success",
                            "UC row approved (silver JSON was not changed here — only ``hitl_reviews``). "
                            + HITL_FLASH_HINT_AFTER_UC,
                        )
                        st.toast("UC row approved.", icon="✅")
                        st.rerun()
                    except Exception as ex:  # noqa: BLE001
                        st.error(str(ex))
            with c2:
                if st.button(
                    "Reject gate",
                    key=_uc_gate_button_key(
                        "r-legacy", onboard_run_id, phase, artifact_type
                    ),
                ):
                    try:
                        approve_or_reject(
                            catalog,
                            str(onboard_run_id),
                            str(phase),
                            str(artifact_type),
                            st.session_state["reviewer"],
                            "rejected",
                        )
                        advance_to_next_pending_group(
                            catalog=str(catalog),
                            current_onboard_run_id=str(onboard_run_id),
                            current_phase=str(phase),
                            current_artifact_type=str(artifact_type),
                        )
                        set_hitl_flash_banner(
                            "warning",
                            "UC row rejected. " + HITL_FLASH_HINT_AFTER_UC,
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
    pending = _df_pending_rows(df)
    action_df = pending if not pending.empty else df
    if not pending.empty:
        st.success(f"{len(pending)} pending UC row(s) in the current result set.")

    groups = (
        action_df[["onboard_run_id", "phase", "artifact_type"]]
        .drop_duplicates()
        .sort_values(["onboard_run_id", "phase", "artifact_type"])
        .itertuples(index=False)
    )

    for onboard_run_id, phase, artifact_type in groups:
        sub = action_df[
            _mask_same_uc_group(action_df, onboard_run_id, phase, artifact_type)
        ]
        sub_pending = pending[
            _mask_same_uc_group(pending, onboard_run_id, phase, artifact_type)
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
