"""
IA (Identity Agent) **term** HITL reviewer (Streamlit).

Same interaction model as :mod:`hitl_reviewer.ia.grain_review_ui` (one item at a time,
option cards, batch **Approve** write + optional UC approve).

Silver path filename is ``identity_term_hitl.json`` (see ``edvise_genai_ia.resolve_run_paths``).
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from edvise.utils.institution_naming import format_institution_display_name
from hitl_reviewer._shared import (
    init_sel_key,
    inject_hitl_css,
    render_action_bar,
    render_hitl_header,
    render_option_cards,
)
from hitl_reviewer.hitl_json_batch_commit import persist_ia_term_hitl_from_session
from hitl_reviewer.silver_hitl_paths import set_item_choice, set_item_reviewer_note
from hitl_reviewer.unity_volume_files import read_unity_file_text, write_unity_file_text


def is_ia_term_phase(phase: str, artifact_type: str) -> bool:
    return str(phase).strip().lower() == "ia_gate_1" and str(artifact_type).strip().lower() == "term"


def is_term_domain_item(item: dict) -> bool:
    d = str(item.get("domain") or "").lower().strip()
    return d in ("term", "identity_term")


def term_item_indices(items: list[Any]) -> list[int]:
    out: list[int] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if not is_term_domain_item(it):
            continue
        opts = it.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        out.append(i)
    return out


def invalidate_ia_term_run_cache(onboard_run_id: str) -> None:
    k = f"ia-term-run-total-{str(onboard_run_id).strip()}"
    if k in st.session_state:
        del st.session_state[k]


def ia_term_run_total_items(pending_df: pd.DataFrame | None, onboard_run_id: str) -> int | None:
    """Count term HITL items across all ``term`` artifact paths for this onboard run."""
    if pending_df is None or pending_df.empty:
        return None
    rid = str(onboard_run_id).strip()
    cache_key = f"ia-term-run-total-{rid}"
    if cache_key in st.session_state:
        return int(st.session_state[cache_key])
    sub = pending_df[
        (pending_df["onboard_run_id"].astype(str) == rid)
        & (pending_df["phase"].astype(str).str.lower() == "ia_gate_1")
        & (pending_df["artifact_type"].astype(str).str.lower() == "term")
    ]
    total = 0
    for path in sub["artifact_path"].dropna().astype(str).str.strip().unique():
        if not path:
            continue
        try:
            raw = read_unity_file_text(path)
            data = json.loads(raw)
        except Exception:
            continue
        arr = data.get("items")
        if not isinstance(arr, list):
            continue
        for it in arr:
            if isinstance(it, dict) and is_term_domain_item(it):
                total += 1
    st.session_state[cache_key] = total
    return total


def _render_term_hitl_context(item: dict[str, Any]) -> None:
    hctx = item.get("hitl_context")
    if isinstance(hctx, str) and hctx.strip():
        st.subheader("Context")
        st.text(hctx.strip())
    elif isinstance(hctx, dict) and hctx:
        st.subheader("Context")
        st.json(hctx)
    else:
        st.caption("No ``hitl_context`` for this item.")


def render_ia_term_hitl_cards(
    *,
    data: dict[str, Any],
    items: list[Any],
    silver_path: str,
    sk: str,
    onboard_run_id: str,
    pending_df: pd.DataFrame | None,
    uc_group_pending: bool = False,
    approve_uc_if_complete: Callable[[], None] | None = None,
) -> None:
    inject_hitl_css()
    idxs = term_item_indices(items)
    if not idxs:
        st.warning(
            "This JSON has no **term** domain items with options — "
            "if this is ``identity_grain_hitl.json``, open the **grain** gate (artifact type ``grain``)."
        )
        return

    inst_raw = (data.get("institution_id") or "").strip()
    run_total = ia_term_run_total_items(pending_df, str(onboard_run_id)) if pending_df is not None else None
    nav_key = f"ia-term-nav-{sk}"
    if nav_key not in st.session_state:
        st.session_state[nav_key] = 0
    n_items = len(idxs)
    cur = max(0, min(int(st.session_state[nav_key]), n_items - 1))
    st.session_state[nav_key] = cur
    i = idxs[cur]
    item = items[i]
    if not isinstance(item, dict):
        st.error("Invalid HITL item.")
        return

    tbl = str(item.get("table") or "").replace("_", " ").strip().title() or "—"
    render_hitl_header(
        inst_raw=inst_raw,
        format_fn=format_institution_display_name,
        onboard_run_id=str(onboard_run_id),
        tbl=tbl,
        domain_label="Term",
        cur=cur,
        n_items=n_items,
        run_total=run_total,
        file_index=i,
        item_id=item.get("item_id", ""),
        inst_class="hitl-inst hitl-ia-inst",
    )

    q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
    st.markdown(
        f'<div class="hitl-qpanel hitl-ia-qpanel">{html.escape(q)}</div>',
        unsafe_allow_html=True,
    )

    _render_term_hitl_context(item)

    options = item.get("options")
    if not isinstance(options, list):
        options = []
    n_opt = len(options)
    sel_key = f"ia-term-sel-{sk}-{i}"
    init_sel_key(sel_key, item.get("choice"), n_opt)

    json_choice = item.get("choice")
    ia_rec_ix = (
        0
        if json_choice is None
        else max(0, min(int(json_choice) - 1, n_opt - 1))
    )

    render_option_cards(
        options=options,
        sel_key=sel_key,
        ia_rec_ix=ia_rec_ix,
        json_choice=json_choice,
        uc_group_pending=uc_group_pending,
        key_prefix="ia-term",
        sk=sk,
        file_index=i,
    )

    sel_j = int(st.session_state[sel_key])
    sel_opt = options[sel_j] if 0 <= sel_j < len(options) and isinstance(options[sel_j], dict) else {}
    reentry_sel = str(sel_opt.get("reentry") or "").lower()
    custom_key = f"ia-term-custom-{sk}-{i}"
    if custom_key not in st.session_state:
        existing = item.get("reviewer_note")
        st.session_state[custom_key] = str(existing or "") if existing else ""

    if reentry_sel == "generate_hook":
        st.text_area(
            "Describe the custom handling you want applied:",
            key=custom_key,
            height=120,
            disabled=not uc_group_pending,
        )

    render_action_bar(
        nav_key=nav_key,
        cur=cur,
        n_items=n_items,
        sk=sk,
        key_prefix="ia-term",
        file_index=i,
        include_prev_next=True,
        nav_prev_button_key=None,
        nav_next_button_key=None,
        primary_button_key=f"ia-term-save-all-{sk}",
        primary_button_label="Approve",
        primary_help=(
            "Writes **all** term ``choice`` values from this screen (and any already saved on "
            "disk) in one file write, then approves the UC ``hitl_reviews`` row when it is pending."
        ),
        pre_bar_caption=None,
        uc_group_pending=uc_group_pending,
        show_reject_item=True,
        persist_fn=lambda: persist_ia_term_hitl_from_session(
            silver_path=silver_path,
            sk=sk,
            allow_silver_write=uc_group_pending,
        ),
        reject_fn=lambda: _persist_term_reject(
            silver_path=silver_path,
            item_index=i,
            onboard_run_id=str(onboard_run_id),
            allow_write=uc_group_pending,
        ),
        after_persist_success=lambda: invalidate_ia_term_run_cache(str(onboard_run_id)),
        approve_fn=approve_uc_if_complete,
        success_silver_filename="identity_term_hitl.json",
    )


def _persist_term_reject(
    *, silver_path: str, item_index: int, onboard_run_id: str, allow_write: bool
) -> None:
    if not allow_write:
        st.error("Cannot write: this UC gate is not pending; silver JSON edits are disabled.")
        return
    try:
        fresh = json.loads(read_unity_file_text(silver_path))
    except Exception as e:  # noqa: BLE001
        st.error(f"Re-read failed: {e}")
        return
    try:
        set_item_choice(fresh, item_index, None)
        set_item_reviewer_note(
            fresh,
            item_index,
            "[HITL_UI_ITEM_REJECTED] Reviewer flagged this term item in the HITL app.",
        )
        out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
        write_unity_file_text(silver_path, out, overwrite=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Write failed: {e}")
    else:
        st.success("Item flagged and saved to silver volume.")
        invalidate_ia_term_run_cache(onboard_run_id)
        st.rerun()
