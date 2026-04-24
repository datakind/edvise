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
from hitl_reviewer.hitl_json_batch_commit import (
    persist_ia_term_hitl_from_session,
    try_approve_uc_after_json_write,
)
from hitl_reviewer.ia.grain_review_ui import inject_ia_grain_css
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
    inject_ia_grain_css()
    idxs = term_item_indices(items)
    if not idxs:
        st.warning(
            "This JSON has no **term** domain items with options — "
            "if this is ``identity_grain_hitl.json``, open the **grain** gate (artifact type ``grain``)."
        )
        return

    inst_raw = (data.get("institution_id") or "").strip()
    st.markdown(
        f'<p class="hitl-ia-inst">{html.escape(format_institution_display_name(inst_raw))}</p>',
        unsafe_allow_html=True,
    )

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
    meta_parts = [
        f'<span class="hitl-ia-meta">Onboard run <code>{html.escape(str(onboard_run_id))}</code></span>',
        f'<span class="hitl-ia-meta"> · Dataset: <strong>{html.escape(tbl)}</strong></span>',
        '<span class="ia-domain-pill">Term</span>',
    ]
    if run_total is not None:
        meta_parts.append(
            f'<span class="hitl-ia-meta"> · <strong>{cur + 1} of {n_items}</strong> items in this file '
            f"({int(run_total)} term item(s) on this run)</span>"
        )
    else:
        meta_parts.append(
            f'<span class="hitl-ia-meta"> · <strong>{cur + 1} of {n_items}</strong> items in this file</span>'
        )
    _item_id_esc = html.escape(str(item.get("item_id", "")))
    meta_parts.append(
        f'<span class="hitl-ia-meta"> · File index <code>{i}</code> · <code>{_item_id_esc}</code></span>'
    )
    st.markdown("<div>" + "".join(meta_parts) + "</div>", unsafe_allow_html=True)

    q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
    st.markdown(
        f'<div class="hitl-ia-qpanel">{html.escape(q)}</div>',
        unsafe_allow_html=True,
    )

    _render_term_hitl_context(item)

    options = item.get("options")
    if not isinstance(options, list):
        options = []
    n_opt = len(options)
    sel_key = f"ia-term-sel-{sk}-{i}"
    if sel_key not in st.session_state:
        c0 = item.get("choice")
        if c0 is None:
            st.session_state[sel_key] = 0
        else:
            try:
                st.session_state[sel_key] = max(0, min(int(c0) - 1, n_opt - 1))
            except (TypeError, ValueError):
                st.session_state[sel_key] = 0

    json_choice = item.get("choice")
    ia_rec_ix = (
        0
        if json_choice is None
        else max(0, min(int(json_choice) - 1, n_opt - 1))
    )

    st.subheader("Decision")
    for j, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        lab = str(opt.get("label") or f"Option {j + 1}")
        desc = str(opt.get("description") or "")
        selected = int(st.session_state[sel_key]) == j
        card_cls = "ia-opt-card ia-opt-card-sel" if selected else "ia-opt-card"
        badge = ""
        if j == ia_rec_ix:
            if json_choice is None:
                badge = '<span class="ia-rec-badge">IA recommendation</span>'
            else:
                badge = '<span class="ia-rec-badge">Saved in JSON</span>'
        st.markdown(
            f'<div class="{card_cls}"><p class="ia-opt-title">{html.escape(lab)}</p>{badge}'
            f'<div class="ia-opt-desc">{html.escape(desc)}</div></div>',
            unsafe_allow_html=True,
        )
        _sp, _sel = st.columns([4, 1], gap="small")
        with _sp:
            st.empty()
        with _sel:
            if st.button(
                "Select",
                key=f"ia-term-pick-{sk}-{i}-{j}",
                type="primary" if selected else "secondary",
                use_container_width=False,
                disabled=not uc_group_pending,
            ):
                st.session_state[sel_key] = j
                st.rerun()
        st.markdown('<div class="ia-grain-opt-after"></div>', unsafe_allow_html=True)

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

    with st.container(border=True):
        c_prev, c_next, c_save, c_rej = st.columns([1, 1, 2.6, 1.1], gap="small")
        with c_prev:
            if st.button("◀ Prev", key=f"ia-term-prev-{sk}", use_container_width=True):
                st.session_state[nav_key] = max(0, cur - 1)
                st.rerun()
        with c_next:
            if st.button("Next ▶", key=f"ia-term-nxt-{sk}", use_container_width=True):
                st.session_state[nav_key] = min(n_items - 1, cur + 1)
                st.rerun()
        with c_save:
            if st.button(
                "Approve",
                key=f"ia-term-save-all-{sk}",
                type="primary",
                use_container_width=True,
                disabled=not uc_group_pending,
                help=(
                    "Writes **all** term ``choice`` values from this screen (and any already saved on "
                    "disk) in one file write, then approves the UC ``hitl_reviews`` row when it is pending."
                ),
            ):
                ok, err = persist_ia_term_hitl_from_session(
                    silver_path=silver_path,
                    sk=sk,
                    allow_silver_write=uc_group_pending,
                )
                if not ok:
                    st.error(err)
                else:
                    invalidate_ia_term_run_cache(onboard_run_id)
                    ap_ok, ap_err = try_approve_uc_after_json_write(
                        uc_group_pending=uc_group_pending,
                        approve_uc_if_complete=approve_uc_if_complete,
                    )
                    if not ap_ok:
                        st.warning(
                            f"Silver JSON saved, but UC approve failed (fix and retry or use SQL): {ap_err}"
                        )
                    elif uc_group_pending and approve_uc_if_complete is not None:
                        st.success("Saved ``identity_term_hitl.json`` and approved the UC row.")
                        st.toast("JSON + UC complete.", icon="✅")
                    elif not uc_group_pending:
                        st.success(
                            "Saved ``identity_term_hitl.json``. UC row was not **pending**, so UC approve was skipped."
                        )
                    else:
                        st.success("Saved ``identity_term_hitl.json``.")
                    st.rerun()
        with c_rej:
            if st.button(
                "Reject item",
                key=f"ia-term-reject-{sk}-{i}",
                type="secondary",
                use_container_width=True,
                disabled=not uc_group_pending,
            ):
                _persist_term_reject(
                    silver_path=silver_path,
                    item_index=i,
                    onboard_run_id=str(onboard_run_id),
                    allow_write=uc_group_pending,
                )

    if uc_group_pending:
        st.caption("Approve saves your selections and marks this review complete.")
    else:
        st.caption("Read-only: UC gate is not pending; silver JSON cannot be changed from this app.")


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
