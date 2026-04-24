"""
IA (Identity Agent) **grain** HITL reviewer (Streamlit).

Structured ``hitl_context`` (candidate keys + variance profile) matches
:class:`~edvise.genai.mapping.identity_agent.hitl.schemas.GrainAmbiguityHITLContext`.

Silver path filename is ``identity_grain_hitl.json`` (see ``edvise_genai_ia.resolve_run_paths``).
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from edvise.utils.institution_naming import format_institution_display_name
from hitl_reviewer.hitl_json_batch_commit import (
    persist_ia_grain_hitl_from_session,
    try_approve_uc_after_json_write,
)
from hitl_reviewer.silver_hitl_paths import set_item_choice, set_item_reviewer_note
from hitl_reviewer.sma.enriched_schema_contract import silver_relative_path
from hitl_reviewer.unity_volume_files import read_unity_file_text, write_unity_file_text

_QUOTED = re.compile(r"'([^']{2,800})'|\"([^\"]{2,800})\"")


def is_ia_grain_phase(phase: str, artifact_type: str) -> bool:
    return str(phase).strip().lower() == "ia_gate_1" and str(artifact_type).strip().lower() == "grain"


def is_grain_domain_item(item: dict) -> bool:
    d = str(item.get("domain") or "").lower().strip()
    return d in ("grain", "identity_grain")


def grain_item_indices(items: list[Any]) -> list[int]:
    out: list[int] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if not is_grain_domain_item(it):
            continue
        opts = it.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        out.append(i)
    return out


def invalidate_ia_grain_run_cache(onboard_run_id: str) -> None:
    k = f"ia-grain-run-total-{str(onboard_run_id).strip()}"
    if k in st.session_state:
        del st.session_state[k]


def ia_grain_run_total_items(pending_df: pd.DataFrame | None, onboard_run_id: str) -> int | None:
    """Count grain HITL items across all ``grain`` artifact paths for this onboard run."""
    if pending_df is None or pending_df.empty:
        return None
    rid = str(onboard_run_id).strip()
    cache_key = f"ia-grain-run-total-{rid}"
    if cache_key in st.session_state:
        return int(st.session_state[cache_key])
    sub = pending_df[
        (pending_df["onboard_run_id"].astype(str) == rid)
        & (pending_df["phase"].astype(str).str.lower() == "ia_gate_1")
        & (pending_df["artifact_type"].astype(str).str.lower() == "grain")
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
            if isinstance(it, dict) and is_grain_domain_item(it):
                total += 1
    st.session_state[cache_key] = total
    return total


_IA_GRAIN_CSS_VER = "4"


def inject_ia_grain_css_once() -> None:
    if st.session_state.get("_hitl_ia_grain_css_ver") == _IA_GRAIN_CSS_VER:
        return
    st.markdown(
        """
<style>
.hitl-ia-inst { font-size: 1.85rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.35rem 0;
  letter-spacing: -0.02em; }
.hitl-ia-qpanel {
  font-size: 1.2rem; line-height: 1.55; font-weight: 500; margin: 1rem 0 1.25rem 0; padding: 1.1rem 1.25rem;
  background: rgba(99, 102, 241, 0.07); border: 1px solid rgba(99, 102, 241, 0.18); border-radius: 10px;
}
.hitl-ia-meta { font-size: 0.82rem; color: rgba(49, 51, 63, 0.75); margin-bottom: 0.35rem; opacity: 0.7; }
.hitl-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }
.hitl-chip { display: inline-block; padding: 0.12rem 0.55rem; border-radius: 999px; font-size: 0.78rem;
  background: rgba(111, 66, 193, 0.12); border: 1px solid rgba(111, 66, 193, 0.28); }
.ia-domain-pill {
  display: inline-block; padding: 0.15rem 0.65rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
  background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: rgb(6, 95, 70);
}
.ia-uni-good { color: rgb(5, 122, 85); font-weight: 600; }
.ia-uni-warn { color: rgb(180, 83, 9); font-weight: 600; }
.ia-cand-table { width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0.5rem 0 1rem 0; }
.ia-cand-table th, .ia-cand-table td { border: 1px solid rgba(0,0,0,0.08); padding: 0.45rem 0.55rem; vertical-align: top; }
.ia-cand-table th { background: rgba(0,0,0,0.04); text-align: left; }
.ia-notes { white-space: pre-wrap; word-break: break-word; max-width: 36rem; }
.ia-var-line { margin: 0.35rem 0 0.6rem 0; font-size: 0.92rem; line-height: 1.45; }
.ia-var-key { font-weight: 600; font-family: ui-monospace, monospace; }
.ia-opt-card {
  border: 2px solid rgba(0,0,0,0.1); border-radius: 10px; padding: 1rem 1.25rem 0.85rem; margin: 0;
  background: rgba(255,255,255,0.9);
  display: flex; flex-direction: column; justify-content: flex-start;
}
.ia-opt-card-sel {
  border-color: rgba(99, 102, 241, 0.85); background: rgba(99, 102, 241, 0.08);
}
.ia-opt-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.35rem; }
.ia-opt-desc { font-size: 0.9rem; color: rgba(49, 51, 63, 0.88); line-height: 1.45; flex: 1 1 auto; min-height: 2.5rem; }
.ia-variance-panel {
  border: 1px solid rgba(0,0,0,0.1); border-radius: 10px; padding: 0.75rem 1rem;
  margin: 0.75rem 0 1rem 0; background: rgba(0,0,0,0.02);
}
.ia-variance-panel h4 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; color: rgba(49, 51, 63, 0.95); }
.ia-rec-badge {
  display: block; font-size: 0.72rem; font-weight: 600; margin-bottom: 0.4rem; padding: 0.08rem 0.45rem;
  border-radius: 6px; background: rgba(59, 130, 246, 0.15); color: rgb(30, 64, 175);
}
.ia-grain-opt-after { margin-bottom: 0.65rem; height: 0; overflow: hidden; }
</style>
""",
        unsafe_allow_html=True,
    )
    st.session_state["_hitl_ia_grain_css_ver"] = _IA_GRAIN_CSS_VER


def _columns_chips_html(cols: Any) -> str:
    if isinstance(cols, str):
        cols = [c.strip() for c in cols.split(",") if c.strip()]
    if not isinstance(cols, list) or not cols:
        return "—"
    chips = "".join(
        f'<span class="hitl-chip">{html.escape(str(c).strip())}</span>'
        for c in cols
        if str(c).strip()
    )
    return f'<div class="hitl-chip-row">{chips}</div>' if chips else "—"


def _uniqueness_cell(score: Any) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return html.escape(str(score))
    pct = s * 100.0
    cls = "ia-uni-good" if s >= 0.999 else "ia-uni-warn"
    return f'<span class="{cls}">{pct:.2f}%</span>'


def _render_candidate_keys_table(hitl_ctx: dict[str, Any]) -> None:
    ck = hitl_ctx.get("candidate_keys")
    if not isinstance(ck, list) or not ck:
        st.caption("No structured ``candidate_keys`` in ``hitl_context``.")
        return
    st.subheader("Candidate keys")
    body_rows: list[str] = []
    for entry in ck:
        if not isinstance(entry, dict):
            continue
        rank = html.escape(str(entry.get("rank", "")))
        col_html = _columns_chips_html(entry.get("columns"))
        uni = _uniqueness_cell(entry.get("uniqueness_score"))
        notes = html.escape(str(entry.get("notes") or "").strip())
        body_rows.append(
            f"<tr><td>{rank}</td><td>{col_html}</td><td>{uni}</td><td class='ia-notes'>{notes}</td></tr>"
        )
    if not body_rows:
        st.caption("Candidate key entries were empty or malformed.")
        return
    tbl = (
        "<table class='ia-cand-table'><thead><tr>"
        "<th>Rank</th><th>Columns</th><th>Uniqueness</th><th>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    st.markdown(tbl, unsafe_allow_html=True)


def _variance_value_with_chips(text: str) -> str:
    """Highlight quoted examples in variance strings as chips."""
    if not text:
        return ""
    spans: list[tuple[int, int, str]] = []
    for m in _QUOTED.finditer(text):
        a, b = m.span()
        inner = m.group(1) or m.group(2) or ""
        if inner:
            spans.append((a, b, inner))
    if not spans:
        return html.escape(text)
    out: list[str] = []
    pos = 0
    for a, b, inner in spans:
        out.append(html.escape(text[pos:a]))
        out.append(
            f'<span class="hitl-chip">{html.escape(inner)}</span>'
        )
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _render_variance_profile(hitl_ctx: dict[str, Any]) -> None:
    vp = hitl_ctx.get("variance_profile")
    if not isinstance(vp, dict) or not vp:
        return
    # Always-visible panel (avoid ``st.expander`` — it collapses on ``st.rerun()`` after Select).
    inner: list[str] = [
        '<div class="ia-variance-panel">',
        "<h4>Where the ambiguity is</h4>",
    ]
    for col, val in vp.items():
        v = val if isinstance(val, str) else str(val)
        key_html = f'<span class="ia-var-key">{html.escape(str(col))}</span>'
        body = _variance_value_with_chips(v)
        inner.append(f'<div class="ia-var-line">{key_html}<br/>{body}</div>')
    inner.append("</div>")
    st.markdown("".join(inner), unsafe_allow_html=True)


def render_ia_grain_hitl_cards(
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
    inject_ia_grain_css_once()
    vol_rel = silver_relative_path(silver_path) or ""
    with st.expander("📁 Path details", expanded=False):
        st.text(silver_path or "")
        if vol_rel:
            st.caption(f"Volume-relative: `{vol_rel}`")
    idxs = grain_item_indices(items)
    if not idxs:
        st.warning(
            "This JSON has no **grain** domain items with options — "
            "if this is ``identity_term_hitl.json``, use the standard editor (or open "
            "``identity_grain_hitl.json``)."
        )
        return

    inst_raw = (data.get("institution_id") or "").strip()
    st.markdown(
        f'<p class="hitl-ia-inst">{html.escape(format_institution_display_name(inst_raw))}</p>',
        unsafe_allow_html=True,
    )

    run_total = ia_grain_run_total_items(pending_df, str(onboard_run_id)) if pending_df is not None else None
    nav_key = f"ia-grain-nav-{sk}"
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
        '<span class="ia-domain-pill">Grain</span>',
    ]
    if run_total is not None:
        meta_parts.append(
            f'<span class="hitl-ia-meta"> · <strong>{cur + 1} of {n_items}</strong> items in this file '
            f"({int(run_total)} grain item(s) on this run)</span>"
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

    hctx = item.get("hitl_context")
    if isinstance(hctx, dict):
        _render_candidate_keys_table(hctx)
        _render_variance_profile(hctx)
    elif isinstance(hctx, str) and hctx.strip():
        st.subheader("Context")
        st.text(hctx.strip())
    else:
        st.caption("No structured ``hitl_context`` for this item.")

    options = item.get("options")
    if not isinstance(options, list):
        options = []
    n_opt = len(options)
    sel_key = f"ia-grain-sel-{sk}-{i}"
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
                key=f"ia-grain-pick-{sk}-{i}-{j}",
                type="primary" if selected else "secondary",
                use_container_width=True,
            ):
                st.session_state[sel_key] = j
                st.rerun()
        st.markdown('<div class="ia-grain-opt-after"></div>', unsafe_allow_html=True)

    sel_j = int(st.session_state[sel_key])
    sel_opt = options[sel_j] if 0 <= sel_j < len(options) and isinstance(options[sel_j], dict) else {}
    reentry_sel = str(sel_opt.get("reentry") or "").lower()
    custom_key = f"ia-grain-custom-{sk}-{i}"
    if custom_key not in st.session_state:
        existing = item.get("reviewer_note")
        st.session_state[custom_key] = str(existing or "") if existing else ""

    if reentry_sel == "generate_hook":
        st.text_area(
            "Describe the custom handling you want applied:",
            key=custom_key,
            height=120,
        )

    with st.container(border=True):
        c_prev, c_next, c_save, c_rej = st.columns([1, 1, 2.6, 1.1], gap="small")
        with c_prev:
            if st.button("◀ Prev", key=f"ia-grain-prev-{sk}", use_container_width=True):
                st.session_state[nav_key] = max(0, cur - 1)
                st.rerun()
        with c_next:
            if st.button("Next ▶", key=f"ia-grain-nxt-{sk}", use_container_width=True):
                st.session_state[nav_key] = min(n_items - 1, cur + 1)
                st.rerun()
        with c_save:
            if st.button(
                "Approve",
                key=f"ia-grain-save-all-{sk}",
                type="primary",
                use_container_width=True,
                help=(
                    "Writes **all** grain ``choice`` values from this screen (and any already saved on "
                    "disk) in one file write, then approves the UC ``hitl_reviews`` row when it is pending."
                ),
            ):
                ok, err = persist_ia_grain_hitl_from_session(silver_path=silver_path, sk=sk)
                if not ok:
                    st.error(err)
                else:
                    invalidate_ia_grain_run_cache(onboard_run_id)
                    ap_ok, ap_err = try_approve_uc_after_json_write(
                        uc_group_pending=uc_group_pending,
                        approve_uc_if_complete=approve_uc_if_complete,
                    )
                    if not ap_ok:
                        st.warning(
                            f"Silver JSON saved, but UC approve failed (fix and retry or use SQL): {ap_err}"
                        )
                    elif uc_group_pending and approve_uc_if_complete is not None:
                        st.success("Saved ``identity_grain_hitl.json`` and approved the UC row.")
                        st.toast("JSON + UC complete.", icon="✅")
                    elif not uc_group_pending:
                        st.success(
                            "Saved ``identity_grain_hitl.json``. UC row was not **pending**, so UC approve was skipped."
                        )
                    else:
                        st.success("Saved ``identity_grain_hitl.json``.")
                    st.rerun()
        with c_rej:
            if st.button(
                "Reject item",
                key=f"ia-grain-reject-{sk}-{i}",
                type="secondary",
                use_container_width=True,
            ):
                _persist_grain_reject(
                    silver_path=silver_path,
                    item_index=i,
                    onboard_run_id=str(onboard_run_id),
                )

    st.caption("Approve saves your selections and marks this review complete.")


def _persist_grain_reject(*, silver_path: str, item_index: int, onboard_run_id: str) -> None:
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
            "[HITL_UI_ITEM_REJECTED] Reviewer flagged this grain item in the HITL app.",
        )
        out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
        write_unity_file_text(silver_path, out, overwrite=True)
    except Exception as e:  # noqa: BLE001
        st.error(f"Write failed: {e}")
    else:
        st.success("Item flagged and saved to silver volume.")
        invalidate_ia_grain_run_cache(onboard_run_id)
        st.rerun()
