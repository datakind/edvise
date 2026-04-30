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
from hitl_reviewer.ui._shared import (
    init_sel_key,
    inject_hitl_css,
    mark_hitl_nav_visit,
    render_action_bar,
    render_hitl_header,
    render_option_cards,
)
from hitl_reviewer.persistence.hitl_json_batch_commit import (
    persist_ia_grain_hitl_from_session,
)
from hitl_reviewer.persistence.silver_hitl_paths import (
    set_item_choice,
    set_item_reviewer_note,
    silver_volume_path_session_tag,
)
from hitl_reviewer.platform.unity_volume_files import (
    read_unity_file_text,
    write_unity_file_text,
)

_QUOTED = re.compile(r"'([^']{2,800})'|\"([^\"]{2,800})\"")


def is_ia_grain_phase(phase: str, artifact_type: str) -> bool:
    return (
        str(phase).strip().lower() == "ia_gate_1"
        and str(artifact_type).strip().lower() == "grain"
    )


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


def ia_grain_run_total_items(
    pending_df: pd.DataFrame | None, onboard_run_id: str
) -> int | None:
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
    st.markdown("**Candidate keys**")
    st.caption(
        "**Rank 1** is the proposed grain: the same column set as **post_clean_primary_key** in the "
        "Identity Agent response. Ranks **2+** are documented alternatives (e.g. **candidate key override**), "
        "not a second “best” guess of rank 1. Uniqueness % is profiling evidence, not the sort order of ranks."
    )
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
        "</tr></thead><tbody>" + "".join(body_rows) + "</tbody></table>"
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
        out.append(f'<span class="hitl-chip">{html.escape(inner)}</span>')
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def _render_variance_profile(hitl_ctx: dict[str, Any]) -> None:
    vp = hitl_ctx.get("variance_profile")
    if not isinstance(vp, dict) or not vp:
        return
    # Always-visible panel (avoid ``st.expander`` — it collapses on ``st.rerun()`` after Select).
    # Percent in JSON strings is profiling: fraction of *duplicate key groups* where the column
    # takes &gt;1 value (not % of all rows, not statistical variance of numbers).
    _exp = (
        "Percentages in the summaries below (e.g. 100% or 25%–50%) mean: among "
        "duplicate <strong>groups</strong> (rows sharing the same candidate key), what <strong>fraction of "
        "those groups</strong> show this <strong>non-key</strong> column with <strong>more than one value</strong>. "
        "100% means every such group—not “100% of all rows in the file.” Quoted values render as chips."
    )
    inner: list[str] = [
        '<div class="ia-variance-panel">',
        "<h4>Where the ambiguity is</h4>",
        f'<p class="ia-variance-explainer">{_exp}</p>',
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
    after_uc_approve_success: Callable[[], None] | None = None,
) -> None:
    inject_hitl_css()
    _col_l, _ = st.columns([6, 1])
    with _col_l:
        idxs = grain_item_indices(items)
        if not idxs:
            st.warning(
                "This JSON has no **grain** domain items with options — "
                "if this is ``identity_term_hitl.json``, open the **term** gate (artifact type ``term``), "
                "or confirm the path points at ``identity_grain_hitl.json``."
            )
            return

        path_tag = silver_volume_path_session_tag(silver_path)
        psk = f"{sk}-{path_tag}"

        inst_raw = (data.get("institution_id") or "").strip()
        run_total = (
            ia_grain_run_total_items(pending_df, str(onboard_run_id))
            if pending_df is not None
            else None
        )
        nav_key = f"ia-grain-nav-{psk}"
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
            tbl=tbl,
            domain_label="Grain",
            cur=cur,
            n_items=n_items,
            run_total=run_total,
            item_id=item.get("item_id", ""),
        )

        q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
        st.markdown(
            f'<div class="hitl-qpanel">{html.escape(q)}</div>',
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
        sel_key = f"ia-grain-sel-{psk}-{i}"
        init_sel_key(sel_key, item.get("choice"), n_opt)

        json_choice = item.get("choice")
        ia_rec_ix = (
            0 if json_choice is None else max(0, min(int(json_choice) - 1, n_opt - 1))
        )

        render_option_cards(
            options=options,
            sel_key=sel_key,
            ia_rec_ix=ia_rec_ix,
            json_choice=json_choice,
            uc_group_pending=uc_group_pending,
            key_prefix="ia-grain",
            sk=psk,
            file_index=i,
        )

        sel_j = int(st.session_state[sel_key])
        sel_opt = (
            options[sel_j]
            if 0 <= sel_j < len(options) and isinstance(options[sel_j], dict)
            else {}
        )
        reentry_sel = str(sel_opt.get("reentry") or "").lower()
        custom_key = f"ia-grain-custom-{psk}-{i}"
        custom_store_key = f"ia-grain-custom-store-{psk}"
        if custom_store_key not in st.session_state:
            st.session_state[custom_store_key] = {}
        custom_store: dict[int, str] = st.session_state[custom_store_key]
        if i not in custom_store:
            existing = item.get("reviewer_note")
            custom_store[i] = str(existing or "") if existing else ""
        if custom_key not in st.session_state:
            st.session_state[custom_key] = custom_store[i]

        if reentry_sel == "generate_hook":
            if sel_opt.get("resolution") is None:
                st.text_area(
                    "Describe the custom handling you want applied:",
                    key=custom_key,
                    height=120,
                    disabled=not uc_group_pending,
                )
            else:
                st.text_area(
                    "Optional: extra instructions for hook generation (if blank, context + config are used):",
                    key=custom_key,
                    height=120,
                    disabled=not uc_group_pending,
                )

        def _flush_ia_grain_custom_note_to_store() -> None:
            if str(sel_opt.get("reentry") or "").lower() != "generate_hook":
                return
            st_local = st.session_state.setdefault(custom_store_key, {})
            if custom_key in st.session_state:
                st_local[i] = str(st.session_state[custom_key])

        opened_k, all_nav_seen = mark_hitl_nav_visit(
            store_key=f"ia-grain-nav-visit-{psk}",
            silver_path=silver_path,
            cur=cur,
            n_items=n_items,
        )
        approve_blocked = n_items > 1 and not all_nav_seen
        _grain_help_base = (
            "Writes **all** grain ``choice`` values from this screen (and any already saved on "
            "disk) in one file write, then approves the UC ``hitl_reviews`` row when it is pending."
        )
        _grain_help = _grain_help_base
        if approve_blocked:
            _grain_help = (
                f"Open each grain item with Prev/Next first ({opened_k}/{n_items} viewed). "
                + _grain_help_base
            )
        _grain_cap = None
        if n_items > 1:
            _grain_cap = (
                f"This file has **{n_items}** grain item(s) (often one per table). Use **Prev/Next** "
                "to open each one and pick an option — **Approve** saves every table’s choice in one write."
            )
            if approve_blocked:
                _grain_cap += (
                    f" **Approve** is disabled until every item has been opened ({opened_k}/{n_items} so far)."
                )

        render_action_bar(
            nav_key=nav_key,
            cur=cur,
            n_items=n_items,
            sk=psk,
            key_prefix="ia-grain",
            file_index=i,
            include_prev_next=True,
            nav_prev_button_key=None,
            nav_next_button_key=None,
            nav_entity_label="grain",
            primary_button_key=f"ia-grain-save-all-{psk}",
            primary_button_label="Approve",
            primary_help=_grain_help,
            pre_bar_caption=_grain_cap,
            uc_group_pending=uc_group_pending,
            primary_extra_disabled=approve_blocked,
            show_reject_item=True,
            persist_fn=lambda: persist_ia_grain_hitl_from_session(
                silver_path=silver_path,
                sk=sk,
                allow_silver_write=uc_group_pending,
            ),
            reject_fn=lambda: _persist_grain_reject(
                silver_path=silver_path,
                item_index=i,
                onboard_run_id=str(onboard_run_id),
                allow_write=uc_group_pending,
            ),
            after_persist_success=lambda: invalidate_ia_grain_run_cache(
                str(onboard_run_id)
            ),
            approve_fn=approve_uc_if_complete,
            after_uc_approve_success=after_uc_approve_success,
            success_silver_filename="identity_grain_hitl.json",
            before_nav_rerun=_flush_ia_grain_custom_note_to_store,
        )


def _persist_grain_reject(
    *, silver_path: str, item_index: int, onboard_run_id: str, allow_write: bool
) -> None:
    if not allow_write:
        st.error(
            "Cannot write: this UC gate is not pending; silver JSON edits are disabled."
        )
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
