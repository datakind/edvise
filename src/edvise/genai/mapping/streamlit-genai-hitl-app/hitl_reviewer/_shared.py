"""
Shared Streamlit layout for HITL reviewers (IA grain/term, SMA).

CSS uses unified class names: ``hitl-inst``, ``hitl-qpanel``, ``hitl-meta``; options are
``st.button`` with ``hitl-opt-mark`` + scoped CSS; grain table/variance keep ``.ia-`` names.
"""

from __future__ import annotations

import html
from collections.abc import Callable
from typing import Any, Literal

import streamlit as st

from hitl_reviewer.hitl_json_batch_commit import try_approve_uc_after_json_write


def inject_hitl_css() -> None:
    st.markdown(
        """
<style>
/* Constrain review card to readable width */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlock"] {
  max-width: 900px !important;
}

/* Tighten candidate keys table (full rule — font/padding in one place) */
.ia-cand-table { width: 100%; border-collapse: collapse; font-size: 0.82rem; margin: 0.5rem 0 1rem 0; }
.ia-cand-table th, .ia-cand-table td { border: 1px solid rgba(0,0,0,0.08); padding: 0.3rem 0.45rem; vertical-align: top; }
.ia-cand-table th { background: rgba(0,0,0,0.04); text-align: left; }
.ia-notes { max-width: 28rem; white-space: pre-wrap; word-break: break-word; }

/* Tighten variance panel */
.ia-variance-panel { padding: 0.5rem 0.75rem; margin: 0.5rem 0 0.75rem 0; border: 1px solid rgba(0,0,0,0.1);
  border-radius: 10px; background: rgba(0,0,0,0.02);
}
.ia-var-line { font-size: 0.85rem; line-height: 1.45; margin: 0.2rem 0 0.4rem 0; }
.ia-variance-panel h4 { margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600; color: rgba(49, 51, 63, 0.95); }
.ia-var-key { font-weight: 600; font-family: ui-monospace, monospace; }

/* Option card buttons: .hitl-opt-mark in prior block (hidden) + adjacent stButton; avoids action bar */
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button {
  width: 100% !important;
  text-align: left !important;
  white-space: normal !important;
  height: auto !important;
  min-height: unset !important;
  padding: 0.4rem 0.65rem !important;
  border-radius: 6px !important;
  font-size: 0.82rem !important;
  line-height: 1.35 !important;
  margin-bottom: 0.35rem !important;
  box-shadow: none !important;
  display: block !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button p {
  text-align: left !important;
  margin: 0 !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button p:first-child {
  font-weight: 600 !important;
  font-size: 0.85rem !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button p:last-child {
  font-size: 0.78rem !important;
  opacity: 0.75 !important;
  margin-top: 0.15rem !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button[kind="secondary"] {
  border: 1px solid rgba(0,0,0,0.12) !important;
  background: white !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button[kind="primary"] {
  border: 1px solid rgba(99, 102, 241, 0.7) !important;
  background: rgba(99, 102, 241, 0.08) !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button[kind="secondary"]:hover:enabled {
  border-color: rgba(99,102,241,0.5) !important;
  background: rgba(99,102,241,0.04) !important;
}
div[data-testid="stElementContainer"]:has(p.hitl-opt-mark) + div button[kind="primary"]:hover:enabled {
  border-color: rgba(99, 102, 241, 0.85) !important;
  background: rgba(99, 102, 241, 0.12) !important;
}
p.hitl-opt-mark { display: block; height: 0; margin: 0 !important; padding: 0 !important; font-size: 0; line-height: 0; overflow: hidden; }
.hitl-inst { font-size: 1.5rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.25rem 0; letter-spacing: -0.02em; }
.hitl-qpanel { font-size: 1.05rem; line-height: 1.5; font-weight: 500; margin: 0.6rem 0 0.85rem 0;
  padding: 0.75rem 1rem; background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.18);
  border-radius: 8px; overflow-wrap: anywhere; word-break: break-word; }
.hitl-meta { font-size: 0.78rem; color: rgba(49, 51, 63, 0.75); margin-bottom: 0.35rem; }
.hitl-ctx-block { font-size: 0.95rem; line-height: 1.45; color: rgba(49, 51, 63, 0.92); }
/* SMA/IA: wrap long evidence/rationale. Match Streamlit st.text (pre) feel: mono + subtle well,
   but white-space: pre-wrap + overflow-wrap so lines wrap instead of a horizontal scrollbar. */
.hitl-ctx-prose { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.86rem; line-height: 1.45; color: rgba(49, 51, 63, 0.92);
  margin: 0.15rem 0 0.6rem 0; max-width: 100%; box-sizing: border-box;
  padding: 0.5rem 0.65rem; border-radius: 0.25rem; background: rgba(250, 250, 250, 1);
  border: 1px solid rgba(0, 0, 0, 0.08);
  white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }
.hitl-chip-row { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.25rem; }
.hitl-chip { display: inline-block; padding: 0.08rem 0.45rem; border-radius: 999px; font-size: 0.72rem;
  background: rgba(111, 66, 193, 0.12); border: 1px solid rgba(111, 66, 193, 0.28); }
.ia-domain-pill {
  display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.72rem; font-weight: 600;
  background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.35); color: rgb(6, 95, 70);
}
.ia-uni-good { color: rgb(5, 122, 85); font-weight: 600; }
.ia-uni-warn { color: rgb(180, 83, 9); font-weight: 600; }
</style>
""",
        unsafe_allow_html=True,
    )


def render_institution_line(*, inst_raw: str, format_fn: Callable[[str], str]) -> None:
    st.markdown(
        f'<p class="hitl-inst">{html.escape(format_fn(inst_raw))}</p>',
        unsafe_allow_html=True,
    )


def render_sma_status_meta_line(*, prebuilt_line_html: str) -> None:
    """SMA+nav: run wide pending line (``hitl-meta``) — `prebuilt` is pre-escaped / safe inner HTML."""
    st.markdown(
        f'<p class="hitl-meta">{prebuilt_line_html}</p>',
        unsafe_allow_html=True,
    )


def render_hitl_header(
    *,
    inst_raw: str,
    format_fn: Callable[[str], str],
    onboard_run_id: str,
    tbl: str,
    domain_label: str,
    cur: int,
    n_items: int,
    run_total: int | None,
    file_index: int,
    item_id: Any,
    inst_class: str = "hitl-inst",
) -> None:
    st.markdown(
        f'<p class="{inst_class}">{html.escape(format_fn(inst_raw))}</p>',
        unsafe_allow_html=True,
    )
    noun = str(domain_label).strip().lower()
    meta_parts = [
        f'<span class="hitl-meta">Onboard run <code>{html.escape(str(onboard_run_id))}</code></span>',
        f'<span class="hitl-meta"> · Dataset: <strong>{html.escape(tbl)}</strong></span>',
        f'<span class="ia-domain-pill">{html.escape(domain_label)}</span>',
    ]
    if run_total is not None:
        meta_parts.append(
            f'<span class="hitl-meta"> · <strong>{cur + 1} of {n_items}</strong> items in this file '
            f"({int(run_total)} {noun} item(s) on this run)</span>"
        )
    else:
        meta_parts.append(
            f'<span class="hitl-meta"> · <strong>{cur + 1} of {n_items}</strong> items in this file</span>'
        )
    _item_id_esc = html.escape(str(item_id))
    meta_parts.append(
        f'<span class="hitl-meta"> · File index <code>{file_index}</code> · <code>{_item_id_esc}</code></span>'
    )
    st.markdown("<div>" + "".join(meta_parts) + "</div>", unsafe_allow_html=True)


def init_sel_key(sel_key: str, choice: Any, n_opt: int) -> None:
    if sel_key not in st.session_state:
        if choice is None:
            st.session_state[sel_key] = 0
        else:
            try:
                st.session_state[sel_key] = max(0, min(int(choice) - 1, n_opt - 1))
            except (TypeError, ValueError):
                st.session_state[sel_key] = 0


def render_option_cards(
    *,
    options: list[Any],
    sel_key: str,
    ia_rec_ix: int,
    json_choice: Any,
    uc_group_pending: bool,
    key_prefix: str,
    sk: str,
    file_index: int,
    option_label_format: Literal["raw", "numbered"] = "raw",
) -> None:
    st.markdown("**Decision**")
    for j, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        lab_raw = str(opt.get("label") or f"Option {j + 1}")
        if option_label_format == "numbered":
            lab = f"{j + 1}. {lab_raw}"
        else:
            lab = lab_raw
        desc = str(opt.get("description") or "")
        desc_short = desc[:120].rstrip() + "…" if len(desc) > 120 else desc
        selected = int(st.session_state[sel_key]) == j
        is_rec = j == ia_rec_ix
        if is_rec and json_choice is not None:
            badge = " · ✦ Saved in JSON"
        elif is_rec:
            badge = " · ✦ IA recommendation"
        else:
            badge = ""
        btn_label = f"**{lab}**{badge}  \n{desc_short}"
        st.markdown(
            f'<p class="hitl-opt-mark" data-opt="{j}">.</p>',
            unsafe_allow_html=True,
        )
        if st.button(
            btn_label,
            key=f"{key_prefix}-pick-{sk}-{file_index}-{j}",
            type="primary" if selected else "secondary",
            use_container_width=True,
            disabled=not uc_group_pending,
        ):
            st.session_state[sel_key] = j
            st.rerun()


def render_action_bar(
    *,
    nav_key: str,
    cur: int,
    n_items: int,
    sk: str,
    key_prefix: str,
    file_index: int,
    include_prev_next: bool,
    nav_prev_button_key: str | None,
    nav_next_button_key: str | None,
    primary_button_key: str,
    primary_button_label: str,
    primary_help: str,
    pre_bar_caption: str | None,
    uc_group_pending: bool,
    show_reject_item: bool,
    persist_fn: Callable[[], tuple[bool, str]],
    reject_fn: Callable[[], None] | None,
    after_persist_success: Callable[[], None],
    approve_fn: Callable[[], None] | None,
    success_silver_filename: str | None,
) -> None:
    if pre_bar_caption is not None:
        st.caption(pre_bar_caption)
    with st.container(border=True):
        if include_prev_next:
            c_prev, c_next, c_save, c_rej = st.columns([1, 1, 2.6, 1.1], gap="small")
            with c_prev:
                pk = f"{key_prefix}-prev-{sk}" if nav_prev_button_key is None else nav_prev_button_key
                if st.button("◀ Prev", key=pk, use_container_width=True):
                    st.session_state[nav_key] = max(0, cur - 1)
                    st.rerun()
            with c_next:
                nk = (
                    f"{key_prefix}-nxt-{sk}" if nav_next_button_key is None else nav_next_button_key
                )
                if st.button("Next ▶", key=nk, use_container_width=True):
                    st.session_state[nav_key] = min(n_items - 1, cur + 1)
                    st.rerun()
        else:
            c_save, c_rej = st.columns([2.6, 1.1], gap="small")
        with c_save:
            if st.button(
                primary_button_label,
                key=primary_button_key,
                type="primary",
                use_container_width=True,
                disabled=not uc_group_pending,
                help=primary_help,
            ):
                ok, err = persist_fn()
                if not ok:
                    st.error(err)
                else:
                    after_persist_success()
                    ap_ok, ap_err = try_approve_uc_after_json_write(
                        uc_group_pending=uc_group_pending,
                        approve_uc_if_complete=approve_fn,
                    )
                    if not ap_ok:
                        if success_silver_filename is not None:
                            st.warning(
                                "Silver JSON saved, but UC approve failed (fix and retry or use "
                                f"SQL): {ap_err}"
                            )
                        else:
                            st.warning(f"JSON saved, but UC approve failed: {ap_err}")
                    elif success_silver_filename is not None:
                        if uc_group_pending and approve_fn is not None:
                            st.success(
                                f"Saved ``{success_silver_filename}`` and approved the UC row."
                            )
                            st.toast("JSON + UC complete.", icon="✅")
                        elif not uc_group_pending:
                            st.success(
                                f"Saved ``{success_silver_filename}``. UC row was not **pending**, so "
                                "UC approve was skipped."
                            )
                        else:
                            st.success(f"Saved ``{success_silver_filename}``.")
                        st.rerun()
                    else:
                        if uc_group_pending:
                            st.success("Saved manifest JSON and approved the UC row.")
                            st.toast("JSON + UC complete.", icon="✅")
                        else:
                            st.success(
                                "Saved manifest JSON. UC was not pending, so UC approve was skipped."
                            )
                        st.rerun()
        with c_rej:
            if show_reject_item:
                if st.button(
                    "Reject item",
                    key=f"{key_prefix}-reject-{sk}-{file_index}",
                    type="secondary",
                    use_container_width=True,
                    disabled=not uc_group_pending,
                ):
                    if reject_fn is not None:
                        reject_fn()
    if show_reject_item and success_silver_filename is not None:
        if uc_group_pending:
            st.caption("Approve saves your selections and marks this review complete.")
        else:
            st.caption(
                "Read-only: UC gate is not pending; silver JSON cannot be changed from this app."
            )