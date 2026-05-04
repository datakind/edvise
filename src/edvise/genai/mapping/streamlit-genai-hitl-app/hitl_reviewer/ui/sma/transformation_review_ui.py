"""
SMA Step 2b — transformation review HITL (``sma_gate_2_transformation_review``).

Review JSON shape matches :class:`TransformationReviewHITLFile`: top-level ``items`` are
``TransformationHITLItem`` dicts (``flagged_steps``, ``steps``, ``options``, ``choice``, ``status``).
"""

from __future__ import annotations

import html
import json
from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from edvise.utils.institution_naming import format_institution_display_name
from hitl_reviewer.persistence.hitl_json_batch_commit import (
    persist_sma_transformation_review_from_session,
)
from hitl_reviewer.persistence.silver_hitl_paths import silver_volume_path_session_tag
from hitl_reviewer.ui._shared import (
    init_sel_key,
    inject_hitl_css,
    mark_hitl_nav_visit,
    render_action_bar,
    render_institution_line,
    render_option_cards,
    render_sma_status_meta_line,
)
from hitl_reviewer.ui.sma.manifest_review_ui import (
    _sma_wrapped_prose_block,
    invalidate_sma_run_pending_cache,
    render_sma_option_descriptions,
    sma_run_pending_ordered_pairs,
)


PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW: str = "sma_gate_2_transformation_review"
ARTIFACT_TYPES_TRANSFORMATION_REVIEW: tuple[str, ...] = (
    "cohort_transformation_review",
    "course_transformation_review",
)


def is_sma_transformation_review_phase(phase: str, artifact_type: str) -> bool:
    ph = str(phase).strip().lower()
    at = str(artifact_type).strip().lower()
    return ph == PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW and at in ARTIFACT_TYPES_TRANSFORMATION_REVIEW


def _after_transformation_review_persist(*, silver_path: str, onboard_run_id: str) -> None:
    invalidate_sma_run_pending_cache(str(onboard_run_id))


def render_sma_transformation_review_cards(
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
    reject_uc_fn: Callable[[], None] | None = None,
    reject_uc_button_key: str | None = None,
) -> None:
    """HITL UI for ``cohort_transformation_review.json`` / ``course_transformation_review.json``."""
    inject_hitl_css()
    school_raw = (data.get("school_id") or data.get("institution_id") or "").strip()
    render_institution_line(
        inst_raw=school_raw,
        format_fn=format_institution_display_name,
    )
    st.caption(
        f"Domain: **{html.escape(str(data.get('domain') or 'transformation_review'))}** · "
        f"entity: **{html.escape(str(data.get('entity_type') or '?'))}**"
    )
    path_tag = silver_volume_path_session_tag(silver_path)
    psk = f"{sk}-{path_tag}"

    option_item_indices: list[int] = []
    for ix, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        opts = it.get("options")
        if not opts or not isinstance(opts, list) or len(opts) < 1:
            continue
        option_item_indices.append(ix)
        rk = f"sv{psk}item{ix}{it.get('item_id', ix)}"
        if rk not in st.session_state:
            c_raw = it.get("choice")
            if c_raw is None:
                st.session_state[rk] = 0
            else:
                try:
                    jix = int(c_raw)
                except (TypeError, ValueError):
                    jix = 1
                st.session_state[rk] = max(0, min(len(opts) - 1, jix - 1))

    pending_ixs = [
        j
        for j in option_item_indices
        if isinstance(items[j], dict) and items[j].get("choice") is None
    ]
    nav_ixs = pending_ixs if pending_ixs else option_item_indices
    cur_key = f"sma-tr-nav-{psk}"
    pre_bar_caption: str | None = None
    cur_nav = 0
    i = 0
    y_pending = len(pending_ixs)

    if nav_ixs:
        if cur_key not in st.session_state:
            st.session_state[cur_key] = 0
        cur_nav = max(0, min(int(st.session_state[cur_key]), len(nav_ixs) - 1))
        st.session_state[cur_key] = cur_nav
        i = nav_ixs[cur_nav]
        y_pending = len(pending_ixs)
        run_line = ""
        if pending_df is not None and not pending_df.empty:
            ordered = sma_run_pending_ordered_pairs(
                pending_df,
                str(onboard_run_id),
                phase=PHASE_SMA_GATE_2_TRANSFORMATION_REVIEW,
                artifact_types=ARTIFACT_TYPES_TRANSFORMATION_REVIEW,
            )
            y_run = len(ordered)
            pr = (silver_path.strip(), int(i))
            if y_run > 0 and pr in ordered:
                x_run = ordered.index(pr) + 1
                run_line = (
                    f"Onboard run <code>{html.escape(str(onboard_run_id))}</code> — "
                    f"<strong>{x_run} of {y_run} pending</strong>"
                )
            elif y_run > 0:
                run_line = (
                    f"Onboard run <code>{html.escape(str(onboard_run_id))}</code> — "
                    f"{y_run} pending on this run; this item may already be resolved in JSON."
                )
            else:
                run_line = (
                    f"Onboard run <code>{html.escape(str(onboard_run_id))}</code> — "
                    "<strong>0 of 0 pending</strong>"
                )
        elif y_pending > 0:
            x_pf = cur_nav + 1 if pending_ixs and nav_ixs == pending_ixs else 0
            run_line = f"<strong>{x_pf} of {y_pending} pending</strong> (this file)"
        else:
            if uc_group_pending:
                run_line = (
                    "No unresolved items in this file — browsing "
                    f"{len(nav_ixs)} item(s); change a selection and Save to update."
                )
            else:
                run_line = f"No unresolved items — browsing {len(nav_ixs)} item(s)."
        if run_line:
            render_sma_status_meta_line(prebuilt_line_html=run_line)

        item = items[i]
        if not isinstance(item, dict):
            st.error("Invalid HITL item.")
            return

        tf = (str(item.get("target_field") or "")).strip()
        et = (str(item.get("entity_type") or data.get("entity_type") or "")).strip()
        conf = item.get("confidence")
        title = (
            f"Transformation review — `{html.escape(et)}.{html.escape(tf)}`"
            if tf
            else f"Item {i + 1}"
        )
        conf_html = (
            f' · confidence <code>{html.escape(str(conf))}</code>'
            if conf is not None
            else ""
        )
        st.markdown(
            f'<div class="hitl-qpanel">{title}{conf_html}</div>',
            unsafe_allow_html=True,
        )

        notes_parts: list[str] = []
        rn = (str(item.get("reviewer_notes") or "")).strip()
        vn = (str(item.get("validation_notes") or "")).strip()
        if rn:
            notes_parts.append(f"**Reviewer notes (model)**\n{rn}")
        if vn:
            notes_parts.append(f"**Validation notes (model)**\n{vn}")
        if notes_parts:
            with st.expander("Plan notes", expanded=False):
                _sma_wrapped_prose_block("\n\n".join(notes_parts))

        flagged = item.get("flagged_steps")
        if isinstance(flagged, list) and flagged:
            with st.expander("Flagged steps (evidence)", expanded=True):
                st.json(flagged)

        steps_raw = item.get("steps")
        with st.expander("Proposed transformation steps", expanded=False):
            if isinstance(steps_raw, list):
                st.json(steps_raw)
            else:
                st.caption("No steps array on this item.")

        options = item.get("options")
        if not options or not isinstance(options, list):
            options = []
        n = len(options)
        render_sma_option_descriptions(options=options)

        sel_key = f"sv{psk}item{i}{item.get('item_id', i)}"
        init_sel_key(sel_key, item.get("choice"), n)
        json_choice = item.get("choice")
        if json_choice is None:
            ia_rec_ix = 0
        else:
            try:
                jix = int(json_choice)
            except (TypeError, ValueError):
                jix = 1
            ia_rec_ix = max(0, min(jix - 1, n - 1)) if n > 0 else 0
        pre_bar_caption = (
            f"Item {cur_nav + 1} of {len(nav_ixs)} in this file "
            f"({len(option_item_indices)} with options)."
        )
        render_option_cards(
            options=options,
            sel_key=sel_key,
            ia_rec_ix=ia_rec_ix,
            json_choice=json_choice,
            uc_group_pending=uc_group_pending,
            key_prefix="sma-tr",
            sk=psk,
            file_index=int(i),
            option_label_format="numbered",
            recommendation_badge_label="SMA recommendation",
        )

        try:
            sel_j = max(0, min(int(st.session_state.get(sel_key, ia_rec_ix)), n - 1))
        except (TypeError, ValueError):
            sel_j = 0
        cur_opt = (
            options[sel_j]
            if sel_j < len(options) and isinstance(options[sel_j], dict)
            else {}
        )
        oid = str(cur_opt.get("option_id") or "").strip().lower()
        steps_edit_key = f"tr-steps-{psk}-{i}-{item.get('item_id', i)}"
        if steps_edit_key not in st.session_state:
            if isinstance(steps_raw, list):
                st.session_state[steps_edit_key] = json.dumps(
                    steps_raw, indent=2, ensure_ascii=False
                )
            else:
                st.session_state[steps_edit_key] = "[]"

        if oid == "correct" or sel_j == 1:
            st.markdown("**Correct — edit steps (JSON array)**")
            st.caption(
                "Each element must be a valid transformation step object (`function_name`, `column`, …). "
                "Required when **Correct** is selected before Save."
            )
            st.text_area(
                "steps_json",
                height=280,
                key=steps_edit_key,
                disabled=not uc_group_pending,
                label_visibility="collapsed",
            )
    else:
        st.info("No HITL items with `options` in this transformation review JSON.")

    def _persist() -> tuple[bool, str]:
        return persist_sma_transformation_review_from_session(
            silver_path=silver_path,
            sk=psk,
            option_item_indices=option_item_indices,
            allow_silver_write=uc_group_pending,
        )

    sma_opened, sma_all_seen = (1, True)
    sma_approve_blocked = False
    if nav_ixs and len(nav_ixs) > 1:
        sma_opened, sma_all_seen = mark_hitl_nav_visit(
            store_key=f"sma-tr-nav-visit-{psk}",
            silver_path=silver_path,
            cur=cur_nav,
            n_items=len(nav_ixs),
        )
        sma_approve_blocked = not sma_all_seen

    _help_base = (
        "Writes ``choice`` (1-based option index), syncs ``status``, and — when **Correct** is chosen — "
        "writes ``steps`` from the JSON editor. Then approves the UC row when pending."
    )
    _help = _help_base
    if sma_approve_blocked:
        _help = (
            f"Open each item with Prev/Next first ({sma_opened}/{len(nav_ixs)} viewed). " + _help_base
        )

    nav_n = max(1, len(nav_ixs) if nav_ixs else 1)
    render_action_bar(
        nav_key=cur_key,
        cur=cur_nav,
        n_items=nav_n,
        sk=psk,
        key_prefix="sma-tr",
        file_index=int(i) if nav_ixs else 0,
        include_prev_next=bool(nav_ixs),
        nav_prev_button_key=f"prev-tr-{psk}",
        nav_next_button_key=f"nxt-tr-{psk}",
        primary_button_key=f"ssave-tr-{psk}",
        primary_button_label="Save JSON & approve UC",
        primary_help=_help,
        pre_bar_caption=pre_bar_caption,
        uc_group_pending=uc_group_pending,
        primary_extra_disabled=sma_approve_blocked,
        show_reject_item=False,
        persist_fn=_persist,
        reject_fn=None,
        after_persist_success=lambda: _after_transformation_review_persist(
            silver_path=silver_path, onboard_run_id=str(onboard_run_id)
        ),
        approve_fn=approve_uc_if_complete,
        after_uc_approve_success=after_uc_approve_success,
        success_silver_filename=None,
        saved_json_description="transformation review JSON",
        reject_uc_fn=reject_uc_fn,
        reject_uc_button_key=reject_uc_button_key,
    )
