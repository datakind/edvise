"""
SMA (Schema Mapping Agent) HITL reviewer (Streamlit) — manifest JSON with option selection.

Renders the cohort/course manifest HITL editor. Shared layout uses :mod:`hitl_reviewer._shared`.
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
    render_institution_line,
    render_option_cards,
    render_sma_status_meta_line,
)
from hitl_reviewer.hitl_json_batch_commit import persist_hitl_choice_radios_from_session
from hitl_reviewer.sma.enriched_schema_contract import (
    enriched_schema_contract_path_from_manifest,
    extract_column_panel_fields,
    load_json_object_from_text,
)
from hitl_reviewer.unity_volume_files import read_unity_file_text


def is_sma_phase(phase: str, artifact_type: str) -> bool:
    return str(phase).strip().lower() == "sma_gate_1" and str(artifact_type).strip().lower() in (
        "cohort_manifest",
        "course_manifest",
    )


def _sma_wrapped_prose_block(text: str) -> None:
    """Render multi-line text with wrapping (``st.text`` uses ``<pre>`` and overflows horizontally)."""
    body = str(text or "")
    if not body:
        return
    st.markdown(
        f'<div class="hitl-ctx-prose">{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def invalidate_sma_run_pending_cache(onboard_run_id: str) -> None:
    rid = str(onboard_run_id).strip()
    for suffix in ("total", "order"):
        k = f"sma-run-pending-{suffix}-{rid}"
        if k in st.session_state:
            del st.session_state[k]


def sma_run_pending_ordered_pairs(
    pending_df: pd.DataFrame, onboard_run_id: str
) -> list[tuple[str, int]]:
    """
    Pending SMA HITL items (``choice`` unset, has ``options``) across manifests for the run.

    Order: ``cohort_manifest`` artifact rows first, then ``course_manifest``, each file's
    ``items`` in list order. Used for run-wide ``X of Y pending``.
    """
    rid = str(onboard_run_id).strip()
    cache_key = f"sma-run-pending-order-{rid}"
    if cache_key in st.session_state:
        return list(st.session_state[cache_key])
    sub = pending_df[
        (pending_df["onboard_run_id"].astype(str) == rid)
        & (pending_df["phase"].astype(str).str.lower() == "sma_gate_1")
        & (
            pending_df["artifact_type"]
            .astype(str)
            .str.lower()
            .isin(["cohort_manifest", "course_manifest"])
        )
    ]
    ordered: list[tuple[str, int]] = []
    for at in ("cohort_manifest", "course_manifest"):
        rows = sub[sub["artifact_type"].astype(str).str.lower() == at]
        if rows.empty:
            continue
        path = str(rows.iloc[0]["artifact_path"]).strip()
        if not path:
            continue
        try:
            raw = read_unity_file_text(path)
            data = json.loads(raw)
        except Exception:
            continue
        items = data.get("items")
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            opts = item.get("options")
            if not opts or not isinstance(opts, list) or len(opts) < 1:
                continue
            if item.get("choice") is None:
                ordered.append((path, idx))
    st.session_state[cache_key] = ordered
    st.session_state[f"sma-run-pending-total-{rid}"] = len(ordered)
    return ordered


def render_sma_review_context(*, item: dict) -> None:
    """``hitl_context``, ``validation_errors``, and ``current_field_mapping`` rationale/notes."""
    ctx = (item.get("hitl_context") or "").strip()
    raw_errs = item.get("validation_errors")
    err_lines: list[str] = []
    if isinstance(raw_errs, list):
        err_lines = [str(e).strip() for e in raw_errs if str(e).strip()]

    cfm = item.get("current_field_mapping")
    rationale = ""
    val_notes = ""
    if isinstance(cfm, dict):
        rationale = (str(cfm.get("rationale") or "")).strip()
        val_notes = (str(cfm.get("validation_notes") or "")).strip()

    if not ctx and not err_lines and not rationale and not val_notes:
        return
    with st.expander("Review context", expanded=True):
        if ctx:
            st.markdown(
                '<p class="hitl-ctx-block"><strong>Evidence</strong></p>', unsafe_allow_html=True
            )
            _sma_wrapped_prose_block(ctx)
        if err_lines:
            st.markdown(
                '<p class="hitl-ctx-block"><strong>Validation issues</strong></p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<ul class="hitl-ctx-block">'
                + "".join(f"<li>{html.escape(e)}</li>" for e in err_lines)
                + "</ul>",
                unsafe_allow_html=True,
            )
        if rationale:
            st.markdown(
                '<p class="hitl-ctx-block"><strong>Model rationale</strong> (current mapping)</p>',
                unsafe_allow_html=True,
            )
            _sma_wrapped_prose_block(rationale)
        if val_notes:
            st.markdown(
                '<p class="hitl-ctx-block"><strong>Predicted validation notes</strong></p>',
                unsafe_allow_html=True,
            )
            _sma_wrapped_prose_block(val_notes)


def render_sma_option_descriptions(*, options: list) -> None:
    """Per-option ``description`` from the HITL JSON (read before choosing)."""
    entries: list[tuple[int, str, str]] = []
    for j, opt in enumerate(options):
        if not isinstance(opt, dict):
            continue
        lab = (str(opt.get("label") or f"Option {j + 1}")).strip()
        desc = (str(opt.get("description") or "")).strip()
        entries.append((j + 1, lab, desc))
    if not entries:
        return
    has_any_desc = any(d for _, _, d in entries)
    if not has_any_desc:
        return
    with st.expander("What each option means", expanded=True):
        for num, lab, desc in entries:
            st.markdown(
                f'<p class="hitl-ctx-block"><strong>{num}. {html.escape(lab)}</strong></p>',
                unsafe_allow_html=True,
            )
            if desc:
                _sma_wrapped_prose_block(desc)


def render_sma_source_column_panel(
    *,
    item: dict,
    enriched_contract_path: str | None,
) -> None:
    if not enriched_contract_path:
        st.caption(":gray[Column details unavailable]")
        return
    fm = item.get("current_field_mapping")
    if not isinstance(fm, dict):
        st.caption(":gray[Column details unavailable]")
        return
    source_column = fm.get("source_column")
    source_table = fm.get("source_table")
    if not source_table or not isinstance(source_table, str) or not str(source_table).strip():
        st.caption(":gray[Column details unavailable]")
        return
    cache_key = f"esc-json-{enriched_contract_path}"
    if cache_key not in st.session_state:
        try:
            raw_c = read_unity_file_text(enriched_contract_path)
            st.session_state[cache_key] = {"ok": True, "obj": load_json_object_from_text(raw_c)}
        except Exception as ex:  # noqa: BLE001
            st.session_state[cache_key] = {"ok": False, "err": str(ex)}
    cached = st.session_state[cache_key]
    if not cached.get("ok"):
        st.caption(":gray[Column details unavailable]")
        return
    panel = extract_column_panel_fields(
        cached["obj"],
        dataset_name=str(source_table).strip(),
        source_column=str(source_column).strip() if source_column else None,
    )
    if panel is None:
        st.caption(":gray[Column details unavailable]")
        return
    exp = st.expander("Source column details", expanded=True)
    with exp:
        null_pct = panel.get("null_percentage")
        n_null = panel.get("null_count")
        try:
            pct_s = f"{float(null_pct):.2f}" if null_pct is not None else "—"
        except (TypeError, ValueError):
            pct_s = str(null_pct) if null_pct is not None else "—"
        nrows = n_null if n_null is not None else "—"
        uniq = panel.get("unique_count")
        uniq_s = str(uniq) if uniq is not None else "—"
        inst_tok = panel.get("inst_null_tokens") or []
        ds_tok = panel.get("dataset_null_tokens") or []
        tok_note = ""
        merged = [t for t in (inst_tok + ds_tok) if t]
        if merged:
            shown = ", ".join(repr(t) for t in merged[:12])
            tok_note = f" Institution null tokens: {shown}."

        rows = [
            {"Field": "Original name", "Value": str(panel.get("original_name", "—"))},
            {"Field": "Type", "Value": str(panel.get("dtype", "—"))},
            {
                "Field": "Null rate",
                "Value": f"{pct_s}% ({nrows} rows){tok_note}",
            },
            {"Field": "Unique values", "Value": uniq_s},
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        label = "Unique values" if panel.get("chip_mode") == "unique" else "Sample values"
        st.caption(label)
        chips = panel.get("chip_values") or []
        mode = panel.get("chip_mode", "sample")
        if chips:
            esc = "".join(
                f'<span class="hitl-chip">{html.escape(str(v))[:200]}</span>'
                for v in chips[:80]
            )
            st.markdown(
                f'<div class="hitl-chip-row" title="{html.escape(str(mode))}">{esc}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("—")


def _default_choice_index(item: dict, n_opts: int) -> int:
    c_raw = item.get("choice")
    if c_raw is None:
        return 0
    try:
        c_int = int(c_raw)
    except (TypeError, ValueError):
        c_int = 1
    return max(0, min(n_opts - 1, c_int - 1))


def _after_sma_persist(
    *, silver_path: str, onboard_run_id: str
) -> None:
    invalidate_sma_run_pending_cache(str(onboard_run_id))
    esc_path = enriched_schema_contract_path_from_manifest(silver_path, str(onboard_run_id))
    if esc_path:
        ck = f"esc-json-{esc_path}"
        if ck in st.session_state:
            del st.session_state[ck]


def render_sma_hitl_cards(
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
    """Full SMA manifest HITL UI, including Save + optional UC approve (``ssave{sk}``)."""
    inject_hitl_css()
    school_raw = (data.get("school_id") or data.get("institution_id") or "").strip()
    render_institution_line(
        inst_raw=school_raw,
        format_fn=format_institution_display_name,
    )

    option_item_indices: list[int] = []
    for ix, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        opts = it.get("options")
        if not opts or not isinstance(opts, list) or len(opts) < 1:
            continue
        option_item_indices.append(ix)
        rk = f"sv{sk}item{ix}{it.get('item_id', ix)}"
        if rk not in st.session_state:
            st.session_state[rk] = _default_choice_index(it, len(opts))

    pending_ixs = [
        j
        for j in option_item_indices
        if isinstance(items[j], dict) and items[j].get("choice") is None
    ]
    nav_ixs = pending_ixs if pending_ixs else option_item_indices
    cur_key = f"sma-nav-{sk}"
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
            ordered = sma_run_pending_ordered_pairs(pending_df, str(onboard_run_id))
            y_run = len(ordered)
            pr = (silver_path.strip(), int(i))
            if y_run > 0 and pr in ordered:
                x_run = ordered.index(pr) + 1
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    f"<strong>{x_run} of {y_run} pending</strong>"
                )
            elif y_run > 0:
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    f"{y_run} pending on this run; this item is already resolved in JSON "
                    "(browse or change a choice and Save)."
                )
            else:
                run_line = (
                    f'Onboard run <code>{html.escape(str(onboard_run_id))}</code> — '
                    "<strong>0 of 0 pending</strong>"
                )
        elif y_pending > 0:
            x_pf = cur_nav + 1 if pending_ixs and nav_ixs == pending_ixs else 0
            run_line = f"<strong>{x_pf} of {y_pending} pending</strong> (this manifest)"
        else:
            if uc_group_pending:
                run_line = (
                    "No unresolved items in this file — browsing "
                    f"{len(nav_ixs)} item(s) with options; change a radio and Save to update."
                )
            else:
                run_line = (
                    f"No unresolved items — browsing {len(nav_ixs)} item(s). "
                    "UC gate is finalized; this app does not allow JSON changes."
                )
        if run_line:
            render_sma_status_meta_line(prebuilt_line_html=run_line)

        item = items[i]
        if not isinstance(item, dict):
            st.error("Invalid HITL item.")
            return
        q = (item.get("hitl_question") or "").strip() or f"Item {i + 1}"
        st.markdown(
            f'<div class="hitl-qpanel">{html.escape(q)}</div>',
            unsafe_allow_html=True,
        )
        options = item.get("options")
        if not options or not isinstance(options, list):
            options = []
        n = len(options)
        render_sma_review_context(item=item)
        render_sma_option_descriptions(options=options)
        enriched = enriched_schema_contract_path_from_manifest(silver_path, str(onboard_run_id))
        render_sma_source_column_panel(
            item=item,
            enriched_contract_path=enriched,
        )
        sel_key = f"sv{sk}item{i}{item.get('item_id', i)}"
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
            f"Item {cur_nav + 1} of {len(nav_ixs)} "
            f"with options in this JSON ({len(option_item_indices)} total)."
        )
        render_option_cards(
            options=options,
            sel_key=sel_key,
            ia_rec_ix=ia_rec_ix,
            json_choice=json_choice,
            uc_group_pending=uc_group_pending,
            key_prefix="sma",
            sk=sk,
            file_index=int(i),
            option_label_format="numbered",
        )
    else:
        st.info("No HITL items with `options` in this JSON.")

    def _persist() -> tuple[bool, str]:
        return persist_hitl_choice_radios_from_session(
            silver_path=silver_path,
            sk=sk,
            option_item_indices=option_item_indices,
            default_choice_index=_default_choice_index,
            allow_silver_write=uc_group_pending,
        )

    nav_n = max(1, len(nav_ixs) if nav_ixs else 1)
    render_action_bar(
        nav_key=cur_key,
        cur=cur_nav,
        n_items=nav_n,
        sk=sk,
        key_prefix="sma",
        file_index=int(i) if nav_ixs else 0,
        include_prev_next=bool(nav_ixs),
        nav_prev_button_key=f"prev-{sk}",
        nav_next_button_key=f"nxt-{sk}",
        primary_button_key=f"ssave{sk}",
        primary_button_label="Save JSON & approve UC",
        primary_help=(
            "Writes every ``choice`` from the radios into this manifest, then approves the UC row "
            "when pending."
        ),
        pre_bar_caption=pre_bar_caption,
        uc_group_pending=uc_group_pending,
        show_reject_item=False,
        persist_fn=_persist,
        reject_fn=None,
        after_persist_success=lambda: _after_sma_persist(
            silver_path=silver_path, onboard_run_id=str(onboard_run_id)
        ),
        approve_fn=approve_uc_if_complete,
        success_silver_filename=None,
    )
