"""
Hook preview JSON (post–LLM hook generation, pre–materialize).

- ``ia_gate_1_hooks`` — IdentityAgent grain/term HookSpecs
- ``sma_gate_2_hook_preview`` — Schema Mapping Agent transform HookSpecs (cohort/course)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from hitl_reviewer.utils.institution_naming import format_institution_display_name
from hitl_reviewer.persistence.silver_hitl_paths import silver_volume_path_session_tag
from hitl_reviewer.ui._shared import (
    HITL_UC_GATE_SPINNER_LABEL,
    HITL_FLASH_HINT_AFTER_UC,
    apply_hitl_nav_delta,
    inject_hitl_css,
    render_hitl_header,
    set_hitl_flash_banner,
)


def assemble_hook_spec_drafts_as_module_text(hook_spec: dict[str, Any]) -> str:
    """
    Concatenate ``functions[].draft`` in hook_spec JSON (same logic as
    ``edvise.genai.mapping.identity_agent.hitl.hook_generation``).

    Implemented here so the HITL app does not depend on that submodule being present in
    whichever ``edvise`` wheel was bundled for Databricks (avoids stale-wheel import errors).
    """
    functions = hook_spec.get("functions")
    if not isinstance(functions, list):
        return ""
    parts: list[str] = []
    for fn in functions:
        if not isinstance(fn, dict):
            continue
        draft = (fn.get("draft") or "").strip()
        if draft:
            parts.append(draft)
    return "\n\n".join(parts).strip()


def _domain_label_for_hook_preview(domain: Any) -> str:
    d = str(domain or "").strip().lower()
    return {
        "identity_grain": "Grain",
        "identity_term": "Term",
        "schema_mapping_transform_cohort": "Cohort transform",
        "schema_mapping_transform_course": "Course transform",
    }.get(d, str(domain or "").strip() or "Hook")


def _render_hook_preview_spec_body(
    *, row: dict[str, Any], spec_index: int, key_prefix: str
) -> None:
    """One spec row: review context + module / raw JSON (widget keys include ``spec_index``)."""
    hs = row.get("hook_spec")
    rc = (
        row.get("review_context")
        if isinstance(row.get("review_context"), dict)
        else None
    )
    if rc:
        st.markdown(
            "**Review context** (HITL item + config slice used for hook generation)"
        )
        q = (rc.get("hitl_question") or "").strip()
        if q:
            st.markdown(q)
        tbl = (rc.get("table") or "").strip()
        if tbl:
            st.caption(f"table `{tbl}`")
        hg = rc.get("hook_group_id")
        hgt = rc.get("hook_group_tables")
        if hg or (isinstance(hgt, list) and hgt):
            st.caption(
                f"hook group `{hg}` · tables {hgt if isinstance(hgt, list) else []}"
            )
        note = rc.get("reviewer_note")
        if isinstance(note, str) and note.strip():
            st.markdown("Reviewer note:")
            st.markdown(note.strip())
        hx = rc.get("hitl_context")
        if hx is not None:
            st.markdown("HITL evidence:")
            if isinstance(hx, str):
                st.markdown(hx)
            else:
                st.json(hx)
        tgt = rc.get("target")
        if isinstance(tgt, dict) and tgt:
            st.caption(
                f"target · `{tgt.get('config', '')}.{tgt.get('field', '')}` "
                f"on `{tgt.get('table', '')}`"
            )
        cs = rc.get("config_snippet")
        if isinstance(cs, dict) and cs:
            st.markdown("**Config snippet** (`grain_contract` or `term_config`):")
            # Nested ``st.expander`` would nest poorly; gate JSON like raw hook_spec.
            show_config_snippet = st.checkbox(
                "Show config snippet JSON",
                key=f"hk-cs-{key_prefix}-{spec_index}",
                value=False,
            )
            if show_config_snippet:
                st.json(cs)
        et = (str(rc.get("entity_type") or "")).strip()
        tf = (str(rc.get("target_field") or "")).strip()
        if et and tf:
            st.caption(f"SMA · entity `{et}` · target_field `{tf}`")
        ftp = rc.get("field_transformation_plan")
        if isinstance(ftp, dict) and ftp:
            st.markdown("**Field transformation plan** (Step 2b / `hook_required`):")
            st.json(ftp)
        mm = rc.get("manifest_mapping_record")
        if isinstance(mm, dict) and mm:
            st.markdown("**Manifest mapping record**:")
            st.json(mm)
        st.divider()
    if isinstance(hs, dict):
        rel_file = (hs.get("file") or "").strip()
        if rel_file:
            st.caption(f"Materializes to **`{rel_file}`** (under the volume root)")
        module_text = assemble_hook_spec_drafts_as_module_text(hs)
        if module_text:
            st.markdown(
                "**Module preview** (concatenated `draft` defs — same layout as the `.py` body)"
            )
            st.code(module_text, language="python", line_numbers=True)
        else:
            st.warning(
                "No non-empty `functions[].draft` strings — enable raw JSON below."
            )
        show_raw = st.checkbox(
            "Show raw `hook_spec` JSON",
            key=f"hk-raw-{key_prefix}-{spec_index}",
            value=not bool(module_text),
        )
        if show_raw:
            st.json(hs)
    else:
        st.write(row)


def is_ia_hook_preview_phase(phase: str, artifact_type: str) -> bool:
    ph = str(phase).strip().lower()
    at = str(artifact_type).strip().lower()
    if ph != "ia_gate_1_hooks":
        return False
    # Canonical names plus legacy shorthand (older rows / manual UC registrations).
    return at in (
        "grain_hook_preview",
        "term_hook_preview",
        "grain_hook",
        "term_hook",
    )


def is_sma_transform_hook_preview_phase(phase: str, artifact_type: str) -> bool:
    ph = str(phase).strip().lower()
    at = str(artifact_type).strip().lower()
    if ph != "sma_gate_2_hook_preview":
        return False
    return at in (
        "cohort_transformation_hook_preview",
        "course_transformation_hook_preview",
    )


def render_ia_hook_preview_cards(
    *,
    data: dict[str, Any],
    silver_path: str,
    sk: str,
    uc_group_pending: bool,
    approve_uc_if_complete: Callable[[], None],
    after_uc_approve_success: Callable[[], None] | None,
    reject_uc_fn: Callable[[], None] | None = None,
    reject_uc_button_key: str | None = None,
) -> None:
    """Display ``specs[]`` from hook preview JSON and UC approve / reject."""
    specs = data.get("specs")
    inject_hitl_css()
    st.subheader("Hook preview")
    st.caption(f"domain `{data.get('domain', '')}`")
    st.caption(silver_path)

    if not isinstance(specs, list) or not specs:
        st.info(
            "No generated hook specs in this file (`specs` empty). Approve to continue."
        )
    else:
        spec_ixs = [j for j, row in enumerate(specs) if isinstance(row, dict)]
        if not spec_ixs:
            st.warning("No valid spec objects in ``specs`` (expected dict rows).")
        else:
            path_tag = silver_volume_path_session_tag(silver_path)
            psk = f"{sk}-{path_tag}"
            nav_key = f"hk-prev-nav-{psk}"
            if nav_key not in st.session_state:
                st.session_state[nav_key] = 0
            n_items = len(spec_ixs)
            cur = max(0, min(int(st.session_state[nav_key]), n_items - 1))
            st.session_state[nav_key] = cur
            i = spec_ixs[cur]
            row = specs[i]
            if not isinstance(row, dict):
                st.error("Invalid spec row.")
            else:
                inst_raw = (data.get("institution_id") or "").strip()
                rc = (
                    row.get("review_context")
                    if isinstance(row.get("review_context"), dict)
                    else None
                )
                tbl_raw = (rc.get("table") or "").strip() if rc else ""
                tbl_display = (
                    tbl_raw.replace("_", " ").strip().title() if tbl_raw else "—"
                )
                domain_label = _domain_label_for_hook_preview(data.get("domain"))
                render_hitl_header(
                    inst_raw=inst_raw,
                    format_fn=format_institution_display_name,
                    tbl=tbl_display,
                    domain_label=domain_label,
                    cur=cur,
                    n_items=n_items,
                    run_total=None,
                    item_id=row.get("item_id", ""),
                )
                if n_items > 1:
                    c_prev, c_next, _nav_pad = st.columns([1, 1, 3], gap="small")
                    with c_prev:
                        st.button(
                            "◀ Prev",
                            key=f"hk-prev-{psk}",
                            use_container_width=True,
                            disabled=cur <= 0,
                            help=(
                                "First generated hook in this file."
                                if cur <= 0
                                else "Previous hook spec in this JSON file."
                            ),
                            on_click=apply_hitl_nav_delta,
                            kwargs={
                                "nav_key": nav_key,
                                "n_items": n_items,
                                "delta": -1,
                                "before_nav_rerun": None,
                            },
                        )
                    with c_next:
                        st.button(
                            "Next ▶",
                            key=f"hk-nxt-{psk}",
                            use_container_width=True,
                            disabled=cur >= n_items - 1,
                            help=(
                                "Last generated hook in this file."
                                if cur >= n_items - 1
                                else "Next hook spec in this JSON file."
                            ),
                            on_click=apply_hitl_nav_delta,
                            kwargs={
                                "nav_key": nav_key,
                                "n_items": n_items,
                                "delta": 1,
                                "before_nav_rerun": None,
                            },
                        )
                with st.container(border=True):
                    _render_hook_preview_spec_body(
                        row=row, spec_index=i, key_prefix=psk
                    )

    c_apr, c_rej = st.columns([1.4, 1], gap="small")
    with c_apr:
        if st.button(
            "Approve UC (hook preview)",
            key=f"hk-apr-{sk}",
            type="primary",
            use_container_width=True,
            disabled=not uc_group_pending,
        ):
            try:
                with st.spinner(HITL_UC_GATE_SPINNER_LABEL):
                    approve_uc_if_complete()
                    if after_uc_approve_success is not None:
                        after_uc_approve_success()
                st.success("Hook preview approved.")
                set_hitl_flash_banner(
                    "success",
                    "Hook preview approved. " + HITL_FLASH_HINT_AFTER_UC,
                )
                st.rerun()
            except Exception as ex:  # noqa: BLE001
                st.error(str(ex))
    with c_rej:
        if (
            reject_uc_fn is not None
            and (reject_uc_button_key or "").strip()
            and uc_group_pending
        ):
            if st.button(
                "Reject gate",
                key=str(reject_uc_button_key).strip(),
                type="secondary",
                use_container_width=True,
                disabled=not uc_group_pending,
                help="Skips silver edits; updates ``hitl_reviews`` only.",
            ):
                try:
                    with st.spinner(HITL_UC_GATE_SPINNER_LABEL):
                        reject_uc_fn()
                except Exception as ex:  # noqa: BLE001
                    st.error(str(ex))
    if reject_uc_fn is not None and uc_group_pending:
        st.caption(
            "**Approve UC** updates ``hitl_reviews`` only (no silver JSON write from this screen). "
            "**Reject gate** marks the row rejected there as well."
        )
