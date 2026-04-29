"""
IA hook preview JSON (post–LLM hook generation, pre–apply) for ``ia_gate_1_hooks``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st


def assemble_hook_spec_drafts_as_module_text(hook_spec: dict[str, Any]) -> str:
    """
    Concatenate ``functions[].draft`` in hook_spec JSON (same logic as
    ``edvise.genai.mapping.identity_agent.hitl.hook_preview``).

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


def is_ia_hook_preview_phase(phase: str, artifact_type: str) -> bool:
    ph = str(phase).strip().lower()
    at = str(artifact_type).strip().lower()
    if ph != "ia_gate_1_hooks":
        return False
    return at in ("grain_hook_preview", "term_hook_preview")


def render_ia_hook_preview_cards(
    *,
    data: dict[str, Any],
    silver_path: str,
    sk: str,
    uc_group_pending: bool,
    approve_uc_if_complete: Callable[[], None],
    after_uc_approve_success: Callable[[], None] | None,
) -> None:
    """Display ``specs[]`` from hook preview JSON and UC approve (reject is on the parent bar)."""
    specs = data.get("specs")
    st.subheader("Hook preview")
    st.caption(
        f"institution `{data.get('institution_id', '')}` · domain `{data.get('domain', '')}`"
    )
    st.caption(silver_path)

    if not isinstance(specs, list) or not specs:
        st.info("No generated hook specs in this file (`specs` empty). Approve to continue.")
    else:
        for i, row in enumerate(specs):
            if not isinstance(row, dict):
                continue
            item_id = str(row.get("item_id") or f"spec_{i}")
            hs = row.get("hook_spec")
            title = f"`{item_id}`"
            with st.expander(title, expanded=(i == 0)):
                rc = row.get("review_context") if isinstance(row.get("review_context"), dict) else None
                if rc:
                    st.markdown("**Review context** (HITL item + config slice used for hook generation)")
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
                        st.json(cs)
                    st.divider()
                if isinstance(hs, dict):
                    rel_file = (hs.get("file") or "").strip()
                    if rel_file:
                        st.caption(f"Materializes to **`{rel_file}`** (under the volume root)")
                    module_text = assemble_hook_spec_drafts_as_module_text(hs)
                    if module_text:
                        st.markdown("**Module preview** (concatenated `draft` defs — same layout as the `.py` body)")
                        st.code(module_text, language="python", line_numbers=True)
                    else:
                        st.warning(
                            "No non-empty `functions[].draft` strings — enable raw JSON below."
                        )
                    # Streamlit forbids expander-inside-expander; use a checkbox instead of a nested expander.
                    show_raw = st.checkbox(
                        "Show raw `hook_spec` JSON",
                        key=f"hk-raw-{sk}-{i}",
                        value=not bool(module_text),
                    )
                    if show_raw:
                        st.json(hs)
                else:
                    st.write(row)

    if st.button(
        "Approve UC (hook preview)",
        key=f"hk-apr-{sk}",
        type="primary",
        disabled=not uc_group_pending,
    ):
        approve_uc_if_complete()
        if after_uc_approve_success is not None:
            after_uc_approve_success()
        st.success("Hook preview approved.")
        st.rerun()
