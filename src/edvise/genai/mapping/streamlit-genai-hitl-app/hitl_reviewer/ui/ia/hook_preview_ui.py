"""
IA hook preview JSON (post–LLM hook generation, pre–apply) for ``ia_gate_1_hooks``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from edvise.genai.mapping.identity_agent.hitl.hook_preview import (
    assemble_hook_spec_drafts_as_module_text,
)


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
                            "No non-empty `functions[].draft` strings — showing raw JSON below."
                        )
                        st.json(hs)
                    with st.expander("Raw `hook_spec` JSON", expanded=False):
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
