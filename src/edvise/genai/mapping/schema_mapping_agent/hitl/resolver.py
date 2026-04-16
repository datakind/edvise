"""
Schema Mapping Agent (2a) HITL on-disk helpers.

Gate and (future) apply logic live here; Pydantic models are in :mod:`edvise.genai.mapping.schema_mapping_agent.hitl.schemas`.
IdentityAgent equivalents: :mod:`edvise.genai.mapping.identity_agent.hitl.resolver`.
"""

from __future__ import annotations

from pathlib import Path

from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import load_sma_hitl
from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import SMAHITLItem
from edvise.genai.mapping.shared.hitl import raise_if_hitl_pending


def check_sma_hitl_gate(hitl_path: str | Path) -> None:
    """
    Raises :class:`~edvise.genai.mapping.shared.hitl.HITLBlockingError` if any SMA
    HITL items are still pending (no ``choice``, or ``direct_edit`` without mapping).

    Intended to run before downstream pipeline steps (e.g. Step 2b) on every run;
    there is no optional or execution-mode bypass for this check.

    Prints and returns cleanly when the gate passes. Does not mutate files.
    """
    path = Path(hitl_path)
    envelope = load_sma_hitl(path)

    if not envelope.items:
        print("✓ No SMA HITL items — pipeline gate clear.")
        return

    if envelope.is_clear:
        print(f"✓ SMA HITL gate clear — {len(envelope.items)} item(s) reviewed.")
        return

    def _format_item(i: SMAHITLItem) -> str:
        if i.choice is None:
            return f"[{i.item_id}] {i.target_field} — {i.hitl_question[:80]}..."
        return (
            f"[{i.item_id}] {i.target_field} — direct_edit selected; "
            "populate direct_edit_field_mapping."
        )

    raise_if_hitl_pending(
        envelope.gate_pending,
        hitl_path=path,
        format_item=_format_item,
        instructions=(
            "  • Set 'choice' to the 1-based index of your selected option "
            "(1 … number of options), or populate direct_edit_field_mapping "
            "if you chose direct_edit.\n"
            "  • Re-run this cell."
        ),
    )


__all__ = ["check_sma_hitl_gate"]
