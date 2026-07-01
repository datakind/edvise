"""UC / notebook gate checks for Step 2b transformation HITL JSON (review + hook_required)."""

from __future__ import annotations

from pathlib import Path

from edvise.genai.mapping.shared.hitl import raise_if_hitl_pending
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json

from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    TransformationHITLItem,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas import (
    InstitutionSMATransformationHookHITLItems,
    SMATransformationHookHITLItem,
    TransformationReviewHITLFile,
)


def check_transformation_review_hitl_gate(hitl_path: str | Path) -> None:
    """Raise HITLBlockingError when any item is still pending (no status / choice)."""
    path = Path(hitl_path)
    env = read_pydantic_json(path, TransformationReviewHITLFile)
    if not env.items:
        print(f"✓ No transformation review HITL items — gate clear ({path.name}).")
        return
    unresolved = env.pending
    if not unresolved:
        print(
            f"✓ Transformation review HITL gate clear — {len(env.items)} item(s) in {path.name}."
        )
        return

    def _fmt(it: TransformationHITLItem) -> str:
        return f"[{it.item_id}] {it.entity_type}.{it.target_field}"

    raise_if_hitl_pending(
        unresolved,
        hitl_path=path,
        format_item=_fmt,
        instructions=(
            "  • For each item set ``status`` to approved | corrected | hook_required, **or** set "
            "``choice`` to the 1-based option index (1=approve, 2=correct, 3=hook_required).\n"
            "  • For **correct**, edit ``steps`` on the item before saving.\n"
            "  • Save the JSON, then approve the UC hitl_reviews row."
        ),
    )


def check_transformation_hook_hitl_gate(hitl_path: str | Path) -> None:
    """Raise HITLBlockingError when any item has no ``choice``."""
    path = Path(hitl_path)
    env = read_pydantic_json(path, InstitutionSMATransformationHookHITLItems)
    if not env.items:
        print(f"✓ No transformation hook HITL items — gate clear ({path.name}).")
        return
    if env.is_clear:
        print(
            f"✓ Transformation hook HITL gate clear — {len(env.items)} item(s) in {path.name}."
        )
        return

    def _fmt(it: SMATransformationHookHITLItem) -> str:
        return f"[{it.item_id}] {it.entity_type}.{it.target_field}"

    raise_if_hitl_pending(
        env.pending,
        hitl_path=path,
        format_item=_fmt,
        instructions=(
            "  • Set 'choice' to the 1-based index of your selected option for each item.\n"
            "  • Save the JSON from the Streamlit reviewer, then approve the UC row."
        ),
    )


__all__ = [
    "check_transformation_hook_hitl_gate",
    "check_transformation_review_hitl_gate",
]
