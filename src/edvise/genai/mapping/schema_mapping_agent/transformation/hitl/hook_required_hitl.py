"""
Step 2b — ``hook_required`` transformation plans: build review JSON, write, merge resolutions.

Gate checks live in :mod:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates`.
Envelope types live in :mod:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates import (
    check_transformation_hook_hitl_gate,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas import (
    InstitutionSMATransformationHookHITLItems,
    SMATransformationHookHITLItem,
    SMATransformationHookHITLOption,
    SMATransformationHookResolution,
    default_transformation_hook_hitl_options,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json

logger = logging.getLogger(__name__)


def iter_hook_required_plans(
    transformation_data: dict[str, Any],
    entity_type: Literal["cohort", "course"],
) -> list[dict[str, Any]]:
    """
    Plan dicts under ``transformation_maps[entity_type]`` with truthy ``hook_required``
    and a non-empty ``target_field``.
    """
    tmaps = transformation_data.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return []
    plans = _plans_from_entity_section(tmaps.get(entity_type))
    if not plans:
        return []
    out: list[dict[str, Any]] = []
    for plan in plans:
        if not plan.get("hook_required"):
            continue
        tf = str(plan.get("target_field") or "").strip()
        if not tf:
            logger.warning(
                "Skipping hook_required plan without target_field in %s transformation map",
                entity_type,
            )
            continue
        out.append(plan)
    return out


def _plans_from_entity_section(entity_blob: object) -> list[dict[str, Any]] | None:
    if not isinstance(entity_blob, dict):
        return None
    raw_plans = entity_blob.get("plans")
    if not isinstance(raw_plans, list):
        return None
    out: list[dict[str, Any]] = []
    for p in raw_plans:
        if isinstance(p, dict):
            out.append(p)
    return out


def build_transformation_hook_hitl_envelope_for_entity(
    transformation_data: dict[str, Any],
    *,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    options: list[SMATransformationHookHITLOption] | None = None,
) -> InstitutionSMATransformationHookHITLItems:
    """
    Scan ``transformation_data['transformation_maps'][entity]`` for plans with truthy
    ``hook_required`` and build a review envelope.
    """
    opts = options or default_transformation_hook_hitl_options()
    tmaps = transformation_data.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return InstitutionSMATransformationHookHITLItems(
            institution_id=institution_id,
            entity_type=entity_type,
            items=[],
        )
    entity_blob = tmaps.get(entity_type)
    plans = _plans_from_entity_section(entity_blob)
    if not plans:
        return InstitutionSMATransformationHookHITLItems(
            institution_id=institution_id,
            entity_type=entity_type,
            items=[],
        )

    items: list[SMATransformationHookHITLItem] = []
    for plan in plans:
        if not plan.get("hook_required"):
            continue
        tf = str(plan.get("target_field") or "").strip()
        if not tf:
            logger.warning(
                "Skipping hook_required plan without target_field in %s transformation map",
                entity_type,
            )
            continue
        slug = tf.lower().replace(" ", "_")
        item_id = f"{institution_id}_{entity_type}_{slug}_hook_required"
        ctx_parts: list[str] = []
        for key in ("reviewer_notes", "validation_notes"):
            v = plan.get(key)
            if isinstance(v, str) and v.strip():
                ctx_parts.append(f"{key}: {v.strip()}")
        hitl_context = "\n\n".join(ctx_parts) if ctx_parts else None
        q = (
            f"Step 2b set **hook_required** on `{tf}` ({entity_type}). "
            "Choose whether to proceed with the partial utility chain or defer the field."
        )
        items.append(
            SMATransformationHookHITLItem(
                item_id=item_id,
                institution_id=institution_id,
                entity_type=entity_type,
                target_field=tf,
                hitl_question=q,
                hitl_context=hitl_context,
                plan_snapshot=dict(plan),
                current_field_mapping={"target_field": tf},
                options=[o.model_copy(deep=True) for o in opts],
                choice=None,
            )
        )

    return InstitutionSMATransformationHookHITLItems(
        institution_id=institution_id,
        entity_type=entity_type,
        items=items,
    )


def write_transformation_hook_hitl_envelope(
    path: str | Path, env: InstitutionSMATransformationHookHITLItems
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(env.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _find_plan_index(plans: list[dict[str, Any]], target_field: str) -> int | None:
    for i, p in enumerate(plans):
        if str(p.get("target_field") or "").strip() == target_field:
            return i
    return None


def apply_transformation_hook_hitl_resolutions(
    transformation_data: dict[str, Any],
    *,
    cohort_hitl_path: str | Path | None,
    course_hitl_path: str | Path | None,
) -> dict[str, Any]:
    """
    Load reviewer-resolved HITL JSON (when present) and patch matching plans in a deep copy of
    ``transformation_data``.
    """
    out: dict[str, Any] = json.loads(json.dumps(transformation_data))
    tmaps = out.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return out

    def apply_file(entity_type: Literal["cohort", "course"], path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        env = read_pydantic_json(path, InstitutionSMATransformationHookHITLItems)
        entity_blob = tmaps.get(entity_type)
        plans = _plans_from_entity_section(entity_blob)
        if not plans:
            return
        for item in env.items:
            sel = item.selected_option()
            if sel is None:
                continue
            res = sel.resolution
            ix = _find_plan_index(plans, item.target_field)
            if ix is None:
                logger.warning(
                    "HITL item %s targets unknown field %s — skipping",
                    item.item_id,
                    item.target_field,
                )
                continue
            plan = plans[ix]
            if res.clear_hook_required:
                plan["hook_required"] = False
            if res.replace_steps:
                plan["steps"] = list(res.steps if res.steps is not None else [])
            if res.reviewer_notes is not None:
                plan["reviewer_notes"] = res.reviewer_notes

    apply_file("cohort", Path(cohort_hitl_path) if cohort_hitl_path else None)
    apply_file("course", Path(course_hitl_path) if course_hitl_path else None)
    return out


__all__ = [
    "InstitutionSMATransformationHookHITLItems",
    "SMATransformationHookHITLItem",
    "SMATransformationHookHITLOption",
    "SMATransformationHookResolution",
    "apply_transformation_hook_hitl_resolutions",
    "build_transformation_hook_hitl_envelope_for_entity",
    "check_transformation_hook_hitl_gate",
    "default_transformation_hook_hitl_options",
    "iter_hook_required_plans",
    "write_transformation_hook_hitl_envelope",
]
