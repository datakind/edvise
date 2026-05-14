"""Orchestrate HookSpec LLM calls for Step 2b ``hook_required`` plans (preview rows + reload)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from edvise.utils.llm_utils import llm_complete_with_parse_retry

from edvise.genai.mapping.shared.hitl.hook_spec.schemas import HITLDomain, HookSpec
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_required_hitl import (
    iter_hook_required_plans,
)
from edvise.genai.mapping.shared.hitl.hook_spec.parse import parse_hook_spec
from edvise.genai.mapping.shared.hitl.hook_spec.paths import ensure_hook_spec_file

from .prompt import (
    _slug_target,
    build_sma_transform_hook_system_prompt,
    build_sma_transform_hook_user_message,
    manifest_mapping_for_target,
    sma_transform_hook_item_id,
)

logger = logging.getLogger(__name__)


def generate_sma_transform_hook_spec(
    *,
    item_id: str,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    target_field: str,
    plan: dict[str, Any],
    manifest_record: dict[str, Any] | None,
    llm_complete: Callable[[str, str], str],
) -> HookSpec:
    slug = _slug_target(target_field)
    fn_name = f"transform_{entity_type}_{slug}"
    system = build_sma_transform_hook_system_prompt()
    user = build_sma_transform_hook_user_message(
        item_id=item_id,
        institution_id=institution_id,
        entity_type=entity_type,
        target_field=target_field,
        required_function_name=fn_name,
        plan=plan,
        manifest_record=manifest_record,
    )
    spec = llm_complete_with_parse_retry(
        llm_complete,
        system,
        user,
        parse_hook_spec,
        logger=logger,
    )
    spec = ensure_hook_spec_file(
        spec,
        institution_id=institution_id,
        domain=HITLDomain.TRANSFORM,
    )
    return spec


def generate_sma_transform_hook_preview_rows_for_entity(
    transformation_data: dict[str, Any],
    manifest_map: dict[str, Any],
    *,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    llm_complete: Callable[[str, str], str],
) -> list[dict[str, Any]]:
    """
    One LLM call per ``hook_required`` plan; returns rows for
    :func:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation.preview.write_sma_transform_hook_preview_json`.
    """
    rows: list[dict[str, Any]] = []
    for plan in iter_hook_required_plans(transformation_data, entity_type):
        tf = str(plan.get("target_field") or "").strip()
        item_id = sma_transform_hook_item_id(institution_id, entity_type, tf)
        mrec = manifest_mapping_for_target(manifest_map, entity_type, tf)
        try:
            spec = generate_sma_transform_hook_spec(
                item_id=item_id,
                institution_id=institution_id,
                entity_type=entity_type,
                target_field=tf,
                plan=plan,
                manifest_record=mrec,
                llm_complete=llm_complete,
            )
        except Exception:
            logger.exception(
                "SMA transform hook generation failed for %s %s", entity_type, tf
            )
            raise
        rows.append(
            {
                "item_id": item_id,
                "hook_spec": spec.model_dump(mode="json"),
                "review_context": {
                    "entity_type": entity_type,
                    "target_field": tf,
                    "manifest_mapping_record": mrec,
                    "field_transformation_plan": dict(plan),
                },
            }
        )
    return rows


def load_hook_specs_from_sma_preview_path(path: str | Path) -> list[HookSpec]:
    """Read a written preview JSON and return HookSpec instances (for materialize)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs_raw = data.get("specs")
    if not isinstance(specs_raw, list):
        return []
    out: list[HookSpec] = []
    for row in specs_raw:
        if not isinstance(row, dict):
            continue
        hs = row.get("hook_spec")
        if not isinstance(hs, dict):
            continue
        out.append(HookSpec.model_validate(hs))
    return out


__all__ = [
    "generate_sma_transform_hook_preview_rows_for_entity",
    "generate_sma_transform_hook_spec",
    "load_hook_specs_from_sma_preview_path",
]
