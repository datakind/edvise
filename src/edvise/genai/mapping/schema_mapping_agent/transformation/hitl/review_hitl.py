"""
Step 2b — ``review_required`` transformation plans: build review JSON, write, merge resolutions.

Gate checks live in :mod:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates`.
Envelope types live in :mod:`~edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import ReviewStatus
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.gates import (
    check_transformation_review_hitl_gate,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.schemas import (
    TransformationReviewHITLFile,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.pipeline_artifacts import default_pipeline_version

from ..schemas import (
    TransformationHITLItem,
    TransformationMap,
    extract_transformation_review,
)

logger = logging.getLogger(__name__)


def _plans_from_entity_section(entity_blob: object) -> list[dict[str, Any]] | None:
    if not isinstance(entity_blob, dict):
        return None
    raw_plans = entity_blob.get("plans")
    if not isinstance(raw_plans, list):
        return None
    return [p for p in raw_plans if isinstance(p, dict)]


def _find_plan_index(plans: list[dict[str, Any]], target_field: str) -> int | None:
    for i, p in enumerate(plans):
        if str(p.get("target_field") or "").strip() == target_field:
            return i
    return None


def build_transformation_review_hitl_file_for_entity(
    transformation_data: dict[str, Any],
    *,
    institution_id: str,
    entity_type: Literal["cohort", "course"],
    pipeline_version: str | None = None,
) -> TransformationReviewHITLFile:
    """Build review file content from the Step 2b transformation wrapper dict."""
    tmaps = transformation_data.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return TransformationReviewHITLFile(
            institution_id=institution_id,
            entity_type=entity_type,
            pipeline_version=pipeline_version or default_pipeline_version(),
            items=[],
        )
    sec = tmaps.get(entity_type)
    if not isinstance(sec, dict):
        return TransformationReviewHITLFile(
            institution_id=institution_id,
            entity_type=entity_type,
            pipeline_version=pipeline_version or default_pipeline_version(),
            items=[],
        )
    pv = pipeline_version or default_pipeline_version()
    tm_dict = {
        **sec,
        "institution_id": institution_id,
        "pipeline_version": pv,
        "entity_type": entity_type,
    }
    tm = TransformationMap.model_validate(tm_dict)
    rev = extract_transformation_review(tm, institution_id)
    return TransformationReviewHITLFile(
        institution_id=institution_id,
        entity_type=entity_type,
        pipeline_version=rev.pipeline_version,
        reviewed=rev.reviewed,
        items=list(rev.hitl_items),
    )


def write_transformation_review_hitl_file(
    path: str | Path, env: TransformationReviewHITLFile
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(env.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _effective_item_status(item: TransformationHITLItem) -> str | None:
    """Resolved status from ``status`` or 1-based ``choice`` index (Streamlit pattern)."""
    if item.status != "pending":
        if item.status == "unmappable":
            return "hook_required"
        return item.status
    ch = item.choice
    if ch is None:
        return None
    try:
        ix = int(str(ch).strip())
    except (TypeError, ValueError):
        return None
    if ix == 1:
        return "approved"
    if ix == 2:
        return "corrected"
    if ix == 3:
        return "hook_required"
    return None


def _strip_review_metadata(plan: dict[str, Any]) -> None:
    for k in ("review_required", "flagged_steps", "hitl_options"):
        plan.pop(k, None)
    # Match manifest HITL resolver: human-finalized records carry corrected_by_hitl.
    # Keeps model confidence for audit while satisfying FieldTransformationPlan rules.
    plan["review_status"] = ReviewStatus.corrected_by_hitl.value


def _steps_to_jsonable(item: TransformationHITLItem) -> list[dict[str, Any]]:
    return [s.model_dump(mode="json") for s in item.steps]


def apply_transformation_review_resolutions(
    transformation_data: dict[str, Any],
    *,
    cohort_review_path: str | Path | None,
    course_review_path: str | Path | None,
) -> dict[str, Any]:
    """
    Deep-copy ``transformation_data`` and merge reviewer resolutions from review JSON files.

    Resolution rules (per item, using ``status`` or 1-based ``choice``):
    - **approved** — leave ``steps`` as in ``transformation_data``; ``hook_required`` false;
      strip pending-review keys; set ``review_status`` to ``corrected_by_hitl``.
    - **corrected** — replace ``plan['steps']`` from the item's ``steps`` field; ``hook_required`` false.
    - **hook_required** — ``hook_required`` true; optional ``reviewer_note`` on plan; keep item steps
      as documentation only; strip pending-review keys.
    """
    out: dict[str, Any] = json.loads(json.dumps(transformation_data))
    tmaps = out.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return out

    def apply_file(entity_type: Literal["cohort", "course"], path: Path | None) -> None:
        if path is None or not path.is_file():
            return
        env = read_pydantic_json(path, TransformationReviewHITLFile)
        entity_blob = tmaps.get(entity_type)
        plans = _plans_from_entity_section(entity_blob)
        if not plans:
            return
        for item in env.items:
            eff = _effective_item_status(item)
            if eff is None:
                continue
            ix = _find_plan_index(plans, item.target_field)
            if ix is None:
                logger.warning(
                    "Transformation review item %s targets unknown field %s — skipping",
                    item.item_id,
                    item.target_field,
                )
                continue
            plan = plans[ix]
            # Legacy review files may still carry status/option "unmappable".
            if eff == "unmappable":
                eff = "hook_required"
            if eff == "approved":
                plan["hook_required"] = False
                _strip_review_metadata(plan)
            elif eff == "corrected":
                plan["hook_required"] = False
                plan["steps"] = _steps_to_jsonable(item)
                _strip_review_metadata(plan)
            elif eff == "hook_required":
                plan["hook_required"] = True
                note = (item.reviewer_note or "").strip()
                if note:
                    plan["reviewer_notes"] = note
                _strip_review_metadata(plan)

    apply_file("cohort", Path(cohort_review_path) if cohort_review_path else None)
    apply_file("course", Path(course_review_path) if course_review_path else None)
    return out


__all__ = [
    "TransformationReviewHITLFile",
    "apply_transformation_review_resolutions",
    "build_transformation_review_hitl_file_for_entity",
    "check_transformation_review_hitl_gate",
    "write_transformation_review_hitl_file",
]
