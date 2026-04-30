"""
Step 2b — ``review_required`` transformation plans: UC HITL gate before hook preview.

Extracts low-confidence / inferred-step plans into per-entity JSON (same ``items`` envelope
shape as hook HITL for Streamlit generic option UI). After UC approval,
:func:`apply_transformation_review_resolutions` merges reviewer outcomes into the in-memory
transformation map wrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from edvise.genai.mapping.shared.hitl import raise_if_hitl_pending
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.pipeline_artifacts import default_pipeline_version

from ..schemas import (
    TransformationHITLItem,
    TransformationMap,
    extract_transformation_review,
)

logger = logging.getLogger(__name__)


class TransformationReviewHITLFile(BaseModel):
    """
    On-disk JSON for ``cohort_transformation_review.json`` / ``course_transformation_review.json``.

    Uses ``items`` (not ``hitl_items``) so the Streamlit HITL app generic option editor applies.
    """

    model_config = ConfigDict(extra="forbid")

    institution_id: str
    entity_type: Literal["cohort", "course"]
    domain: Literal["transformation_review"] = "transformation_review"
    pipeline_version: str = Field(default_factory=default_pipeline_version)
    reviewed: bool = False
    items: list[TransformationHITLItem] = Field(default_factory=list)

    @classmethod
    def model_validate(cls, obj: Any, **kwargs: Any) -> TransformationReviewHITLFile:
        if isinstance(obj, dict):
            obj = dict(obj)
            if "items" not in obj and "hitl_items" in obj:
                obj["items"] = obj.pop("hitl_items")
            else:
                obj.pop("hitl_items", None)
        return super().model_validate(obj, **kwargs)

    @property
    def pending(self) -> list[TransformationHITLItem]:
        out: list[TransformationHITLItem] = []
        for i in self.items:
            if i.status != "pending":
                continue
            ch = i.choice
            if ch is None or str(ch).strip() == "":
                out.append(i)
        return out

    @property
    def is_clear(self) -> bool:
        return len(self.pending) == 0


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
        return "unmappable"
    return None


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
            "  • For each item set ``status`` to approved | corrected | unmappable, **or** set "
            "``choice`` to the 1-based option index (1=approve, 2=correct, 3=unmappable).\n"
            "  • For **correct**, edit ``steps`` on the item before saving.\n"
            "  • Save the JSON, then approve the UC hitl_reviews row."
        ),
    )


def _strip_review_metadata(plan: dict[str, Any]) -> None:
    for k in ("review_required", "flagged_steps", "hitl_options"):
        plan.pop(k, None)


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
    - **approved** — leave ``steps`` as in ``transformation_data``; strip review metadata keys.
    - **corrected** — replace ``plan['steps']`` from the item's ``steps`` field.
    - **unmappable** — ``steps=[]``, ``output_dtype`` null; strip review metadata.
    """
    out: dict[str, Any] = json.loads(json.dumps(transformation_data))
    tmaps = out.get("transformation_maps")
    if not isinstance(tmaps, dict):
        return out

    def apply_file(
        entity_type: Literal["cohort", "course"], path: Path | None
    ) -> None:
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
            if eff == "approved":
                _strip_review_metadata(plan)
            elif eff == "corrected":
                plan["steps"] = _steps_to_jsonable(item)
                _strip_review_metadata(plan)
            elif eff == "unmappable":
                plan["steps"] = []
                plan["output_dtype"] = None
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
