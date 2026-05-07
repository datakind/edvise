"""
Step 2b — ``hook_required`` transformation plans: option-choice HITL after hook preview.

After Step 2b, the pipeline runs transform HookSpec generation + UC ``sma_gate_2_hook_preview``,
materializes ``transform_hooks.py``, then builds per-entity review JSON for
``sma_gate_2_hook_required``, waits for approval, applies reviewer resolutions, persists
``transformation_map.json``, and executes transforms.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edvise.genai.mapping.shared.hitl import raise_if_hitl_pending
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json

logger = logging.getLogger(__name__)


class SMATransformationHookResolution(BaseModel):
    """Structured mutation applied to one plan when a reviewer option wins."""

    model_config = ConfigDict(extra="forbid")

    clear_hook_required: bool = Field(
        default=True,
        description="When true, sets plan.hook_required to false after review.",
    )
    replace_steps: bool = Field(
        default=False,
        description="When true, replaces plan.steps with ``steps`` (default empty list).",
    )
    steps: list[Any] | None = Field(
        default=None,
        description="New steps when replace_steps is true; null means [].",
    )
    reviewer_notes: str | None = Field(
        default=None,
        description="When set, overwrites plan.reviewer_notes.",
    )


class SMATransformationHookHITLOption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    option_id: str
    label: str
    description: str
    resolution: SMATransformationHookResolution


class SMATransformationHookHITLItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    institution_id: str
    entity_type: Literal["cohort", "course"]
    target_field: str
    hitl_question: str = Field(
        ...,
        description="Actionable question for reviewers (Streamlit renders as headline).",
    )
    hitl_context: str | None = Field(
        default=None,
        description="Freeform evidence string (e.g. model reviewer_notes / validation_notes).",
    )
    plan_snapshot: dict[str, Any] = Field(
        ...,
        description="Snapshot of the FieldTransformationPlan dict for JSON inspect in UI.",
    )
    current_field_mapping: dict[str, Any] = Field(
        ...,
        description="Minimal stub so SMA Streamlit shows the target_field headline.",
    )
    options: list[SMATransformationHookHITLOption] = Field(..., min_length=1)
    choice: int | None = Field(
        default=None,
        description="1-based index into options; null until reviewed.",
    )

    def selected_option(self) -> SMATransformationHookHITLOption | None:
        if self.choice is None:
            return None
        ix = int(self.choice) - 1
        if ix < 0 or ix >= len(self.options):
            return None
        return self.options[ix]


class InstitutionSMATransformationHookHITLItems(BaseModel):
    model_config = ConfigDict(extra="forbid")

    institution_id: str
    entity_type: Literal["cohort", "course"]
    domain: Literal["transformation_hook_required"] = "transformation_hook_required"
    items: list[SMATransformationHookHITLItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _items_match_envelope(self) -> InstitutionSMATransformationHookHITLItems:
        for it in self.items:
            if it.institution_id != self.institution_id:
                raise ValueError(
                    f"item_id={it.item_id!r} institution_id does not match envelope"
                )
            if it.entity_type != self.entity_type:
                raise ValueError(
                    f"item_id={it.item_id!r} entity_type does not match envelope"
                )
        return self

    @property
    def pending(self) -> list[SMATransformationHookHITLItem]:
        return [i for i in self.items if i.choice is None]

    @property
    def is_clear(self) -> bool:
        return len(self.pending) == 0


def default_transformation_hook_hitl_options() -> list[SMATransformationHookHITLOption]:
    """Standard reviewer paths — clear the flag; either keep steps or drop them."""
    return [
        SMATransformationHookHITLOption(
            option_id="accept_partial_chain",
            label="Accept partial utility chain",
            description=(
                "Clear hook_required and keep the model's transformation steps — execution "
                "proceeds with best-effort utilities."
            ),
            resolution=SMATransformationHookResolution(
                clear_hook_required=True,
                replace_steps=False,
                steps=None,
            ),
        ),
        SMATransformationHookHITLOption(
            option_id="defer_field_empty_steps",
            label="Defer field (empty steps)",
            description=(
                "Clear hook_required and remove all steps — the executor skips this target "
                "until you add a custom hook or edit the transformation map."
            ),
            resolution=SMATransformationHookResolution(
                clear_hook_required=True,
                replace_steps=True,
                steps=[],
            ),
        ),
    ]


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
    "iter_hook_required_plans",
    "InstitutionSMATransformationHookHITLItems",
    "SMATransformationHookHITLItem",
    "SMATransformationHookHITLOption",
    "SMATransformationHookResolution",
    "apply_transformation_hook_hitl_resolutions",
    "build_transformation_hook_hitl_envelope_for_entity",
    "check_transformation_hook_hitl_gate",
    "default_transformation_hook_hitl_options",
    "write_transformation_hook_hitl_envelope",
]
