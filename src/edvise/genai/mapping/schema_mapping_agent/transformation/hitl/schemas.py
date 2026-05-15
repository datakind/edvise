"""Pydantic envelopes and options for Schema Mapping Agent Step 2b transformation HITL."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    TransformationHITLItem,
)
from edvise.genai.mapping.shared.pipeline_artifacts import default_pipeline_version


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


__all__ = [
    "InstitutionSMATransformationHookHITLItems",
    "SMATransformationHookHITLItem",
    "SMATransformationHookHITLOption",
    "SMATransformationHookResolution",
    "TransformationReviewHITLFile",
    "default_transformation_hook_hitl_options",
]
