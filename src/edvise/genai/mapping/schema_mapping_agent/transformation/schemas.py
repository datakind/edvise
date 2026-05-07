"""Pydantic models for Step 2b — transformation map (value transforms on resolved Series)."""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from edvise.genai.mapping.shared.hitl.confidence import (
    PIPELINE_HITL_CONFIDENCE_THRESHOLD,
)
from edvise.genai.mapping.shared.pipeline_artifacts import default_pipeline_version

from ..manifest.schemas import (
    EntityType,
    ReviewStatus,
    StrictBaseModel,
    _omit_field_blocks_from_class_source,
)


class CastNullableIntStep(StrictBaseModel):
    function_name: Literal["cast_nullable_int"]
    column: str
    rationale: Optional[str] = None


class CastNullableFloatStep(StrictBaseModel):
    function_name: Literal["cast_nullable_float"]
    column: str
    rationale: Optional[str] = None


class CastStringStep(StrictBaseModel):
    function_name: Literal["cast_string"]
    column: str
    rationale: Optional[str] = None


class CastBooleanStep(StrictBaseModel):
    function_name: Literal["cast_boolean"]
    column: str
    boolean_map: Optional[Dict[str, bool]] = None
    rationale: Optional[str] = None


class CastDatetimeStep(StrictBaseModel):
    function_name: Literal["cast_datetime"]
    column: str
    rationale: Optional[str] = None


class CoerceNumericStep(StrictBaseModel):
    function_name: Literal["coerce_numeric"]
    column: str
    rationale: Optional[str] = None


class CoerceDatetimeStep(StrictBaseModel):
    function_name: Literal["coerce_datetime"]
    column: str
    fmt: Optional[str] = None
    rationale: Optional[str] = None


class StripWhitespaceStep(StrictBaseModel):
    function_name: Literal["strip_whitespace"]
    column: str
    rationale: Optional[str] = None


class LowercaseStep(StrictBaseModel):
    function_name: Literal["lowercase"]
    column: str
    rationale: Optional[str] = None


class UppercaseStep(StrictBaseModel):
    function_name: Literal["uppercase"]
    column: str
    rationale: Optional[str] = None


class MapValuesStep(StrictBaseModel):
    function_name: Literal["map_values"]
    column: str
    mapping: Dict[str, Any]
    default: Optional[str] = Field(
        default="passthrough",
        description=(
            "'passthrough' keeps original value for unmapped entries, "
            "null fills with NA."
        ),
    )
    rationale: Optional[str] = None


class NormalizeGradeStep(StrictBaseModel):
    function_name: Literal["normalize_grade"]
    column: str
    rationale: Optional[str] = None


class NormalizeEnrollmentStep(StrictBaseModel):
    function_name: Literal["normalize_enrollment"]
    column: str
    rationale: Optional[str] = None


class NormalizePellStep(StrictBaseModel):
    function_name: Literal["normalize_pell"]
    column: str
    rationale: Optional[str] = None


class NormalizeCredentialStep(StrictBaseModel):
    function_name: Literal["normalize_credential"]
    column: str
    rationale: Optional[str] = None


class NormalizeStudentAgeStep(StrictBaseModel):
    function_name: Literal["normalize_student_age"]
    column: str
    rationale: Optional[str] = None


class FillNullsStep(StrictBaseModel):
    function_name: Literal["fill_nulls"]
    column: str
    value: Any
    rationale: Optional[str] = None


class ReplaceNullTokensStep(StrictBaseModel):
    function_name: Literal["replace_null_tokens"]
    column: str
    null_tokens: List[str]
    rationale: Optional[str] = None


class ReplaceValuesWithNullStep(StrictBaseModel):
    function_name: Literal["replace_values_with_null"]
    column: str
    to_replace: Union[str, List[str]]
    rationale: Optional[str] = None


class StripTrailingDecimalStep(StrictBaseModel):
    function_name: Literal["strip_trailing_decimal"]
    column: str
    rationale: Optional[str] = None


class FillConstantStep(StrictBaseModel):
    function_name: Literal["fill_constant"]
    column: str
    value: str
    rationale: Optional[str] = None


class ExtractYearStep(StrictBaseModel):
    function_name: Literal["extract_year"]
    column: str
    rationale: Optional[str] = None


class SubstringAfterFirstDelimiterStep(StrictBaseModel):
    function_name: Literal["substring_after_first_delimiter"]
    column: str
    delimiter: str = "-"
    rationale: Optional[str] = None


class AcademicYearFromTermCodeDisplayStep(StrictBaseModel):
    function_name: Literal["academic_year_from_term_code_display"]
    column: str
    rationale: Optional[str] = None


class AcademicTermCategoryFromTermCodeDisplayStep(StrictBaseModel):
    function_name: Literal["academic_term_category_from_term_code_display"]
    column: str
    rationale: Optional[str] = None


class BirthyearToAgeBucketStep(StrictBaseModel):
    function_name: Literal["birthyear_to_age_bucket"]
    column: str
    extra_columns: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional columns resolved from base DataFrame before step runs. "
            "{'param_name': 'column_name'} — e.g. "
            "{'reference_year_series': 'acad_year'}"
        ),
    )
    rationale: Optional[str] = None


class ConditionalCreditsStep(StrictBaseModel):
    function_name: Literal["conditional_credits"]
    column: str
    extra_columns: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Additional columns resolved from base DataFrame before step runs. "
            "{'param_name': 'column_name'} — e.g. "
            "{'grade_series': 'course_grade'}"
        ),
    )
    rationale: Optional[str] = None


class TermComponentsToDatetimeStep(StrictBaseModel):
    function_name: Literal["term_components_to_datetime"]
    column: str
    extra_columns: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Required: season column bound to utility parameter ``season_series``. "
            "Example: {'season_series': '_edvise_term_season'}. "
            "``column`` must be the academic year string column (``_edvise_term_academic_year``)."
        ),
    )
    rationale: Optional[str] = None


class TermSeasonToConferralDateStep(StrictBaseModel):
    function_name: Literal["term_season_to_conferral_date"]
    column: str
    extra_columns: Dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Required: canonical season Series bound to ``season_series`` "
            "(e.g. output column of map_values on a season fragment). "
            "``column`` must be the academic year string column (YYYY-YY or compatible)."
        ),
    )
    rationale: Optional[str] = None


TransformationStep = Annotated[
    Union[
        CastNullableIntStep,
        CastNullableFloatStep,
        CastStringStep,
        CastBooleanStep,
        CastDatetimeStep,
        CoerceNumericStep,
        CoerceDatetimeStep,
        StripWhitespaceStep,
        LowercaseStep,
        UppercaseStep,
        MapValuesStep,
        NormalizeGradeStep,
        NormalizeEnrollmentStep,
        NormalizePellStep,
        NormalizeCredentialStep,
        NormalizeStudentAgeStep,
        FillNullsStep,
        ReplaceNullTokensStep,
        ReplaceValuesWithNullStep,
        StripTrailingDecimalStep,
        FillConstantStep,
        ExtractYearStep,
        SubstringAfterFirstDelimiterStep,
        AcademicYearFromTermCodeDisplayStep,
        AcademicTermCategoryFromTermCodeDisplayStep,
        BirthyearToAgeBucketStep,
        ConditionalCreditsStep,
        TermComponentsToDatetimeStep,
        TermSeasonToConferralDateStep,
    ],
    Field(discriminator="function_name"),
]


class FlaggedStep(StrictBaseModel):
    step_index: int = Field(
        ...,
        description="0-based index of the flagged step in the plan's steps array.",
    )
    function_name: str = Field(
        ...,
        description="Matches the flagged step's function_name.",
    )
    reason: Literal[
        "inferred_season_mapping",
        "inferred_value_mapping",
        "ambiguous_format",
        "low_confidence_utility_chain",
        "proxy_source",
    ]
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Evidence for the reviewer: sample_values, inferred_mapping, "
            "format assumptions, or other step-specific context."
        ),
    )


class TransformationHITLOption(StrictBaseModel):
    option_id: Literal["approve", "correct", "unmappable"]
    label: str
    description: str
    resolution: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "approve: {'approved': True}. "
            "correct: {'steps': [...corrected steps...]}. "
            "unmappable: {'steps': [], 'output_dtype': None}. "
            "Null on 'correct' when reviewer supplies steps out-of-band."
        ),
    )


class FieldTransformationPlan(StrictBaseModel):
    """
    Pure value transformation recipe for a single target field.

    No sourcing logic here — source table, join, and row selection are all
    declared in the corresponding FieldMappingRecord in the manifest.
    The field executor resolves the source Series from the manifest and
    passes it to these steps.
    """

    target_field: str = Field(..., description="Target Edvise schema field")
    output_dtype: Optional[str] = Field(
        default=None,
        description="Expected output dtype after all transformation steps",
    )
    steps: List[TransformationStep] = Field(
        default_factory=list,
        description=(
            "Ordered transformation steps applied to the resolved source Series. "
            "Empty = unmappable field (no steps to run). "
            "Steps are pure Series → Series — no join or sourcing logic here."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence in the transformation plan (0.0–1.0). "
            "1.0 = direct utility chain, no ambiguity; "
            "0.9 = standard chain with minor format assumption; "
            "0.7–0.8 = inferred mapping from sample_values or ambiguous format; "
            "≤ PIPELINE_HITL_CONFIDENCE_THRESHOLD = review_required must be true. "
            "Omit (null) for unmappable fields with empty steps."
        ),
    )
    review_required: Optional[bool] = Field(
        default=None,
        description=(
            "True when confidence is at or below PIPELINE_HITL_CONFIDENCE_THRESHOLD, "
            "or when the plan contains an inferred mapping that requires human "
            "confirmation before execution. Omit (null) for high-confidence plans."
        ),
    )
    review_status: Optional[ReviewStatus] = Field(
        default=None,
        description=(
            "Pipeline/HITL telemetry — set after validation and refinement, not by the "
            "initial transformation LLM. Omit in agent output. "
            f"After transformation review UC gate, set to {ReviewStatus.corrected_by_hitl.value!r} "
            "(same convention as manifest ``resolve_sma_items``) so low model confidence can remain "
            "without ``review_required``."
        ),
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments",
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description="Notes about validation concerns or issues with the transformation plan",
    )
    hook_required: Optional[bool] = Field(
        default=None,
        description=(
            "Set to true when no existing utility or combination of utilities can produce "
            "the correct output (same idea as IdentityAgent's hook_required path — custom work, not "
            "a built-in transform). Include a full explanation in reviewer_notes: what "
            "transformation is needed, what was attempted, and why no existing utility covers it. "
            "hook_required is a plan-level flag — not a step type. steps may be empty or contain "
            "a best-effort partial chain."
        ),
    )
    flagged_steps: Optional[List[FlaggedStep]] = Field(
        default=None,
        description=(
            "When review_required is true: non-empty list of steps that drove uncertainty "
            "(evidence only — resolution is plan-level). Omit (null) when review_required is omitted."
        ),
    )
    hitl_options: Optional[List[TransformationHITLOption]] = Field(
        default=None,
        description=(
            "When review_required is true: exactly three entries in order approve, correct, unmappable "
            "with field-specific labels and descriptions. Omit (null) when review_required is omitted."
        ),
    )

    @model_validator(mode="after")
    def confidence_review_and_hitl_consistency(self) -> FieldTransformationPlan:
        if (
            self.confidence is not None
            and self.confidence <= PIPELINE_HITL_CONFIDENCE_THRESHOLD
        ):
            hitl_finalized = self.review_status == ReviewStatus.corrected_by_hitl
            if self.review_required is not True and not hitl_finalized:
                raise ValueError(
                    "review_required must be True when confidence <= "
                    f"{PIPELINE_HITL_CONFIDENCE_THRESHOLD} unless review_status is "
                    f"{ReviewStatus.corrected_by_hitl.value!r} (transformation review HITL applied); "
                    f"(got confidence={self.confidence})"
                )

        if self.review_required is True:
            if self.confidence is None:
                raise ValueError(
                    "confidence must be set when review_required is True "
                    f"(PIPELINE_HITL_CONFIDENCE_THRESHOLD={PIPELINE_HITL_CONFIDENCE_THRESHOLD})"
                )
            if not self.flagged_steps:
                raise ValueError(
                    "flagged_steps must be a non-empty list when review_required is True"
                )
            if not self.hitl_options or len(self.hitl_options) != 3:
                raise ValueError(
                    "hitl_options must contain exactly three options when review_required is True"
                )
            ho_ids = [o.option_id for o in self.hitl_options]
            if ho_ids != ["approve", "correct", "unmappable"]:
                raise ValueError(
                    "hitl_options must be approve, correct, unmappable in that order "
                    f"(got {ho_ids!r})"
                )
        else:
            if self.flagged_steps is not None:
                raise ValueError(
                    "flagged_steps must be omitted (null) unless review_required is True"
                )
            if self.hitl_options is not None:
                raise ValueError(
                    "hitl_options must be omitted (null) unless review_required is True"
                )
        return self


class TransformationHITLItem(StrictBaseModel):
    item_id: str = Field(
        ...,
        description="Unique: <institution_id>_<entity_type>_<target_field>.",
    )
    institution_id: str
    entity_type: EntityType
    target_field: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged_steps: List[FlaggedStep] = Field(
        ...,
        min_length=1,
        description=(
            "One entry per uncertain step. Evidence for reviewer — "
            "resolution is always at the plan level, not per step."
        ),
    )
    steps: List[TransformationStep] = Field(
        ...,
        description=(
            "Full proposed steps array — what the reviewer is approving or correcting."
        ),
    )
    reviewer_notes: Optional[str] = None
    validation_notes: Optional[str] = None
    options: List[TransformationHITLOption] = Field(
        ...,
        min_length=3,
        max_length=3,
        description=(
            "Always exactly three options in order: approve, correct, unmappable."
        ),
    )
    status: Literal["pending", "approved", "corrected", "unmappable"] = "pending"
    # Streamlit HITL persists 1-based index as JSON int; manual edits may use strings.
    choice: Optional[Union[str, int]] = Field(
        default=None,
        description=(
            "1-based index into ``options`` (1=approve, 2=correct, 3=unmappable). "
            "Stored as int by the generic option UI."
        ),
    )

    @model_validator(mode="after")
    def _options_must_be_approve_correct_unmappable(self) -> "TransformationHITLItem":
        ids = [o.option_id for o in self.options]
        if ids != ["approve", "correct", "unmappable"]:
            raise ValueError(
                f"options must be exactly ['approve', 'correct', 'unmappable'] "
                f"in that order, got {ids}"
            )
        return self


class TransformationReview(StrictBaseModel):
    institution_id: str
    pipeline_version: str = Field(default_factory=default_pipeline_version)
    reviewed: bool = False
    hitl_items: List[TransformationHITLItem] = Field(
        default_factory=list,
        description="Low-confidence plans requiring reviewer confirmation.",
    )

    @property
    def is_clear(self) -> bool:
        """True when all hitl_items are resolved."""
        return all(
            i.status in ("approved", "corrected", "unmappable") for i in self.hitl_items
        )

    @property
    def has_pending(self) -> bool:
        return any(i.status == "pending" for i in self.hitl_items)


def _default_transformation_hitl_options() -> List[TransformationHITLOption]:
    return [
        TransformationHITLOption(
            option_id="approve",
            label="Approve",
            description="Run the proposed transformation steps as-is.",
            resolution={"approved": True},
        ),
        TransformationHITLOption(
            option_id="correct",
            label="Correct",
            description=(
                "Replace with corrected steps via resolution or an out-of-band correction."
            ),
            resolution=None,
        ),
        TransformationHITLOption(
            option_id="unmappable",
            label="Unmappable",
            description="Mark as unmappable: no steps and no output dtype.",
            resolution={"steps": [], "output_dtype": None},
        ),
    ]


def extract_transformation_review(
    transformation_map: TransformationMap,
    institution_id: str,
) -> TransformationReview:
    """
    Extract all plans where review_required is True into a TransformationReview artifact.

    Iterates both cohort and course TransformationMap entities. For each plan
    where review_required is True, emits a TransformationHITLItem with:
    - item_id: <institution_id>_<entity_type>_<target_field>
    - flagged_steps: from the plan when present; otherwise a single placeholder flag
    - options: from plan.hitl_options when present; otherwise generic approve/correct/unmappable

    Returns TransformationReview with empty hitl_items when no plans require review.
    Written to transformation_review.json by the pipeline alongside transformation_map.json.
    """
    raw_et = transformation_map.entity_type
    entity_enum: EntityType = (
        raw_et if isinstance(raw_et, EntityType) else EntityType(str(raw_et))
    )

    hitl_items: List[TransformationHITLItem] = []
    for plan in transformation_map.plans:
        if plan.review_required is not True:
            continue
        if plan.confidence is None:
            raise ValueError(
                f"target_field={plan.target_field!r}: review_required is True but confidence is null; "
                "set confidence on the plan before extracting TransformationHITLItem records."
            )
        fn = plan.steps[0].function_name if plan.steps else ""
        flagged_steps = plan.flagged_steps
        if not flagged_steps:
            flagged_steps = [
                FlaggedStep(
                    step_index=0,
                    function_name=fn,
                    reason="low_confidence_utility_chain",
                    context={
                        "pending_detailed_flagging": True,
                        "note": (
                            "Placeholder until enrichment populates per-step flags."
                        ),
                    },
                )
            ]
        options = (
            plan.hitl_options
            if plan.hitl_options is not None
            else _default_transformation_hitl_options()
        )
        hitl_items.append(
            TransformationHITLItem(
                item_id=(f"{institution_id}_{entity_enum.value}_{plan.target_field}"),
                institution_id=institution_id,
                entity_type=entity_enum,
                target_field=plan.target_field,
                confidence=plan.confidence,
                flagged_steps=list(flagged_steps),
                steps=list(plan.steps),
                reviewer_notes=plan.reviewer_notes,
                validation_notes=plan.validation_notes,
                options=list(options),
            )
        )
    return TransformationReview(
        institution_id=institution_id,
        pipeline_version=transformation_map.pipeline_version,
        reviewed=False,
        hitl_items=hitl_items,
    )


class TransformationMap(StrictBaseModel):
    """
    Per-entity transformation map. Release and institution identifiers are injected by the
    pipeline after the Step 2b LLM; the model authors entity_type, target_schema, and plans.
    """

    pipeline_version: str = Field(
        default_factory=default_pipeline_version,
        description="Edvise/git release — set by the pipeline, not the LLM.",
    )
    institution_id: str = Field(..., description="Institution identifier")
    entity_type: EntityType = Field(..., description="cohort or course")
    target_schema: str = Field(..., description="Target schema name")
    plans: List[FieldTransformationPlan] = Field(
        ...,
        min_length=1,
        description=(
            "Per-target-field transformation plans. "
            "Each plan receives a pre-resolved Series from the field executor "
            "and applies pure Series → Series transformation steps."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_schema_version_key(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        legacy, current = "schema_version", "pipeline_version"
        if legacy in d and current not in d:
            d[current] = d.pop(legacy)
        return d

    @field_validator("plans")
    @classmethod
    def target_fields_must_be_unique(
        cls, v: List[FieldTransformationPlan]
    ) -> List[FieldTransformationPlan]:
        targets = [p.target_field for p in v]
        if len(targets) != len(set(targets)):
            raise ValueError(
                "Each target_field must appear only once in the transformation map"
            )
        return v


_AGENT_EXCLUDED_TM_TOP_FIELDS = frozenset({"pipeline_version", "institution_id"})


def _transformation_map_source_for_agent_prompt() -> str:
    """Hide pipeline-injected fields so Step 2b prompts do not show them to the LLM."""
    return _omit_field_blocks_from_class_source(
        inspect.getsource(TransformationMap),
        _AGENT_EXCLUDED_TM_TOP_FIELDS,
    )


def get_transformation_map_schema_context() -> str:
    """
    Focused schema reference for Agent 2b prompt context: TransformationStep models,
    FieldTransformationPlan, HITL companion models, and TransformationMap.
    """
    models = [
        CastNullableIntStep,
        CastNullableFloatStep,
        CastStringStep,
        CastBooleanStep,
        CastDatetimeStep,
        CoerceNumericStep,
        CoerceDatetimeStep,
        StripWhitespaceStep,
        LowercaseStep,
        UppercaseStep,
        MapValuesStep,
        NormalizeGradeStep,
        NormalizeEnrollmentStep,
        NormalizePellStep,
        NormalizeCredentialStep,
        NormalizeStudentAgeStep,
        FillNullsStep,
        ReplaceNullTokensStep,
        ReplaceValuesWithNullStep,
        StripTrailingDecimalStep,
        FillConstantStep,
        ExtractYearStep,
        BirthyearToAgeBucketStep,
        ConditionalCreditsStep,
        TermComponentsToDatetimeStep,
        TermSeasonToConferralDateStep,
        FieldTransformationPlan,
        FlaggedStep,
        TransformationHITLOption,
        TransformationHITLItem,
        TransformationReview,
        TransformationMap,
    ]
    sections = []
    for model in models:
        try:
            if model is TransformationMap:
                source = _transformation_map_source_for_agent_prompt()
            else:
                source = inspect.getsource(model)
            sections.append(source)
        except (OSError, TypeError):
            pass
    return "\n\n".join(sections)
