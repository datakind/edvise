"""Pydantic models for Step 2b — transformation map (value transforms on resolved Series)."""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from edvise.genai.mapping.shared.pipeline_artifacts import resolve_pipeline_version

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


class NewUtilityNeededStep(StrictBaseModel):
    function_name: Literal["NEW_UTILITY_NEEDED"]
    description: str
    rationale: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_gap(self) -> bool:
        return True


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
        NewUtilityNeededStep,
    ],
    Field(discriminator="function_name"),
]


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
    review_status: Optional[ReviewStatus] = Field(
        default=None,
        description=(
            "Pipeline/HITL telemetry — set after validation and refinement, not by the "
            "initial transformation LLM. Omit in agent output."
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


class TransformationMap(StrictBaseModel):
    """
    Per-entity transformation map. Release and institution identifiers are injected by the
    pipeline after the Step 2b LLM; the model authors entity_type, target_schema, and plans.
    """

    pipeline_version: str = Field(
        default_factory=resolve_pipeline_version,
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
    FieldTransformationPlan, and TransformationMap.
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
        NewUtilityNeededStep,
        FieldTransformationPlan,
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
