from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Shared base + enums
# =============================================================================

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class EntityType(str, Enum):
    cohort = "cohort"
    course = "course"


class ReviewStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    corrected = "corrected"
    rejected = "rejected"


# =============================================================================
# 2a — Field Mapping Manifest
# =============================================================================

class FieldMappingRecord(StrictBaseModel):
    target_field: str = Field(..., description="Target Edvise schema field")
    source_columns: List[str] = Field(
        default_factory=list,
        description="Source columns proposed for this target field. Empty = unmappable field.",
    )
    source_table: Optional[str] = Field(
        default=None,
        description="Source table name (e.g. 'student_df', 'course_df'). None = unmappable field.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Agent confidence in the proposed mapping. "
            "1.0 for human-authored cold-start records and confirmed unmappable fields. "
            "Drives HITL gate threshold — low confidence triggers mandatory review."
        ),
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Explanation for why the mapping was selected or why the field is unmappable.",
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.proposed,
        description="Human review outcome",
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments or corrections",
    )
    corrected_source_columns: Optional[List[str]] = Field(
        default=None,
        description="Reviewer-corrected source columns, if applicable",
    )

    @field_validator("source_columns")
    @classmethod
    def source_columns_no_empty_strings(cls, v: List[str]) -> List[str]:
        if any(not col.strip() for col in v):
            raise ValueError("source_columns cannot contain empty strings")
        return v

    @field_validator("corrected_source_columns")
    @classmethod
    def corrected_source_columns_no_empty_strings(
        cls, v: Optional[List[str]]
    ) -> Optional[List[str]]:
        if v is not None and any(not col.strip() for col in v):
            raise ValueError("corrected_source_columns cannot contain empty strings")
        return v


class ColumnAlias(StrictBaseModel):
    table: str = Field(
        ...,
        description="Source table containing the aliased column",
    )
    source_column: str = Field(
        ...,
        description="Column name as it appears in the source table",
    )
    canonical_column: str = Field(
        ...,
        description=(
            "Canonical column name to use for join key matching. "
            "Typically the name as it appears in the primary table or "
            "the normalized form shared across tables."
        ),
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Explanation of why these column names refer to the same concept",
    )


class FieldMappingManifest(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
    institution_id: str = Field(..., description="Institution identifier")
    entity_type: EntityType = Field(..., description="cohort or course")
    target_schema: str = Field(..., description="Target schema name")
    mappings: List[FieldMappingRecord] = Field(
        ...,
        min_length=1,
        description="Per-target-field mapping proposals",
    )
    column_aliases: List[ColumnAlias] = Field(
        default_factory=list,
        description=(
            "Cross-table column name aliases identified during mapping. "
            "Captures cases where the same concept appears under different "
            "names across source tables — e.g. term_descr in course_df vs "
            "term_desc in student_df. Consumed by join resolver for key matching."
        ),
    )

    @field_validator("mappings")
    @classmethod
    def target_fields_must_be_unique(
        cls, v: List[FieldMappingRecord]
    ) -> List[FieldMappingRecord]:
        targets = [m.target_field for m in v]
        if len(targets) != len(set(targets)):
            raise ValueError("Each target_field must appear only once in the manifest")
        return v


# =============================================================================
# 2b — Transformation Map
# =============================================================================

# -----------------------------------------------------------------------------
# Pre-collapse config
# DataFrame-level grain reduction applied BEFORE field-level plans run.
# Used when the source DataFrame has finer grain than the target schema —
# e.g. UCF course_df has section-level rows but target requires course-level.
# Executor runs this once on the full DataFrame before any field plans execute.
# -----------------------------------------------------------------------------

class PreCollapseConfig(StrictBaseModel):
    subset: List[str] = Field(
        ...,
        description=(
            "SOURCE column names defining the target grain to deduplicate on. "
            "Must use source column names since pre_collapse runs before any "
            "transformation steps — target column names do not exist yet. "
            "e.g. ['student_id', 'term_descr', 'crse_prefix', 'crse_number'] for UCF course "
            "(not ['student_id', 'academic_term', 'course_prefix', 'course_number'])."
        ),
    )
    keep: Literal["first", "last"] = Field(
        default="first",
        description="Which row to keep per unique combination of subset columns.",
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Sort column ascending before deduplication to make keep deterministic.",
    )
    rationale: Optional[str] = None


# -----------------------------------------------------------------------------
# Collapse config
# Declares how to reduce student-term grain to student grain per field.
# The executor uses schema_contract.unique_keys as the groupby keys;
# CollapseConfig declares the within-group row selection strategy.
# Only relevant for cohort maps — course maps use pre_collapse instead.
# -----------------------------------------------------------------------------

class CollapseStrategy(str, Enum):
    any_row = "any_row"
    # Invariant fields — value is identical across all term rows per student.
    # Executor takes first row per groupby key (arbitrary but deterministic).
    # Examples: gender, race, enrollment_type

    first_by = "first_by"
    # Term-1 specific fields — take first row ordered by `order_by` ascending.
    # Requires `order_by` to be set.
    # Examples: program_of_study_term_1, cohort_term, cohort

    where_not_null = "where_not_null"
    # Take first row where `condition_col` is non-null per student.
    # Requires `condition_col` to be set.
    # Examples: first_bachelors_grad_date, major_grad (condition_col: degree_earned_term)

    constant = "constant"
    # No row selection — field is a constant value for all rows.
    # source_columns must be empty; value expressed in transformation steps.
    # Examples: credential_type_sought_year_1 at UCF (always "Bachelor's")

    none = "none"
    # No collapse needed — source data is already at the correct grain.
    # Used for all course-level fields after pre_collapse has run.


class CollapseConfig(StrictBaseModel):
    strategy: CollapseStrategy = Field(..., description="Row selection strategy")
    order_by: Optional[str] = Field(
        default=None,
        description="Column to sort ascending before taking first row. Required for first_by.",
    )
    condition_col: Optional[str] = Field(
        default=None,
        description="Column that must be non-null for row selection. Required for where_not_null.",
    )

    @model_validator(mode="after")
    def validate_strategy_args(self) -> "CollapseConfig":
        if self.strategy == CollapseStrategy.first_by and not self.order_by:
            raise ValueError("order_by is required when strategy is first_by")
        if self.strategy == CollapseStrategy.where_not_null and not self.condition_col:
            raise ValueError("condition_col is required when strategy is where_not_null")
        return self


# -----------------------------------------------------------------------------
# Typed step models
# Each maps 1:1 to a callable in transformation_utilities.py.
# `column` is always explicit — the executor never infers the target column.
# -----------------------------------------------------------------------------

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
    fmt: Optional[str] = Field(
        default=None,
        description="strptime format string. None = pandas infers.",
    )
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
    mapping: Dict[str, Any] = Field(..., description="Value mapping dict")
    rationale: Optional[str] = None

    @field_validator("mapping")
    @classmethod
    def mapping_must_not_be_empty(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not v:
            raise ValueError("mapping must not be empty")
        return v


class NormalizeTermCodeStep(StrictBaseModel):
    function_name: Literal["normalize_term_code"]
    column: str
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
    value: Any = Field(..., description="Fill value")
    rationale: Optional[str] = None


class ReplaceNullTokensStep(StrictBaseModel):
    function_name: Literal["replace_null_tokens"]
    column: str
    null_tokens: List[str] = Field(..., description="Token strings to replace with NA")
    rationale: Optional[str] = None


class ReplaceValuesWithNullStep(StrictBaseModel):
    function_name: Literal["replace_values_with_null"]
    column: str
    to_replace: Union[str, List[str]]
    rationale: Optional[str] = None


class CombineColumnsStep(StrictBaseModel):
    function_name: Literal["combine_columns"]
    cols: List[str] = Field(..., min_length=2)
    output_col: str
    sep: str = Field(default="")
    rationale: Optional[str] = None


class DeduplicateRowsStep(StrictBaseModel):
    function_name: Literal["deduplicate_rows"]
    subset: Optional[List[str]] = Field(
        default=None,
        description="Columns to consider. None = all columns.",
    )
    keep: Literal["first", "last", "none"] = Field(default="first")
    rationale: Optional[str] = None


class StripTrailingDecimalStep(StrictBaseModel):
    function_name: Literal["strip_trailing_decimal"]
    column: str
    rationale: Optional[str] = None


class FillConstantStep(StrictBaseModel):
    function_name: Literal["fill_constant"]
    column: str = Field(
        ...,
        description="Target field name — no source column exists for constant fields",
    )
    value: str = Field(..., description="Constant string value to fill all rows with")
    rationale: Optional[str] = None


class NormalizeYearRangeStep(StrictBaseModel):
    function_name: Literal["normalize_year_range"]
    column: str = Field(..., description="Column containing year range e.g. '2018-2019' or '2018-19'")
    rationale: Optional[str] = None


class ExtractYearStep(StrictBaseModel):
    function_name: Literal["extract_year"]
    column: str = Field(..., description="Column containing year range string e.g. '2018-2019'")
    rationale: Optional[str] = None


class ParseYyyymmStep(StrictBaseModel):
    function_name: Literal["parse_yyyymm"]
    column: str = Field(..., description="Column containing YYYYMM string e.g. '202301'")
    rationale: Optional[str] = None


class BirthyearToAgeBucketStep(StrictBaseModel):
    function_name: Literal["birthyear_to_age_bucket"]
    column: str = Field(..., description="Column containing birth year as integer")
    reference_year_column: Optional[str] = Field(
        default=None,
        description=(
            "Optional column containing reference year in YYYY-YY format "
            "(e.g. academic_year). If provided, age is calculated relative to "
            "that year rather than current year."
        ),
    )
    rationale: Optional[str] = None


class ConditionalCreditsStep(StrictBaseModel):
    function_name: Literal["conditional_credits"]
    grade_column: str = Field(..., description="Column containing grade values")
    credits_column: str = Field(..., description="Column containing credits attempted")
    rationale: Optional[str] = None


class StemsLookupStep(StrictBaseModel):
    function_name: Literal["stems_lookup"]
    column: str = Field(..., description="Column containing CIP code (may be datetime dtype)")
    stems_table: str = Field(
        ...,
        description="Key to look up stems DataFrame from executor context dict",
    )
    rationale: Optional[str] = None


class CrossTableLookupStep(StrictBaseModel):
    function_name: Literal["cross_table_lookup"]
    base_join_keys: List[str] = Field(
        ...,
        description="Join key column names in the base DataFrame (source names)",
    )
    lookup_table: str = Field(
        ...,
        description="Key to look up DataFrame from executor context dict",
    )
    lookup_join_keys: List[str] = Field(
        ...,
        description="Join key column names in the lookup DataFrame (may differ from base)",
    )
    lookup_value_col: str = Field(
        ...,
        description="Column in lookup DataFrame to pull as the output value",
    )
    rationale: Optional[str] = None


class NewUtilityNeededStep(StrictBaseModel):
    function_name: Literal["NEW_UTILITY_NEEDED"]
    description: str = Field(..., description="Required utility behavior")
    rationale: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_gap(self) -> bool:
        return True


# Discriminated union — executor dispatches on function_name literal
TransformationStep = Union[
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
    NormalizeTermCodeStep,
    NormalizeGradeStep,
    NormalizeEnrollmentStep,
    NormalizePellStep,
    NormalizeCredentialStep,
    NormalizeStudentAgeStep,
    FillNullsStep,
    ReplaceNullTokensStep,
    ReplaceValuesWithNullStep,
    CombineColumnsStep,
    DeduplicateRowsStep,
    StripTrailingDecimalStep,
    FillConstantStep,
    ExtractYearStep,
    ParseYyyymmStep,
    BirthyearToAgeBucketStep,
    ConditionalCreditsStep,
    StemsLookupStep,
    CrossTableLookupStep,
    NewUtilityNeededStep,
]


# -----------------------------------------------------------------------------
# Field transformation plan + map
# -----------------------------------------------------------------------------

class FieldTransformationPlan(StrictBaseModel):
    target_field: str = Field(..., description="Target Edvise schema field")
    source_columns: List[str] = Field(
        default_factory=list,
        description="Source columns used to produce the target field. Empty = unmappable or constant.",
    )
    source_table: Optional[str] = Field(
        default=None,
        description="Source table name. None = unmappable field or constant derivation.",
    )
    output_dtype: Optional[str] = Field(
        default=None,
        description="Expected output dtype after all transformations",
    )
    collapse: Optional[CollapseConfig] = Field(
        default=None,
        description=(
            "Row collapse strategy for grain reduction from student-term to student grain. "
            "None = no collapse needed (course-level or already correct grain). "
            "Executor reads groupby keys from schema_contract.unique_keys."
        ),
    )
    steps: List[TransformationStep] = Field(
        default_factory=list,
        description="Ordered transformation steps. Empty = unmappable field.",
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.proposed,
        description="Human review outcome",
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments",
    )

    @model_validator(mode="after")
    def validate_constant_has_no_source(self) -> "FieldTransformationPlan":
        if (
            self.collapse
            and self.collapse.strategy == CollapseStrategy.constant
            and self.source_columns
        ):
            raise ValueError(
                "constant collapse strategy implies no source columns — "
                "source_columns must be empty for constant-derived fields"
            )
        return self


class TransformationMap(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
    institution_id: str = Field(..., description="Institution identifier")
    entity_type: EntityType = Field(..., description="cohort or course")
    target_schema: str = Field(..., description="Target schema name")
    pre_collapse: Optional[PreCollapseConfig] = Field(
        default=None,
        description=(
            "DataFrame-level grain reduction applied before any field plans run. "
            "Required when source grain is finer than target schema grain — "
            "e.g. section-level course rows needing reduction to course-level. "
            "Cohort maps typically do not need this; field-level CollapseConfig handles "
            "student-term to student reduction per field."
        ),
    )
    plans: List[FieldTransformationPlan] = Field(
        ...,
        min_length=1,
        description="Per-target-field transformation plans",
    )

    @field_validator("plans")
    @classmethod
    def target_fields_must_be_unique(
        cls, v: List[FieldTransformationPlan]
    ) -> List[FieldTransformationPlan]:
        targets = [p.target_field for p in v]
        if len(targets) != len(set(targets)):
            raise ValueError("Each target_field must appear only once in the transformation map")
        return v