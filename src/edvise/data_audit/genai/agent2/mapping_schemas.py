from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator


# -----------------------------
# Shared enums
# -----------------------------

class EntityType(str, Enum):
    cohort = "cohort"
    course = "course"


class ReviewStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    corrected = "corrected"
    rejected = "rejected"


# -----------------------------
# Approved utility names for 2b
# Expand this as your library grows.
# -----------------------------

class UtilityName(str, Enum):
    normalize_columns = "normalize_columns"
    cast_nullable_dtype = "cast_nullable_dtype"
    cast_nullable_int = "cast_nullable_int"
    cast_nullable_float = "cast_nullable_float"
    cast_string = "cast_string"
    coerce_numeric = "coerce_numeric"
    coerce_datetime = "coerce_datetime"
    combine_columns = "combine_columns"
    map_values = "map_values"
    normalize_term_code = "normalize_term_code"
    normalize_grade = "normalize_grade"
    strip_whitespace = "strip_whitespace"
    lowercase = "lowercase"
    uppercase = "uppercase"
    fill_nulls = "fill_nulls"
    replace_null_tokens = "replace_null_tokens"
    deduplicate_rows = "deduplicate_rows"
    new_utility_needed = "NEW_UTILITY_NEEDED"


# -----------------------------
# Base config
# -----------------------------

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# -----------------------------
# 2a: Field mapping manifest
# -----------------------------

class FieldMappingRecord(StrictBaseModel):
    target_field: str = Field(..., description="Target Edvise schema field")
    source_columns: List[str] = Field(
        ...,
        min_length=1,
        description="One or more source columns proposed for this target field",
    )
    source_table: str = Field(
        ...,
        description="Source table name (e.g., 'student_df', 'course_df', 'stems_def_df') to disambiguate columns",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in the proposed mapping",
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Short explanation for why the mapping was selected",
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
    def source_columns_must_not_be_empty_strings(cls, v: List[str]) -> List[str]:
        if not v or any(not col.strip() for col in v):
            raise ValueError("source_columns must contain at least one non-empty string")
        return v

    @field_validator("corrected_source_columns")
    @classmethod
    def corrected_source_columns_nonempty(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None and any(not col.strip() for col in v):
            raise ValueError("corrected_source_columns cannot contain empty strings")
        return v


class FieldMappingManifest(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
    institution_id: str = Field(..., description="Institution identifier")
    entity_type: EntityType = Field(..., description="Entity type being mapped")
    target_schema: str = Field(..., description="Target schema name")
    mappings: List[FieldMappingRecord] = Field(
        ...,
        min_length=1,
        description="Per-target-field mapping proposals",
    )

    @field_validator("mappings")
    @classmethod
    def target_fields_must_be_unique(cls, v: List[FieldMappingRecord]) -> List[FieldMappingRecord]:
        target_fields = [m.target_field for m in v]
        if len(target_fields) != len(set(target_fields)):
            raise ValueError("Each target_field must appear only once in the manifest")
        return v


# -----------------------------
# 2b: Transformation map
# -----------------------------

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


# -----------------------------
# Shared config
# -----------------------------

class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
        str_strip_whitespace=True,
    )


class ReviewStatus(str, Enum):
    proposed = "proposed"
    approved = "approved"
    corrected = "corrected"
    rejected = "rejected"


class EntityType(str, Enum):
    cohort = "cohort"
    course = "course"


# -----------------------------
# Collapse config
# Declares how to reduce student-term grain to student grain per field.
# The executor uses schema_contract.unique_keys as the groupby keys,
# and CollapseConfig to select the right row within each group.
# -----------------------------

class CollapseStrategy(str, Enum):
    any_row = "any_row"
    # Invariant fields - value is the same across all term rows per student.
    # Executor takes first row after groupby (arbitrary but deterministic).
    # Examples: gender, race, enrollment_type, entry_hs_gpa

    first_by = "first_by"
    # Term-1 specific fields - take first row ordered by `order_by` column ascending.
    # Requires `order_by` to be set.
    # Examples: program_of_study_term_1, cohort_term, cohort

    where_not_null = "where_not_null"
    # Take first non-null row for `condition_col` per student.
    # Requires `condition_col` to be set.
    # Examples: first_bachelors_grad_date (where degree_earned_term is not null),
    #           major_grad (where degree_earned_term is not null)

    constant = "constant"
    # No row selection needed — field is derived as a constant value for all rows.
    # source_columns should be empty; value is expressed in the transformation steps.
    # Examples: credential_type_sought_year_1 at UCF (always "Bachelor's")

    none = "none"
    # No collapse needed — source data is already at the correct grain.
    # Used for all course-level fields and any student fields already at student grain.


class CollapseConfig(StrictBaseModel):
    strategy: CollapseStrategy = Field(
        ...,
        description="Row selection strategy for grain reduction"
    )
    order_by: Optional[str] = Field(
        default=None,
        description="Column to order by ascending before taking first row. Required for first_by strategy."
    )
    condition_col: Optional[str] = Field(
        default=None,
        description="Column that must be non-null for row selection. Required for where_not_null strategy."
    )

    @model_validator(mode="after")
    def validate_strategy_args(self) -> "CollapseConfig":
        if self.strategy == CollapseStrategy.first_by and not self.order_by:
            raise ValueError("order_by is required when strategy is first_by")
        if self.strategy == CollapseStrategy.where_not_null and not self.condition_col:
            raise ValueError("condition_col is required when strategy is where_not_null")
        return self


# -----------------------------
# Typed step models
# Each maps 1:1 to a callable in transformation_utilities.py.
# `column` arg is always explicit — executor does not infer target column.
# -----------------------------

class CastNullableIntStep(StrictBaseModel):
    function_name: Literal["cast_nullable_int"]
    column: str = Field(..., description="Column to cast")
    rationale: Optional[str] = None


class CastNullableFloatStep(StrictBaseModel):
    function_name: Literal["cast_nullable_float"]
    column: str = Field(..., description="Column to cast")
    rationale: Optional[str] = None


class CastStringStep(StrictBaseModel):
    function_name: Literal["cast_string"]
    column: str = Field(..., description="Column to cast")
    rationale: Optional[str] = None


class CastBooleanStep(StrictBaseModel):
    function_name: Literal["cast_boolean"]
    column: str = Field(..., description="Column to cast")
    boolean_map: Optional[Dict[str, bool]] = None
    rationale: Optional[str] = None


class CastDatetimeStep(StrictBaseModel):
    function_name: Literal["cast_datetime"]
    column: str = Field(..., description="Column to cast")
    rationale: Optional[str] = None


class CoerceNumericStep(StrictBaseModel):
    function_name: Literal["coerce_numeric"]
    column: str = Field(..., description="Column to coerce")
    rationale: Optional[str] = None


class CoerceDatetimeStep(StrictBaseModel):
    function_name: Literal["coerce_datetime"]
    column: str = Field(..., description="Column to coerce")
    fmt: Optional[str] = Field(
        default=None,
        description="Optional strptime format string. If None, pandas infers."
    )
    rationale: Optional[str] = None


class StripWhitespaceStep(StrictBaseModel):
    function_name: Literal["strip_whitespace"]
    column: str = Field(..., description="Column to strip")
    rationale: Optional[str] = None


class LowercaseStep(StrictBaseModel):
    function_name: Literal["lowercase"]
    column: str = Field(..., description="Column to lowercase")
    rationale: Optional[str] = None


class UppercaseStep(StrictBaseModel):
    function_name: Literal["uppercase"]
    column: str = Field(..., description="Column to uppercase")
    rationale: Optional[str] = None


class MapValuesStep(StrictBaseModel):
    function_name: Literal["map_values"]
    column: str = Field(..., description="Column to map")
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
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class NormalizeGradeStep(StrictBaseModel):
    function_name: Literal["normalize_grade"]
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class NormalizeEnrollmentStep(StrictBaseModel):
    function_name: Literal["normalize_enrollment"]
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class NormalizePellStep(StrictBaseModel):
    function_name: Literal["normalize_pell"]
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class NormalizeCredentialStep(StrictBaseModel):
    function_name: Literal["normalize_credential"]
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class NormalizeStudentAgeStep(StrictBaseModel):
    function_name: Literal["normalize_student_age"]
    column: str = Field(..., description="Column to normalize")
    rationale: Optional[str] = None


class FillNullsStep(StrictBaseModel):
    function_name: Literal["fill_nulls"]
    column: str = Field(..., description="Column to fill")
    value: Any = Field(..., description="Fill value")
    rationale: Optional[str] = None


class ReplaceNullTokensStep(StrictBaseModel):
    function_name: Literal["replace_null_tokens"]
    column: str = Field(..., description="Column to clean")
    null_tokens: List[str] = Field(..., description="Token strings to replace with NA")
    rationale: Optional[str] = None


class ReplaceValuesWithNullStep(StrictBaseModel):
    function_name: Literal["replace_values_with_null"]
    column: str = Field(..., description="Column to clean")
    to_replace: Union[str, List[str]] = Field(..., description="Values to replace with null")
    rationale: Optional[str] = None


class CombineColumnsStep(StrictBaseModel):
    function_name: Literal["combine_columns"]
    cols: List[str] = Field(..., min_length=2, description="Columns to combine")
    output_col: str = Field(..., description="Output column name")
    sep: str = Field(default="", description="Separator string")
    rationale: Optional[str] = None


class DeduplicateRowsStep(StrictBaseModel):
    function_name: Literal["deduplicate_rows"]
    subset: Optional[List[str]] = Field(
        default=None,
        description="Columns to consider for deduplication. None = all columns."
    )
    keep: Literal["first", "last", "none"] = Field(
        default="first",
        description="Which duplicate to keep"
    )
    rationale: Optional[str] = None


class StripTrailingDecimalStep(StrictBaseModel):
    function_name: Literal["strip_trailing_decimal"]
    column: str = Field(..., description="Column to strip trailing .0 from")
    rationale: Optional[str] = None


class NewUtilityNeededStep(StrictBaseModel):
    function_name: Literal["NEW_UTILITY_NEEDED"]
    description: str = Field(
        ...,
        description="Description of the required utility and its expected behavior"
    )
    rationale: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_gap(self) -> bool:
        return True


# Discriminated union — executor dispatches on function_name
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
    NewUtilityNeededStep,
]


# -----------------------------
# Field transformation plan
# -----------------------------

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
        description="Expected output dtype after all transformations"
    )
    collapse: Optional[CollapseConfig] = Field(
        default=None,
        description=(
            "Row collapse strategy for grain reduction from student-term to student. "
            "None = no collapse needed (course-level fields or already correct grain). "
            "Executor uses schema_contract.unique_keys as groupby keys."
        )
    )
    steps: List[TransformationStep] = Field(
        default_factory=list,
        description="Ordered transformation steps. Empty = unmappable field, no steps to execute.",
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.proposed,
        description="Human review outcome",
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments on the plan",
    )

    @model_validator(mode="after")
    def validate_constant_collapse_has_no_source(self) -> "FieldTransformationPlan":
        if (
            self.collapse
            and self.collapse.strategy == CollapseStrategy.constant
            and self.source_columns
        ):
            raise ValueError(
                "constant collapse strategy implies no source columns — "
                "source_columns should be empty for constant-derived fields"
            )
        return self


# -----------------------------
# Transformation map
# -----------------------------

class TransformationMap(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
    institution_id: str = Field(..., description="Institution identifier")
    entity_type: EntityType = Field(..., description="Entity type being transformed")
    target_schema: str = Field(..., description="Target schema name")
    plans: List[FieldTransformationPlan] = Field(
        ...,
        min_length=1,
        description="Per-target-field transformation plans",
    )

    @field_validator("plans")
    @classmethod
    def target_fields_must_be_unique(cls, v: List[FieldTransformationPlan]) -> List[FieldTransformationPlan]:
        target_fields = [p.target_field for p in v]
        if len(target_fields) != len(set(target_fields)):
            raise ValueError("Each target_field must appear only once in the transformation map")
        return v