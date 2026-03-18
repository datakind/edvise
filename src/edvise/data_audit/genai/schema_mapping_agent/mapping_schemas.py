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
    pending = "pending"
    approved = "approved"
    corrected = "corrected"
    rejected = "rejected"


# =============================================================================
# 2a — Field Mapping Manifest
# =============================================================================

class JoinFilter(StrictBaseModel):
    """
    Structured filter applied to the lookup table before joining.

    Examples:
        {"column": "awarded_degree", "operator": "contains", "value": "Associate"}
        {"column": "awarded_degree", "operator": "isin",
         "value": ["Certification", "Certificate - TSI Liable"]}
    """
    column: str
    operator: Literal["contains", "equals", "startswith", "isin"]
    value: Union[str, List[str]]

    @model_validator(mode="after")
    def validate_isin_is_list(self) -> "JoinFilter":
        if self.operator == "isin" and not isinstance(self.value, list):
            raise ValueError("isin operator requires value to be a list")
        if self.operator != "isin" and isinstance(self.value, list):
            raise ValueError("list value only valid for isin operator")
        return self


class JoinConfig(StrictBaseModel):
    """
    Cross-table join declaration on a FieldMappingRecord.

    Purely a join key declaration — row selection logic (which row to keep,
    ordering, filtering) lives in RowSelectionConfig on the parent record.

    The field executor uses this to merge base_table ← lookup_table on join_keys,
    then applies RowSelectionConfig to select the correct row.
    """
    base_table: str = Field(..., description="Driving table (entity base table)")
    lookup_table: str = Field(..., description="Table to join and pull value from")
    join_keys: List[str] = Field(
        ...,
        description=(
            "Canonical join key column names. Must exist in both base and lookup "
            "tables after column_aliases are applied."
        ),
    )


class RowSelectionStrategy(str, Enum):
    any_row = "any_row"
    # Value is invariant across all candidate rows — take any.
    # Examples: gender, race, student_id (same-table);
    #           term_major, term_enrollment_intensity (cross-table)

    first_by = "first_by"
    # Take first row when sorted ascending by order_by.
    # Requires order_by to be set.
    # Examples: program_of_study_term_1, cohort_term (same-table);
    #           first_associates_grad_date (cross-table with order_by: term_order)

    where_not_null = "where_not_null"
    # Take first row where condition_col is non-null.
    # Requires condition_col to be set.
    # Examples: first_bachelors_grad_date, major_grad (same-table)

    constant = "constant"
    # No row selection — field is derived as a constant value for all rows.
    # source_column must be null.
    # Examples: credential_type_sought_year_1 at UCF

    nth = "nth"
    # Take nth matching row ordered by order_by (1-based).
    # Requires n and order_by to be set.
    # Examples: certificate2_date, certificate3_date (cross-table)


class RowSelectionConfig(StrictBaseModel):
    """
    Unified row selection config for both same-table and cross-table fields.

    Replaces CollapseConfig (which only handled same-table cohort fields) and
    JoinConfig.keep (which only handled cross-table fields). Now a single concept
    applies to both cases — the field executor applies it after resolving the
    source DataFrame (with or without a join).

    filter: Applied to the source/lookup DataFrame before row selection.
            Typically used with cross-table degree/certificate fields to subset
            to relevant rows (e.g. awarded_degree contains 'Associate').
    """
    strategy: RowSelectionStrategy
    order_by: Optional[str] = Field(
        default=None,
        description="Column to sort ascending before row selection. Required for first_by and nth.",
    )
    condition_col: Optional[str] = Field(
        default=None,
        description="Column that must be non-null. Required for where_not_null.",
    )
    filter: Optional[JoinFilter] = Field(
        default=None,
        description="Pre-selection filter applied to source/lookup DataFrame rows.",
    )
    n: Optional[int] = Field(
        default=None,
        description="Row index (1-based) to select. Required for nth strategy.",
    )

    @model_validator(mode="after")
    def validate_strategy_args(self) -> "RowSelectionConfig":
        if self.strategy == RowSelectionStrategy.first_by and not self.order_by:
            raise ValueError("order_by is required when strategy is first_by")
        if self.strategy == RowSelectionStrategy.where_not_null and not self.condition_col:
            raise ValueError("condition_col is required when strategy is where_not_null")
        if self.strategy == RowSelectionStrategy.nth:
            if not self.n:
                raise ValueError("n is required when strategy is nth")
            if not self.order_by:
                raise ValueError("order_by is required when strategy is nth")
        return self

    @property
    def fan_out_risk(self) -> bool:
        """True when multiple rows may match and selection is non-trivial."""
        return self.strategy in (
            RowSelectionStrategy.first_by,
            RowSelectionStrategy.nth,
            RowSelectionStrategy.where_not_null,
        )


class FieldMappingRecord(StrictBaseModel):
    """
    Complete sourcing specification for a single target field.

    The manifest record is the single source of truth for:
      - Which column to pull (source_column)
      - Which table it comes from (source_table)
      - How to join to get there if cross-table (join)
      - Which row to select (row_selection)

    The transformation plan only declares what to do with the resolved Series —
    it has no implicit dependency on the manifest beyond receiving the Series.
    """
    target_field: str = Field(..., description="Target Edvise schema field")
    source_column: Optional[str] = Field(
        default=None,
        description=(
            "Single source column to pull. None = unmappable field or constant derivation. "
            "For cross-table fields this is the column in the lookup table."
        ),
    )
    source_table: Optional[str] = Field(
        default=None,
        description=(
            "Source table name. For cross-table fields this is the lookup table — "
            "join.base_table is the driving table. None = unmappable or constant field."
        ),
    )
    join: Optional[JoinConfig] = Field(
        default=None,
        description=(
            "Cross-table join declaration. Required when source_table differs from "
            "the entity base table. None = source_table is the base table, no join needed."
        ),
    )
    row_selection: Optional[RowSelectionConfig] = Field(
        default=None,
        description=(
            "Row selection strategy. Required for all mappable fields — declares "
            "how to reduce multiple candidate rows to a single value. "
            "None only for unmappable fields (source_column is also None)."
        ),
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
        description="Explanation for the mapping decision or why the field is unmappable.",
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description=(
            "Predicted Pandera validation risks — e.g. regex pattern mismatches, "
            "nulls on non-nullable fields, categorical values not in allowed set. "
            "Null if no validation risk identified."
        ),
    )
    review_status: ReviewStatus = Field(
        default=ReviewStatus.pending,
        description=(
            "Human review outcome. Agent always outputs 'pending'. "
            "'approved' is only set after human review at the HITL gate."
        ),
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments or corrections",
    )
    corrected_source_column: Optional[str] = Field(
        default=None,
        description="Reviewer-corrected source column, if applicable",
    )

    @model_validator(mode="after")
    def validate_sourcing_consistency(self) -> "FieldMappingRecord":
        has_source = self.source_column is not None
        has_table = self.source_table is not None

        if has_source and not has_table:
            raise ValueError(
                "source_table must be set when source_column is set"
            )
        if self.join is not None and self.source_table is None:
            raise ValueError(
                "source_table must be set when join is declared — "
                "source_table is the lookup table, join.base_table is the driving table"
            )
        if self.row_selection is not None and not has_source:
            if self.row_selection.strategy != RowSelectionStrategy.constant:
                raise ValueError(
                    "row_selection requires source_column except for constant strategy"
                )
        return self


class ColumnAlias(StrictBaseModel):
    table: str = Field(..., description="Source table containing the aliased column")
    source_column: str = Field(
        ...,
        description="Column name as it appears in the source table",
    )
    canonical_column: str = Field(
        ...,
        description=(
            "Canonical column name for join key matching. "
            "Typically the normalized form shared across tables."
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
        description="Per-target-field mapping records",
    )
    column_aliases: List[ColumnAlias] = Field(
        default_factory=list,
        description=(
            "Cross-table column name aliases. Captures cases where the same concept "
            "appears under different names across source tables. "
            "Consumed by field executor for join key matching."
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
# Transformation plans are pure value transformation recipes.
# No sourcing logic here — all data sourcing lives in the manifest.
# The field executor resolves the source Series from the manifest record
# and passes it to the transformation steps.
# -----------------------------------------------------------------------------

class TransformationStep(StrictBaseModel):
    pass  # base — concrete step models below


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


class NormalizeYearRangeStep(StrictBaseModel):
    function_name: Literal["normalize_year_range"]
    column: str
    rationale: Optional[str] = None


class ExtractYearStep(StrictBaseModel):
    function_name: Literal["extract_year"]
    column: str
    rationale: Optional[str] = None


class ParseYyyymmStep(StrictBaseModel):
    function_name: Literal["parse_yyyymm"]
    column: str
    rationale: Optional[str] = None


class ParseTermDescriptionStep(StrictBaseModel):
    function_name: Literal["parse_term_description"]
    column: str
    rationale: Optional[str] = None


class ExtractAcademicYearFromTermCodeStep(StrictBaseModel):
    function_name: Literal["extract_academic_year_from_term_code"]
    column: str
    rationale: Optional[str] = None


class ExtractTermSeasonFromTermCodeStep(StrictBaseModel):
    function_name: Literal["extract_term_season_from_term_code"]
    column: str
    rationale: Optional[str] = None


class ParseTermCodeToDatetimeStep(StrictBaseModel):
    function_name: Literal["parse_term_code_to_datetime"]
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


class NewUtilityNeededStep(StrictBaseModel):
    function_name: Literal["NEW_UTILITY_NEEDED"]
    description: str
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
    StripTrailingDecimalStep,
    FillConstantStep,
    NormalizeYearRangeStep,
    ExtractYearStep,
    ParseYyyymmStep,
    ParseTermDescriptionStep,
    ExtractAcademicYearFromTermCodeStep,
    ExtractTermSeasonFromTermCodeStep,
    ParseTermCodeToDatetimeStep,
    BirthyearToAgeBucketStep,
    ConditionalCreditsStep,
    NewUtilityNeededStep,
]


# -----------------------------------------------------------------------------
# Field transformation plan + map
# -----------------------------------------------------------------------------

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
    review_status: ReviewStatus = Field(
        default=ReviewStatus.pending,
        description="Human review outcome",
    )
    reviewer_notes: Optional[str] = Field(
        default=None,
        description="Reviewer comments",
    )


class TransformationMap(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
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


def get_manifest_schema_context() -> str:
    """
    Returns a focused schema reference for Agent 2a prompt context.
    Covers only the models relevant to manifest generation —
    RowSelectionStrategy, RowSelectionConfig, JoinFilter, JoinConfig,
    and FieldMappingRecord. Excludes transformation map models.
    """
    import inspect
    models = [
        JoinFilter,
        JoinConfig,
        RowSelectionStrategy,
        RowSelectionConfig,
        FieldMappingRecord,
    ]
    sections = []
    for model in models:
        source = inspect.getsource(model)
        sections.append(source)
    return "\n\n".join(sections)