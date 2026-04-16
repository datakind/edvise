"""Pydantic models for Step 2a — field mapping manifest (sourcing spec)."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Dict, List, Literal, Optional, Union

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
    """
    Post-hoc pipeline / HITL telemetry for a mapping record or transformation plan.

    Assigned by the pipeline after deterministic validation (and optional refinement),
    not by the initial Schema Mapping Agent LLM call.
    """

    auto_approved = "auto_approved"
    # Passed deterministic validation and confidence threshold.
    # Set by the pipeline without human involvement.

    refined_by_llm = "refined_by_llm"
    # Refinement LLM corrected a validation error or low confidence field.
    # Original agent output was wrong; no human involvement.

    proposed_for_hitl = "proposed_for_hitl"
    # Refinement LLM could not fix — sent to human reviewer at HITL gate.
    # Replaces the old "pending" state.

    corrected_by_hitl = "corrected_by_hitl"
    # Human reviewer made a correction at the HITL gate.


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
        if (
            self.strategy == RowSelectionStrategy.where_not_null
            and not self.condition_col
        ):
            raise ValueError(
                "condition_col is required when strategy is where_not_null"
            )
        if self.strategy == RowSelectionStrategy.nth:
            if self.n is None:
                raise ValueError("n is required when strategy is nth")
            if self.n < 1:
                raise ValueError(
                    "n must be >= 1 (1-based index after sorting) when strategy is nth"
                )
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
    review_status: ReviewStatus | None = Field(
        default=None,
        description=(
            "Pipeline-assigned review outcome. Never set by the generating agent. "
            "Assigned after validation: auto_approved, refined_by_llm, "
            "proposed_for_hitl, or corrected_by_hitl."
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
            raise ValueError("source_table must be set when source_column is set")
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


class MappingManifestEnvelope(StrictBaseModel):
    schema_version: str = Field(default="0.1.0")
    institution_id: str = Field(..., description="Institution identifier")
    manifests: Dict[EntityType, FieldMappingManifest] = Field(
        ...,
        description="Per-entity field mapping manifests (cohort and/or course).",
    )


# =============================================================================

_AGENT_EXCLUDED_FMR_SOURCE_FIELDS = frozenset(
    {"review_status", "reviewer_notes", "corrected_source_column"}
)


def _omit_field_blocks_from_class_source(
    source: str, field_names: frozenset[str]
) -> str:
    """
    Drop class attribute blocks starting with `    name:` ... closing `    )`
    from inspect.getsource output. Used to hide pipeline/reviewer-only fields
    from agent-facing schema dumps.
    """
    lines = source.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        skip = any(line.startswith(f"    {name}:") for name in field_names)
        if skip:
            i += 1
            while i < n:
                if lines[i].rstrip() == "    )":
                    i += 1
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "".join(out)


def _field_mapping_record_source_for_agent_prompt() -> str:
    return _omit_field_blocks_from_class_source(
        inspect.getsource(FieldMappingRecord), _AGENT_EXCLUDED_FMR_SOURCE_FIELDS
    )


def get_manifest_schema_context() -> str:
    """
    Returns a focused schema reference for Agent 2a prompt context.
    Covers only the models relevant to manifest generation —
    RowSelectionStrategy, RowSelectionConfig, JoinFilter, JoinConfig,
    FieldMappingRecord, ColumnAlias, FieldMappingManifest, and MappingManifestEnvelope.
    Excludes transformation map models.
    """
    models = [
        JoinFilter,
        JoinConfig,
        RowSelectionStrategy,
        RowSelectionConfig,
        FieldMappingRecord,
        ColumnAlias,
        FieldMappingManifest,
        MappingManifestEnvelope,
    ]
    sections = []
    for model in models:
        if model is FieldMappingRecord:
            source = _field_mapping_record_source_for_agent_prompt()
        else:
            source = inspect.getsource(model)
        sections.append(source)
    return "\n\n".join(sections)


def get_compact_manifest_schema_reference() -> str:
    """
    Condensed manifest schema for LLM prompts (~600 tokens): field names, types,
    required (!) / optional (?), no class bodies. For full Pydantic definitions use
    get_manifest_schema_context().
    """
    return (
        "JoinFilter — Filter rows on lookup/source before join/row selection.\n"
        'JoinFilter: {column: str!, operator: "contains"|"equals"|"startswith"|"isin"!, '
        "value: str|List[str]!}\n"
        "\n"
        "JoinConfig — Drive base_table ← lookup_table on shared join_keys (canonical names after column_aliases).\n"
        "JoinConfig: {base_table: str!, lookup_table: str!, join_keys: List[str]!}\n"
        "\n"
        "RowSelectionStrategy — Pick one row from multiple candidates.\n"
        'RowSelectionStrategy: "any_row"|"first_by"|"where_not_null"|"constant"|"nth"\n'
        "\n"
        "RowSelectionConfig — Strategy-specific args: first_by→order_by!; where_not_null→condition_col!; "
        "nth→n!(int≥1)+order_by!; constant→no row pick (source_column null); filter pre-filters rows.\n"
        "RowSelectionConfig: {strategy: RowSelectionStrategy!, order_by?: str, condition_col?: str, "
        "filter?: JoinFilter, n?: int}; if strategy=nth then n and order_by are REQUIRED\n"
        "\n"
        "FieldMappingRecord — One target field: source, optional join, row_selection, confidence.\n"
        "FieldMappingRecord: {target_field: str!, source_column?: str, source_table?: str, join?: JoinConfig, "
        "row_selection?: RowSelectionConfig, confidence: float[0,1]!, rationale?: str, validation_notes?: str}\n"
        "\n"
        "ColumnAlias — Map source column name to canonical join key name for a table.\n"
        "ColumnAlias: {table: str!, source_column: str!, canonical_column: str!, rationale?: str}\n"
        "\n"
        "FieldMappingManifest — Per-entity mapping bundle; mappings unique by target_field.\n"
        'FieldMappingManifest: {entity_type: "cohort"|"course"!, target_schema: str!, '
        "mappings: List[FieldMappingRecord]!(len≥1), column_aliases?: List[ColumnAlias]}\n"
        "\n"
        "MappingManifestEnvelope — Root document; manifests keyed by entity type.\n"
        'MappingManifestEnvelope: {schema_version: str (default "0.1.0"), institution_id: str!, '
        'manifests: Record<"cohort"|"course", FieldMappingManifest>!}'
    )
