"""
Deterministic validation layer for Agent 2a field mapping manifests.

Runs after the LLM produces a MappingManifestEnvelope and before the
refinement+HITL pass. Catches structural rule violations that the LLM
cannot reliably self-detect (confident-but-wrong errors).

Public API:
    validate_manifest(manifest, schema_contract)
        → list[ManifestValidationError]

Input contracts:
    manifest:         FieldMappingManifest (one entity — cohort or course)
    schema_contract:  EnrichedSchemaContractForSMA (IdentityAgent output)
                      Column existence checked via training.column_details[].normalized_name.
                      Join key candidates derived from unique_keys per dataset.

ValidationError groups:
    COLUMN_EXISTENCE   — source column / table not found in schema contract
    JOIN_STRUCTURE     — join declaration is missing, incorrect, or unresolvable
    ROW_SELECTION      — strategy references a column that doesn't exist
    MAP_UNMAP          — sourcing fields present on unmapped record or vice versa
    ENTITY_GRAIN       — Pandera uniqueness grain field missing or unmapped in manifest

Usage:
    errors = validate_manifest(manifest, schema_contract)
    # Feed errors into refinement+HITL LLM call as structured context
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from edvise.genai.mapping.shared.schema_contract.schemas import (
    EnrichedSchemaContractForSMA,
)

from .schemas import ColumnAlias, FieldMappingManifest, FieldMappingRecord

# ---------------------------------------------------------------------------
# Schema contract helpers
# ---------------------------------------------------------------------------


def _observed_columns(
    schema_contract: EnrichedSchemaContractForSMA,
    table: str,
) -> set[str]:
    """
    Return the set of normalized column names observed in the raw data for a table.
    Sourced from training.column_details[].normalized_name.
    Returns empty set if table is not present.
    """
    dataset = schema_contract.datasets.get(table)
    if dataset is None:
        return set()
    return {cd.normalized_name for cd in dataset.training.column_details}


def _has_table(
    schema_contract: EnrichedSchemaContractForSMA,
    table: str,
) -> bool:
    return table in schema_contract.datasets


def _has_column(
    schema_contract: EnrichedSchemaContractForSMA,
    table: str,
    column: str,
) -> bool:
    return column in _observed_columns(schema_contract, table)


# ---------------------------------------------------------------------------
# ManifestValidationError
# ---------------------------------------------------------------------------


class ManifestValidationErrorCode(str, Enum):
    # COLUMN_EXISTENCE
    SOURCE_TABLE_NOT_FOUND = "SOURCE_TABLE_NOT_FOUND"
    # source_table not present in schema_contract.datasets.
    # Check: _has_table(schema_contract, record.source_table)

    SOURCE_COLUMN_NOT_FOUND = "SOURCE_COLUMN_NOT_FOUND"
    # source_column not present in training.column_details for source_table.
    # Check: _has_column(schema_contract, record.source_table, record.source_column)

    # JOIN_STRUCTURE
    JOIN_BASE_TABLE_NOT_FOUND = "JOIN_BASE_TABLE_NOT_FOUND"
    # join.base_table not present in schema_contract.datasets.
    # Check: _has_table(schema_contract, record.join.base_table)

    JOIN_LOOKUP_TABLE_NOT_FOUND = "JOIN_LOOKUP_TABLE_NOT_FOUND"
    # join.lookup_table not present in schema_contract.datasets.
    # Check: _has_table(schema_contract, record.join.lookup_table)

    JOIN_KEY_NOT_IN_BASE_TABLE = "JOIN_KEY_NOT_IN_BASE_TABLE"
    # A join key is not present in base_table columns after alias resolution.
    # Check: for each key in join.join_keys →
    #   resolve via column_aliases for base_table →
    #   _has_column(schema_contract, base_table, resolved_key)

    JOIN_KEY_NOT_IN_LOOKUP_TABLE = "JOIN_KEY_NOT_IN_LOOKUP_TABLE"
    # A join key is not present in lookup_table columns after alias resolution.
    # Check: for each key in join.join_keys →
    #   resolve via column_aliases for lookup_table →
    #   _has_column(schema_contract, lookup_table, resolved_key)

    MISSING_COLUMN_ALIAS = "MISSING_COLUMN_ALIAS"
    # Join keys don't match directly across base and lookup tables AND no
    # ColumnAlias bridges them. Model produced a join but forgot to declare the alias.
    # Check: join keys don't share a name across tables →
    #   scan manifest.column_aliases for entries that resolve the mismatch →
    #   if none found, raise this error.

    JOIN_DECLARED_ON_SAME_TABLE = "JOIN_DECLARED_ON_SAME_TABLE"
    # join is declared but join.base_table == join.lookup_table.
    # No join needed — same-table field.
    # Check: record.join is not None and record.join.base_table == record.join.lookup_table

    # ROW_SELECTION
    ROW_SELECTION_ORDER_BY_NOT_FOUND = "ROW_SELECTION_ORDER_BY_NOT_FOUND"
    # row_selection.order_by column not present in schema contract for source_table.
    # Applies to first_by and nth strategies.
    # Check: _has_column(schema_contract, record.source_table, record.row_selection.order_by)

    ROW_SELECTION_CONDITION_COL_NOT_FOUND = "ROW_SELECTION_CONDITION_COL_NOT_FOUND"
    # row_selection.condition_col not present in schema contract for source_table.
    # Applies to where_not_null strategy.
    # Check: _has_column(schema_contract, record.source_table, record.row_selection.condition_col)

    ROW_SELECTION_FILTER_COL_NOT_FOUND = "ROW_SELECTION_FILTER_COL_NOT_FOUND"
    # row_selection.filter.column not present in schema contract for source_table.
    # Check: _has_column(schema_contract, record.source_table, record.row_selection.filter.column)

    # MAP_UNMAP
    MAPPED_FIELD_MISSING_SOURCE = "MAPPED_FIELD_MISSING_SOURCE"
    # join is declared but source_column is None.
    # Pydantic catches source_column+source_table consistency, but not
    # the case where join is declared without source_column.
    # Check: record.join is not None and record.source_column is None

    UNMAPPED_FIELD_HAS_SOURCE = "UNMAPPED_FIELD_HAS_SOURCE"
    # source_column is None (unmapped) but source_table or join is still populated.
    # Indicates the model partially filled an unmappable field.
    # Check: record.source_column is None and (record.source_table is not None
    #        or record.join is not None)

    # ENTITY_GRAIN (target Pandera schema Config.unique → manifest source columns)
    GRAIN_KEY_MISSING_SOURCE = "GRAIN_KEY_MISSING_SOURCE"
    # A field in the target schema's uniqueness grain has no mappable source column.
    # Execution needs source_column (or corrected_source_column) for each grain key.
    # Check: for each field in schema.Config.unique for manifest.target_schema,
    #   manifest has a mapping with non-null effective source.


class ManifestValidationError(BaseModel):
    """
    One rule violation on one field in the manifest.

    Consumed by the refinement+HITL LLM call as structured context.
    Also written to the institution's validation_errors.json for audit.
    """

    target_field: str = Field(
        ...,
        description="The FieldMappingRecord.target_field this error applies to.",
    )
    error_code: ManifestValidationErrorCode = Field(
        ...,
        description="Machine-readable error code for routing in refinement prompt.",
    )
    detail: str = Field(
        ...,
        description=(
            "Human-readable description of the specific violation. "
            "Should name the offending value (column name, table name, key) "
            "so the refinement LLM has precise context."
        ),
    )
    offending_value: str | None = Field(
        default=None,
        description=(
            "The specific value that failed the check — e.g. the column name "
            "that wasn't found, the join key that couldn't be resolved. "
            "Null if the error is about absence rather than a wrong value."
        ),
    )


# ---------------------------------------------------------------------------
# Alias resolution helpers
# ---------------------------------------------------------------------------


def _resolve_column_via_aliases(
    table: str,
    column: str,
    column_aliases: list[ColumnAlias],
) -> str:
    """
    Return canonical_column if a ColumnAlias entry exists for (table, column),
    otherwise return column unchanged.
    """
    for alias in column_aliases:
        if alias.table == table and alias.source_column == column:
            return alias.canonical_column
    return column


def _alias_bridges_join(
    base_table: str,
    lookup_table: str,
    join_key: str,
    column_aliases: list[ColumnAlias],
) -> bool:
    """
    Returns True if column_aliases resolves join_key to the same canonical name
    across base_table and lookup_table.
    """
    base_resolved = _resolve_column_via_aliases(base_table, join_key, column_aliases)
    lookup_resolved = _resolve_column_via_aliases(
        lookup_table, join_key, column_aliases
    )
    return base_resolved == lookup_resolved


# ---------------------------------------------------------------------------
# Rule implementations
# ---------------------------------------------------------------------------


def _check_column_existence(
    record: FieldMappingRecord,
    schema_contract: EnrichedSchemaContractForSMA,
    errors: list[ManifestValidationError],
) -> None:
    """
    GROUP: COLUMN_EXISTENCE

    Rules:
      1. source_table must exist in schema_contract.datasets.
      2. source_column must appear in training.column_details for source_table.

    Skips unmapped fields (source_column and source_table are both None).
    """
    if record.source_table is None and record.source_column is None:
        return  # unmapped — handled by MAP_UNMAP checks

    if record.source_table is not None:
        if not _has_table(schema_contract, record.source_table):
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.SOURCE_TABLE_NOT_FOUND,
                    detail=(
                        f"source_table '{record.source_table}' not found in schema contract. "
                        f"Available tables: {sorted(schema_contract.datasets.keys())}"
                    ),
                    offending_value=record.source_table,
                )
            )
            return  # no point checking column if table is missing

    if record.source_column is not None and record.source_table is not None:
        if not _has_column(schema_contract, record.source_table, record.source_column):
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.SOURCE_COLUMN_NOT_FOUND,
                    detail=(
                        f"source_column '{record.source_column}' not found in table "
                        f"'{record.source_table}'. "
                        f"Available columns: {sorted(_observed_columns(schema_contract, record.source_table))}"
                    ),
                    offending_value=record.source_column,
                )
            )


def _check_join_structure(
    record: FieldMappingRecord,
    schema_contract: EnrichedSchemaContractForSMA,
    column_aliases: list,
    errors: list[ManifestValidationError],
) -> None:
    """
    GROUP: JOIN_STRUCTURE

    Rules:
      1. join.base_table must exist in schema_contract.datasets.
      2. join.lookup_table must exist in schema_contract.datasets.
      3. Each join key must exist in base_table columns after alias resolution.
      4. Each join key must exist in lookup_table columns after alias resolution.
      5. If join keys don't match directly across tables, a ColumnAlias must bridge
         them — if none exists, raise MISSING_COLUMN_ALIAS.
      6. join.base_table must not equal join.lookup_table (same-table join is a no-op).

    Skips records with no join declared.
    """
    if record.join is None:
        return

    join = record.join
    base = join.base_table
    lookup = join.lookup_table

    # Rule 6 — same-table join
    if base == lookup:
        errors.append(
            ManifestValidationError(
                target_field=record.target_field,
                error_code=ManifestValidationErrorCode.JOIN_DECLARED_ON_SAME_TABLE,
                detail=(
                    f"join declares base_table='{base}' and lookup_table='{lookup}' "
                    "as the same table. Remove the join — source_table is already the base table."
                ),
                offending_value=base,
            )
        )
        return

    # Rule 1
    if not _has_table(schema_contract, base):
        errors.append(
            ManifestValidationError(
                target_field=record.target_field,
                error_code=ManifestValidationErrorCode.JOIN_BASE_TABLE_NOT_FOUND,
                detail=(
                    f"join.base_table '{base}' not found in schema contract. "
                    f"Available tables: {sorted(schema_contract.datasets.keys())}"
                ),
                offending_value=base,
            )
        )

    # Rule 2
    if not _has_table(schema_contract, lookup):
        errors.append(
            ManifestValidationError(
                target_field=record.target_field,
                error_code=ManifestValidationErrorCode.JOIN_LOOKUP_TABLE_NOT_FOUND,
                detail=(
                    f"join.lookup_table '{lookup}' not found in schema contract. "
                    f"Available tables: {sorted(schema_contract.datasets.keys())}"
                ),
                offending_value=lookup,
            )
        )

    # Rules 3, 4, 5 — only if both tables exist
    if not _has_table(schema_contract, base) or not _has_table(schema_contract, lookup):
        return

    base_cols = _observed_columns(schema_contract, base)
    lookup_cols = _observed_columns(schema_contract, lookup)

    for key in join.join_keys:
        base_resolved = _resolve_column_via_aliases(base, key, column_aliases)
        lookup_resolved = _resolve_column_via_aliases(lookup, key, column_aliases)

        # Rule 3
        if base_resolved not in base_cols:
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.JOIN_KEY_NOT_IN_BASE_TABLE,
                    detail=(
                        f"join key '{key}' (resolved: '{base_resolved}') not found "
                        f"in base_table '{base}'. "
                        f"Available columns: {sorted(base_cols)}"
                    ),
                    offending_value=key,
                )
            )

        # Rule 4
        if lookup_resolved not in lookup_cols:
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.JOIN_KEY_NOT_IN_LOOKUP_TABLE,
                    detail=(
                        f"join key '{key}' (resolved: '{lookup_resolved}') not found "
                        f"in lookup_table '{lookup}'. "
                        f"Available columns: {sorted(lookup_cols)}"
                    ),
                    offending_value=key,
                )
            )

        # Rule 5 — key names differ across tables, check for bridging alias
        if base_resolved != lookup_resolved:
            if not _alias_bridges_join(base, lookup, key, column_aliases):
                errors.append(
                    ManifestValidationError(
                        target_field=record.target_field,
                        error_code=ManifestValidationErrorCode.MISSING_COLUMN_ALIAS,
                        detail=(
                            f"join key '{key}' resolves differently across tables: "
                            f"'{base}' → '{base_resolved}', '{lookup}' → '{lookup_resolved}'. "
                            "No ColumnAlias bridges these names. "
                            "Add a column_aliases entry to the manifest."
                        ),
                        offending_value=key,
                    )
                )


def _check_row_selection(
    record: FieldMappingRecord,
    schema_contract: EnrichedSchemaContractForSMA,
    errors: list[ManifestValidationError],
) -> None:
    """
    GROUP: ROW_SELECTION

    Rules:
      1. row_selection.order_by must exist in schema contract for source_table
         (applies to first_by and nth strategies).
      2. row_selection.condition_col must exist in schema contract for source_table
         (applies to where_not_null strategy).
      3. row_selection.filter.column must exist in schema contract for source_table.

    Pydantic already enforces strategy/arg co-presence — we only check column existence.
    Skips records with no row_selection or no source_table.
    """
    if record.row_selection is None or record.source_table is None:
        return

    rs = record.row_selection
    table = record.source_table
    available = sorted(_observed_columns(schema_contract, table))

    # Rule 1
    if rs.order_by is not None:
        if not _has_column(schema_contract, table, rs.order_by):
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.ROW_SELECTION_ORDER_BY_NOT_FOUND,
                    detail=(
                        f"row_selection.order_by '{rs.order_by}' not found in "
                        f"table '{table}'. "
                        f"Available columns: {available}"
                    ),
                    offending_value=rs.order_by,
                )
            )

    # Rule 2
    if rs.condition_col is not None:
        if not _has_column(schema_contract, table, rs.condition_col):
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.ROW_SELECTION_CONDITION_COL_NOT_FOUND,
                    detail=(
                        f"row_selection.condition_col '{rs.condition_col}' not found in "
                        f"table '{table}'. "
                        f"Available columns: {available}"
                    ),
                    offending_value=rs.condition_col,
                )
            )

    # Rule 3
    if rs.filter is not None:
        if not _has_column(schema_contract, table, rs.filter.column):
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.ROW_SELECTION_FILTER_COL_NOT_FOUND,
                    detail=(
                        f"row_selection.filter.column '{rs.filter.column}' not found in "
                        f"table '{table}'. "
                        f"Available columns: {available}"
                    ),
                    offending_value=rs.filter.column,
                )
            )


def _check_map_unmap_consistency(
    record: FieldMappingRecord,
    errors: list[ManifestValidationError],
) -> None:
    """
    GROUP: MAP_UNMAP

    Rules:
      1. If source_column is None (unmapped), source_table and join must also be None.
         Catches partial unmapped records where the model left stale sourcing fields.
      2. If join is declared, source_column must be set.
         Pydantic catches source_column+source_table consistency, but not
         join declared without source_column.

    Note: Pydantic's validate_sourcing_consistency on FieldMappingRecord already
    catches source_column set without source_table. These rules catch the inverse
    and the join-specific case.
    """
    # Rule 1
    if record.source_column is None:
        if record.source_table is not None or record.join is not None:
            errors.append(
                ManifestValidationError(
                    target_field=record.target_field,
                    error_code=ManifestValidationErrorCode.UNMAPPED_FIELD_HAS_SOURCE,
                    detail=(
                        "Field appears unmapped (source_column is null) but "
                        f"source_table='{record.source_table}' and/or join is still populated. "
                        "Either set source_column or clear source_table and join."
                    ),
                    offending_value=None,
                )
            )

    # Rule 2
    if record.join is not None and record.source_column is None:
        errors.append(
            ManifestValidationError(
                target_field=record.target_field,
                error_code=ManifestValidationErrorCode.MAPPED_FIELD_MISSING_SOURCE,
                detail=(
                    "join is declared but source_column is null. "
                    "A cross-table field must specify the source_column in the lookup table."
                ),
                offending_value=None,
            )
        )


def _record_effective_source(record: FieldMappingRecord) -> str | None:
    if record.corrected_source_column:
        return record.corrected_source_column
    return record.source_column


def _grain_fields_for_target_schema(target_schema: str) -> list[str] | None:
    """Return Pandera Config.unique for known Edvise raw schemas; else None."""
    if target_schema == "RawEdviseStudentDataSchema":
        from edvise.data_audit.schemas.raw_edvise_student import (
            RawEdviseStudentDataSchema,
        )

        return list(RawEdviseStudentDataSchema.Config.unique)
    if target_schema == "RawEdviseCourseDataSchema":
        from edvise.data_audit.schemas.raw_edvise_course import (
            RawEdviseCourseDataSchema,
        )

        return list(RawEdviseCourseDataSchema.Config.unique)
    return None


def _check_entity_grain_keys(
    manifest: FieldMappingManifest,
    errors: list[ManifestValidationError],
) -> None:
    """
    ENTITY_GRAIN — every field in the target schema's Config.unique must have a
    mappable source column so execution can group rows at the correct grain.
    """
    grain_fields = _grain_fields_for_target_schema(manifest.target_schema)
    if not grain_fields:
        return

    by_target: dict[str, FieldMappingRecord] = {}
    for m in manifest.mappings:
        by_target[m.target_field] = m

    for field in grain_fields:
        rec = by_target.get(field)
        if rec is None:
            errors.append(
                ManifestValidationError(
                    target_field=field,
                    error_code=ManifestValidationErrorCode.GRAIN_KEY_MISSING_SOURCE,
                    detail=(
                        f"No manifest mapping for target_field '{field}', which is required "
                        f"for the entity grain ({manifest.target_schema}.Config.unique)."
                    ),
                    offending_value=None,
                )
            )
            continue
        if _record_effective_source(rec) is None:
            errors.append(
                ManifestValidationError(
                    target_field=field,
                    error_code=ManifestValidationErrorCode.GRAIN_KEY_MISSING_SOURCE,
                    detail=(
                        f"Grain key '{field}' is unmapped (no source_column / corrected_source_column). "
                        f"{manifest.target_schema}.Config.unique requires a source column for each key."
                    ),
                    offending_value=None,
                )
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_manifest(
    manifest: FieldMappingManifest,
    schema_contract: EnrichedSchemaContractForSMA,
) -> list[ManifestValidationError]:
    """
    Run all deterministic validation rules against a FieldMappingManifest.

    Returns a list of ManifestValidationError instances, one per rule violation.
    Empty list = manifest is structurally valid and ready for the refinement+HITL pass.

    Column existence is checked against training.column_details[].normalized_name
    per dataset in the enriched schema contract — no separate profiler file needed.

    Rule groups run in order:
        1. COLUMN_EXISTENCE  — source table/column present in schema contract
        2. JOIN_STRUCTURE    — join tables/keys valid, aliases present
        3. ROW_SELECTION     — strategy columns present in schema contract
        4. MAP_UNMAP         — sourcing consistency on mapped/unmapped fields
        5. ENTITY_GRAIN      — each Pandera Config.unique field has a mappable source

    Parameters
    ----------
    manifest:
        FieldMappingManifest for one entity (cohort or course).
    schema_contract:
        EnrichedSchemaContractForSMA from IdentityAgent output.
    """
    errors: list[ManifestValidationError] = []
    aliases = manifest.column_aliases

    for record in manifest.mappings:
        _check_column_existence(record, schema_contract, errors)
        _check_join_structure(record, schema_contract, aliases, errors)
        _check_row_selection(record, schema_contract, errors)
        _check_map_unmap_consistency(record, errors)

    _check_entity_grain_keys(manifest, errors)

    return errors


# Avoid clashing with pydantic.ValidationError in importing modules.
ValidationError = ManifestValidationError
ValidationErrorCode = ManifestValidationErrorCode

__all__ = [
    "ManifestValidationError",
    "ManifestValidationErrorCode",
    "ValidationError",
    "ValidationErrorCode",
    "validate_manifest",
]
