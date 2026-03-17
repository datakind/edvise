"""
Field executor for SchemaMappingAgent pipeline.

Executes a TransformationMap against resolved DataFrames using the manifest
as the complete sourcing specification.

Execution model (per field):
    1. Read FieldMappingRecord from manifest — complete sourcing spec
    2. Resolve source Series:
       a. Same-table: direct column access from source_table DataFrame
       b. Cross-table: merge base_table ← lookup_table on join_keys
    3. Apply RowSelectionConfig — unified row selection for both cases
    4. Pass resolved Series through FieldTransformationPlan steps
    5. Store output Series

Design principles:
    - Manifest owns all sourcing decisions (table, join, row selection, column)
    - Transformation plan owns all value transformation (steps only)
    - Steps are pure Series → Series — no DataFrame context except for
      birthyear_to_age_bucket and conditional_credits (second Series resolved
      from base DataFrame by executor before step chain runs)
    - CollapseConfig removed — replaced by RowSelectionConfig on manifest record
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from edvise.data_audit.genai.schema_mapping_agent.mapping_schemas import (
    BirthyearToAgeBucketStep,
    ConditionalCreditsStep,
    EntityType,
    FieldMappingManifest,
    FieldMappingRecord,
    JoinFilter,
    RowSelectionConfig,
    RowSelectionStrategy,
    TransformationMap,
    TransformationStep,
)
from edvise.data_audit.genai.schema_mapping_agent.step_dispatcher import (
    ExecutionGapError,
    ExecutionError,
    ExecutionResult,
    dispatch_step,
)

logger = logging.getLogger(__name__)

SPARK_THRESHOLD = 500_000


# =============================================================================
# Series resolution
# =============================================================================

def resolve_source_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
) -> Optional[pd.Series]:
    """
    Resolve the source Series for a field mapping record.

    Three cases:
        1. Unmappable / constant — source_column is None → return None
        2. Same-table — no join → direct column access + row selection
        3. Cross-table — join declared → merge + row selection

    Args:
        record: Approved FieldMappingRecord
        dataframes: Dict of dataset_name -> DataFrame
        alias_map: {table: {source_col: canonical_col}} from manifest column_aliases

    Returns:
        Resolved pd.Series or None if unmappable/constant
    """
    if not record.source_column or not record.source_table:
        return None

    if record.join:
        return _resolve_cross_table_series(record, dataframes, alias_map)
    else:
        return _resolve_same_table_series(record, dataframes)


def _resolve_same_table_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
) -> pd.Series:
    """
    Resolve a same-table field — direct column access with row selection applied.
    """
    _validate_table(record.source_table, dataframes)
    df = dataframes[record.source_table]

    if record.source_column not in df.columns:
        raise KeyError(
            f"Column '{record.source_column}' not found in '{record.source_table}'. "
            f"Available: {list(df.columns)}"
        )

    s = df[record.source_column].copy()

    # Apply pre-selection filter if declared
    if record.row_selection and record.row_selection.filter:
        filtered_df = _apply_filter(df, record.row_selection.filter)
        s = filtered_df[record.source_column].copy()

    return s


def _resolve_cross_table_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
) -> pd.Series:
    """
    Resolve a cross-table field — merge base ← lookup, apply row selection.

    Steps:
        1. Validate tables exist
        2. Subset lookup to join keys + target column
        3. Apply pre-selection filter if declared
        4. Sort if order_by declared
        5. Resolve actual join key names via alias_map
        6. Merge base ← lookup (left join)
        7. Apply row selection strategy
        8. Return target column aligned to base DataFrame index
    """
    join = record.join
    _validate_table(join.base_table, dataframes)
    _validate_table(join.lookup_table, dataframes)

    base_df = dataframes[join.base_table]
    lookup_join_cols = _resolve_join_keys(join.join_keys, join.lookup_table, alias_map)
    base_join_cols = _resolve_join_keys(join.join_keys, join.base_table, alias_map)

    # Subset lookup to join keys + target column only
    value_col = record.source_column
    lookup_cols_needed = list(dict.fromkeys(lookup_join_cols + [value_col]))
    lookup_df = dataframes[join.lookup_table][lookup_cols_needed].copy()

    # Apply pre-selection filter
    if record.row_selection and record.row_selection.filter:
        pre_len = len(lookup_df)
        lookup_df = _apply_filter(lookup_df, record.row_selection.filter)
        logger.debug(
            f"[{record.target_field}] Filter on '{record.row_selection.filter.column}': "
            f"{pre_len} → {len(lookup_df)} rows"
        )

    # Sort before row selection if order_by declared
    rs = record.row_selection
    if rs and rs.order_by and rs.strategy in (
        RowSelectionStrategy.first_by,
        RowSelectionStrategy.nth,
    ):
        if rs.order_by not in lookup_df.columns:
            raise KeyError(
                f"[{record.target_field}] order_by column '{rs.order_by}' "
                f"not found in '{join.lookup_table}'"
            )
        lookup_df = lookup_df.sort_values(rs.order_by, ascending=True)

    _validate_columns(base_join_cols, base_df, join.base_table)
    _validate_columns(lookup_join_cols, lookup_df, join.lookup_table)

    # Merge
    merged = base_df.merge(
        lookup_df,
        left_on=base_join_cols,
        right_on=lookup_join_cols,
        how="left",
        suffixes=("", f"_{join.lookup_table}"),
    )

    # Apply row selection strategy
    if rs:
        if rs.strategy == RowSelectionStrategy.first_by:
            merged = merged.drop_duplicates(subset=base_join_cols, keep="first")
        elif rs.strategy == RowSelectionStrategy.nth:
            merged = (
                merged
                .groupby(base_join_cols, sort=False)
                .nth(rs.n - 1)
                .reset_index()
            )
        # any_row / where_not_null — no dedup needed at merge stage
        # where_not_null filtering already applied above via filter

    if len(merged) != len(base_df):
        logger.warning(
            f"[{record.target_field}] Merged row count ({len(merged)}) differs from "
            f"base DataFrame ({len(base_df)}). Check join keys and row selection."
        )

    if value_col not in merged.columns:
        raise KeyError(
            f"[{record.target_field}] Column '{value_col}' not found after merge. "
            f"Available: {list(merged.columns)}"
        )

    return merged[value_col].reset_index(drop=True)


# =============================================================================
# Row selection (same-table cohort grain reduction)
# =============================================================================

def _apply_row_selection(
    s: pd.Series,
    record: FieldMappingRecord,
    base_df: pd.DataFrame,
    unique_keys: list[str],
    expected_n_rows: int,
) -> pd.Series:
    """
    Apply RowSelectionConfig to reduce same-table student-term grain to student grain.

    Only called for same-table cohort fields — cross-table fields have their
    row selection applied during merge in _resolve_cross_table_series().

    Args:
        s: Resolved source Series aligned to base_df index
        record: FieldMappingRecord with row_selection config
        base_df: Base DataFrame for order_by and condition_col access
        unique_keys: Schema contract unique keys — groupby keys
        expected_n_rows: Expected output row count after reduction
    """
    rs = record.row_selection
    if not rs or rs.strategy == RowSelectionStrategy.constant:
        return s.reset_index(drop=True)

    if rs.strategy == RowSelectionStrategy.any_row:
        result = (
            base_df.assign(_s=s.values)
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    elif rs.strategy == RowSelectionStrategy.first_by:
        if rs.order_by not in base_df.columns:
            raise ExecutionError(
                f"first_by order_by '{rs.order_by}' not found in base DataFrame "
                f"for field '{record.target_field}'"
            )
        result = (
            base_df.assign(_s=s.values)
            .sort_values(rs.order_by, ascending=True)
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    elif rs.strategy == RowSelectionStrategy.where_not_null:
        if rs.condition_col not in base_df.columns:
            raise ExecutionError(
                f"where_not_null condition_col '{rs.condition_col}' not found "
                f"in base DataFrame for field '{record.target_field}'"
            )
        result = (
            base_df.assign(_s=s.values)
            .loc[base_df[rs.condition_col].notna()]
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    else:
        raise ExecutionError(
            f"Unexpected row selection strategy '{rs.strategy}' in "
            f"_apply_row_selection for '{record.target_field}'"
        )

    # Reindex to expected_n_rows — where_not_null may produce fewer rows
    if len(result) < expected_n_rows:
        logger.debug(
            f"[{record.target_field}] Row selection produced {len(result)} rows "
            f"(expected {expected_n_rows}) — reindexing with nulls"
        )
        result = result.reindex(range(expected_n_rows))

    return result


# =============================================================================
# Transformation map execution
# =============================================================================

def execute_transformation_map(
    transformation_map: TransformationMap,
    manifest: FieldMappingManifest,
    dataframes: dict[str, pd.DataFrame],
    grain_keys: list[str],
    raise_on_gap: bool = False,
    spark_session: Optional[Any] = None,
) -> ExecutionResult:
    """
    Execute a TransformationMap against resolved DataFrames.

    For each field plan:
        1. Resolve source Series from manifest (join + row selection if needed)
        2. Apply same-table row selection for fields without a join
        3. Run transformation steps (pure Series → Series)

    Args:
        transformation_map: Approved TransformationMap
        manifest: Approved FieldMappingManifest (same entity type)
        dataframes: Dict of dataset_name -> DataFrame
        grain_keys: Source column names in the base DataFrame that define the
                    target output grain. Used as groupby keys in _apply_row_selection.
                    Examples:
                      cohort: ["student_id"] — one row per student
                      course: ["student_id", "term_descr", "crse_prefix", "crse_number"]
                    Must be actual column names in the base DataFrame, not target
                    schema field names.
        raise_on_gap: If True, raise ExecutionGapError on first NEW_UTILITY_NEEDED
        spark_session: Optional Spark session (reserved for future use)

    Returns:
        ExecutionResult with assembled target DataFrame and execution metadata
    """
    alias_map = _build_alias_map(manifest)
    manifest_index = {m.target_field: m for m in manifest.mappings}
    base_table = _infer_base_table(manifest)
    base_df = dataframes[base_table]

    logger.debug(
        f"[{transformation_map.entity_type}] Base table: '{base_table}', "
        f"base rows: {len(base_df)}"
    )

    result_cols: dict[str, pd.Series] = {}
    gaps: list[str] = []
    skipped: list[str] = []
    executed: list[str] = []

    for plan in transformation_map.plans:
        target = plan.target_field
        record = manifest_index.get(target)

        if not record:
            logger.warning(f"No manifest record for '{target}' — skipping")
            continue

        # Unmappable field — no steps
        if not plan.steps and not record.source_column:
            skipped.append(target)
            continue

        # Check for gaps before executing
        gap_steps = [s for s in plan.steps if s.function_name == "NEW_UTILITY_NEEDED"]
        if gap_steps:
            msg = f"Field '{target}' has {len(gap_steps)} NEW_UTILITY_NEEDED step(s)"
            logger.warning(msg)
            if raise_on_gap:
                raise ExecutionGapError(msg)
            gaps.append(target)
            continue

        try:
            # --- 1. Resolve source Series ---
            s = resolve_source_series(record, dataframes, alias_map)

            if s is None:
                # Constant field — start with empty Series for fill_constant step
                s = pd.Series(
                    [pd.NA] * len(base_df),
                    index=base_df.index,
                    dtype="object",
                )

            # --- 2. Apply same-table row selection (no join) ---
            # Cross-table row selection was already applied in resolve_source_series
            needs_row_selection = (
                record.row_selection is not None
                and record.join is None
                and record.row_selection.strategy != RowSelectionStrategy.constant
            )
            if needs_row_selection:
                s = _apply_row_selection(s, record, base_df, grain_keys)

            # --- 3. Run transformation steps ---
            for step in plan.steps:
                s = _execute_step(step, s, base_df)

            result_cols[target] = s
            executed.append(target)
            logger.debug(f"Executed: {target}")

        except ExecutionGapError:
            gaps.append(target)
            if raise_on_gap:
                raise
        except Exception as e:
            raise ExecutionError(f"Failed executing '{target}': {e}") from e

    # Align all Series to the same length before assembly.
    # Row selection may produce fewer rows than the base DataFrame for some fields
    # (e.g. where_not_null on a sparse graduation column). Use the most common
    # length as the target — this avoids assuming any particular grain upfront.
    if result_cols:
        lengths = {t: len(s) for t, s in result_cols.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            target_len = max(unique_lengths, key=list(lengths.values()).count)
            logger.warning(
                f"Series length mismatch before assembly — aligning to {target_len} rows. "
                f"Mismatched fields: "
                f"{ {t: l for t, l in lengths.items() if l != target_len} }"
            )
            result_cols = {
                t: s.reindex(range(target_len))
                for t, s in result_cols.items()
            }

    return ExecutionResult(
        df=pd.DataFrame(result_cols),
        gaps=gaps,
        skipped=skipped,
        executed=executed,
    )


def _execute_step(
    step: TransformationStep,
    s: pd.Series,
    base_df: pd.DataFrame,
) -> pd.Series:
    """
    Dispatch a single transformation step.

    Most steps are pure Series → Series and delegate directly to dispatch_step().

    Steps that declare extra_columns resolve additional Series from base_df
    and pass them as kwargs to the utility function — no hardcoding per function.

    Args:
        step: Typed TransformationStep model
        s: Input Series from prior step in the chain
        base_df: Base DataFrame for extra_columns resolution
    """
    from edvise.data_audit.genai.schema_mapping_agent import transformation_utilities as u
    from edvise.data_audit.genai.schema_mapping_agent.step_dispatcher import dispatch_step

    fn = step.function_name

    if fn == "NEW_UTILITY_NEEDED":
        raise ExecutionGapError(
            f"NEW_UTILITY_NEEDED: {getattr(step, 'description', '(no description)')}"
        )

    # Resolve extra_columns from base_df generically — no hardcoding per function
    extra_kwargs: dict[str, pd.Series] = {}
    if hasattr(step, "extra_columns") and step.extra_columns:
        for param_name, col_name in step.extra_columns.items():
            if col_name not in base_df.columns:
                raise ExecutionError(
                    f"Step '{fn}': extra_columns['{param_name}'] = '{col_name}' "
                    f"not found in base DataFrame. Available: {list(base_df.columns)}"
                )
            extra_kwargs[param_name] = base_df[col_name]

    if extra_kwargs:
        # Step needs extra Series — call utility directly with resolved kwargs
        utility_fn = getattr(u, fn, None)
        if utility_fn is None:
            raise ExecutionError(
                f"No utility function '{fn}' found in transformation_utilities. "
                f"Add it or check the function name."
            )
        return utility_fn(s, **extra_kwargs)

    # Pure Series → Series — delegate to step dispatcher
    return dispatch_step(s, step)


# =============================================================================
# Helpers
# =============================================================================

def _build_alias_map(
    manifest: FieldMappingManifest,
) -> dict[str, dict[str, str]]:
    """Build {table: {source_column: canonical_column}} from manifest column_aliases."""
    alias_map: dict[str, dict[str, str]] = {}
    for alias in manifest.column_aliases:
        alias_map.setdefault(alias.table, {})[alias.source_column] = alias.canonical_column
    return alias_map


def _resolve_join_keys(
    canonical_keys: list[str],
    table: str,
    alias_map: dict[str, dict[str, str]],
) -> list[str]:
    """Map canonical join key names to actual DataFrame column names for a table."""
    table_aliases = alias_map.get(table, {})
    reverse = {v: k for k, v in table_aliases.items()}
    return [reverse.get(k, k) for k in canonical_keys]


def _infer_base_table(manifest: FieldMappingManifest) -> str:
    """
    Identify the base (driving) table from manifest.
    All join blocks declare the same base_table — take from first one found.
    Falls back to most common source_table if no join blocks exist.
    """
    for m in manifest.mappings:
        if m.join:
            return m.join.base_table
    tables = [m.source_table for m in manifest.mappings if m.source_table]
    if not tables:
        raise ValueError(
            f"Cannot infer base table for {manifest.entity_type} manifest"
        )
    return max(set(tables), key=tables.count)


def _validate_table(table: str, dataframes: dict[str, pd.DataFrame]) -> None:
    if table not in dataframes:
        available = list(dataframes.keys())
        suggestions = [d for d in available if table.lower() in d.lower()]
        msg = f"Table '{table}' not found in dataframes. Available: {available}"
        if suggestions:
            msg += f" Did you mean: {suggestions}?"
        raise KeyError(msg)


def _validate_columns(
    cols: list[str],
    df: pd.DataFrame,
    table: str,
) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Join key columns {missing} not found in '{table}'. "
            f"Available: {list(df.columns)}"
        )


def _apply_filter(df: pd.DataFrame, f: JoinFilter) -> pd.DataFrame:
    """Apply a structured JoinFilter to a DataFrame."""
    col = df[f.column].astype("string")
    if f.operator == "contains":
        mask = col.str.contains(str(f.value), na=False, regex=False)
    elif f.operator == "equals":
        mask = col == str(f.value)
    elif f.operator == "startswith":
        mask = col.str.startswith(str(f.value), na=False)
    elif f.operator == "isin":
        mask = col.isin([str(v) for v in f.value])
    else:
        raise ValueError(f"Unknown filter operator: {f.operator}")
    return df[mask].copy()