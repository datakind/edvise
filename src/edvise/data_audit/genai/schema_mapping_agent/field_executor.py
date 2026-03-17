"""
Field executor for SchemaMappingAgent pipeline.

Executes a TransformationMap against resolved DataFrames using the manifest
as the complete sourcing specification.

Execution model (per field):
    1. Read FieldMappingRecord from manifest — complete sourcing spec
    2. Resolve source Series — always returns len(base_df) rows:
       a. Same-table: direct column access
       b. Cross-table: merge base ← lookup, select correct value per base row
    3. Apply grain reduction uniformly — reduce to target grain if base table
       has more rows than grain_keys combinations
    4. Run transformation steps (pure Series → Series)
    5. Store output Series

Key design principle:
    resolve_source_series always returns a Series aligned to base_df (full length).
    Grain reduction always happens AFTER resolution, uniformly for all fields.
    This eliminates the length mismatch problem from having two different code
    paths with row selection happening at different points.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from edvise.data_audit.genai.schema_mapping_agent.mapping_schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinFilter,
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
# Series resolution — always returns len(base_df) rows
# =============================================================================

def resolve_source_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
    base_df: pd.DataFrame,
) -> Optional[pd.Series]:
    """
    Resolve the source Series for a field mapping record.

    Always returns a Series of len(base_df) — aligned to base DataFrame index.
    Grain reduction happens separately and uniformly after resolution.

    Three cases:
        1. Unmappable / constant — source_column is None → return None
        2. Same-table — direct column access, returns len(base_df) rows
        3. Cross-table — merge base ← lookup, selects correct value per base row,
                         returns len(base_df) rows

    Args:
        record: Approved FieldMappingRecord
        dataframes: Dict of dataset_name -> DataFrame
        alias_map: {table: {source_col: canonical_col}} from manifest column_aliases
        base_df: Base DataFrame — used for length alignment validation

    Returns:
        Resolved pd.Series of len(base_df) or None if unmappable/constant
    """
    if not record.source_column or not record.source_table:
        return None

    if record.join:
        return _resolve_cross_table_series(record, dataframes, alias_map, base_df)
    else:
        return _resolve_same_table_series(record, dataframes, base_df)


def _resolve_same_table_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    base_df: pd.DataFrame,
) -> pd.Series:
    """
    Direct column access from source table.
    Returns Series aligned to base_df — full base length, no grain reduction.
    """
    _validate_table(record.source_table, dataframes)
    df = dataframes[record.source_table]

    if record.source_column not in df.columns:
        raise KeyError(
            f"Column '{record.source_column}' not found in '{record.source_table}'. "
            f"Available: {list(df.columns)}"
        )

    return df[record.source_column].reset_index(drop=True)


def _resolve_cross_table_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
    base_df: pd.DataFrame,
) -> pd.Series:
    """
    Resolve a cross-table field via merge.

    Selects the correct value per base row (via filter/sort/dedup on the lookup
    table) then returns a Series of len(base_df) — one value per base row.
    Grain reduction to target grain happens separately in _apply_grain_reduction.

    Steps:
        1. Validate tables exist
        2. Subset lookup to join keys + target column
        3. Apply pre-selection filter if declared
        4. Sort lookup if order_by declared
        5. Deduplicate lookup to one row per join key combination
        6. Resolve actual join key names via alias_map
        7. Left merge base ← lookup
        8. Return target column aligned to base_df
    """
    join = record.join
    _validate_table(join.base_table, dataframes)
    _validate_table(join.lookup_table, dataframes)

    lookup_join_cols = _resolve_join_keys(join.join_keys, join.lookup_table, alias_map)
    base_join_cols = _resolve_join_keys(join.join_keys, join.base_table, alias_map)

    # Subset lookup to join keys + target column only
    value_col = record.source_column
    lookup_cols_needed = list(dict.fromkeys(lookup_join_cols + [value_col]))
    lookup_df = dataframes[join.lookup_table][lookup_cols_needed].copy()

    rs = record.row_selection

    # Apply pre-selection filter
    if rs and rs.filter:
        pre_len = len(lookup_df)
        lookup_df = _apply_filter(lookup_df, rs.filter)
        logger.debug(
            f"[{record.target_field}] Filter on '{rs.filter.column}': "
            f"{pre_len} → {len(lookup_df)} rows"
        )

    # Sort before dedup if order_by declared
    if rs and rs.order_by:
        if rs.order_by not in lookup_df.columns:
            raise KeyError(
                f"[{record.target_field}] order_by column '{rs.order_by}' "
                f"not found in '{join.lookup_table}'"
            )
        lookup_df = lookup_df.sort_values(rs.order_by, ascending=True)

    # Deduplicate lookup to one row per join key combination
    # nth: take the nth row per group (1-based)
    # first_by / any_row / where_not_null: take first row per group
    if rs and rs.strategy == RowSelectionStrategy.nth and rs.n is not None:
        lookup_df = (
            lookup_df
            .groupby(lookup_join_cols, sort=False)
            .nth(rs.n - 1)
            .reset_index()
        )
    else:
        lookup_df = lookup_df.drop_duplicates(subset=lookup_join_cols, keep="first")

    _validate_columns(base_join_cols, base_df, join.base_table)
    _validate_columns(lookup_join_cols, lookup_df, join.lookup_table)

    # Left merge — preserves all base_df rows, fills null where no match
    merged = base_df[base_join_cols].merge(
        lookup_df,
        left_on=base_join_cols,
        right_on=lookup_join_cols,
        how="left",
        suffixes=("", f"_{join.lookup_table}"),
    )

    if len(merged) != len(base_df):
        logger.warning(
            f"[{record.target_field}] Merged row count ({len(merged)}) != "
            f"base_df ({len(base_df)}). Join may have fan-out. "
            f"Check join keys and row selection."
        )

    if value_col not in merged.columns:
        raise KeyError(
            f"[{record.target_field}] Column '{value_col}' not found after merge. "
            f"Available: {list(merged.columns)}"
        )

    return merged[value_col].reset_index(drop=True)


# =============================================================================
# Grain reduction — uniform for all fields
# =============================================================================

def _apply_grain_reduction(
    s: pd.Series,
    record: FieldMappingRecord,
    base_df: pd.DataFrame,
    grain_keys: list[str],
) -> pd.Series:
    """
    Reduce Series from base_df grain to target grain.

    Called uniformly for all fields when base_needs_reduction is True.
    At this point s is always len(base_df) — same-table or cross-table.

    Args:
        s: Source Series of len(base_df)
        record: FieldMappingRecord with row_selection config
        base_df: Base DataFrame for order_by and condition_col access
        grain_keys: Source column names in base_df defining target grain.
                    Used as groupby keys for drop_duplicates.
    """
    rs = record.row_selection
    if not rs or rs.strategy == RowSelectionStrategy.constant:
        return s.reset_index(drop=True)

    if rs.strategy in (RowSelectionStrategy.any_row, RowSelectionStrategy.nth):
        # any_row: value is invariant per grain key — take first
        # nth cross-table: lookup already deduplicated to correct row,
        #                  just reduce base to grain
        return (
            base_df.assign(_s=s.values)
            .drop_duplicates(subset=grain_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    if rs.strategy == RowSelectionStrategy.first_by:
        if rs.order_by not in base_df.columns:
            raise ExecutionError(
                f"first_by order_by '{rs.order_by}' not found in base DataFrame "
                f"for field '{record.target_field}'"
            )
        return (
            base_df.assign(_s=s.values)
            .sort_values(rs.order_by, ascending=True)
            .drop_duplicates(subset=grain_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    if rs.strategy == RowSelectionStrategy.where_not_null:
        if rs.condition_col not in base_df.columns:
            raise ExecutionError(
                f"where_not_null condition_col '{rs.condition_col}' not found "
                f"in base DataFrame for field '{record.target_field}'"
            )
        return (
            base_df.assign(_s=s.values)
            .loc[base_df[rs.condition_col].notna()]
            .drop_duplicates(subset=grain_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )

    raise ExecutionError(
        f"Unexpected row selection strategy '{rs.strategy}' for '{record.target_field}'"
    )


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
        1. Resolve source Series — always len(base_df)
        2. Apply grain reduction uniformly if base table needs it
        3. Run transformation steps (pure Series → Series)

    Args:
        transformation_map: Approved TransformationMap
        manifest: Approved FieldMappingManifest (same entity type)
        dataframes: Dict of dataset_name -> DataFrame
        grain_keys: Source column names in base_df defining target output grain.
                    Used as groupby keys for grain reduction.
                    Must be actual column names in the base DataFrame.
                    Examples:
                        cohort: ["student_id"]
                        course: ["student_id", "term_descr", "crse_prefix", "crse_number"]
        raise_on_gap: If True, raise ExecutionGapError on first NEW_UTILITY_NEEDED
        spark_session: Optional Spark session (reserved for future use)

    Returns:
        ExecutionResult with assembled target DataFrame and execution metadata
    """
    alias_map = _build_alias_map(manifest)
    manifest_index = {m.target_field: m for m in manifest.mappings}
    base_table = _infer_base_table(manifest)
    base_df = dataframes[base_table]

    # Determine whether grain reduction is needed.
    # If grain_keys produce fewer unique rows than len(base_df), the base table
    # is multi-row per entity and grain reduction must run for all fields.
    n_unique = base_df[grain_keys].drop_duplicates().shape[0]
    base_needs_reduction = n_unique < len(base_df)

    logger.debug(
        f"[{transformation_map.entity_type}] Base table: '{base_table}', "
        f"base rows: {len(base_df)}, grain rows: {n_unique}, "
        f"needs reduction: {base_needs_reduction}"
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

        # Unmappable field — no steps and no source column
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
            # --- 1. Resolve source Series (always len(base_df)) ---
            s = resolve_source_series(record, dataframes, alias_map, base_df)

            if s is None:
                # Constant field — start with empty Series for fill_constant step
                s = pd.Series(
                    [pd.NA] * len(base_df),
                    index=base_df.index,
                    dtype="object",
                )

            # --- 2. Grain reduction (uniform for all fields) ---
            if (
                base_needs_reduction
                and record.row_selection is not None
                and record.row_selection.strategy != RowSelectionStrategy.constant
            ):
                s = _apply_grain_reduction(s, record, base_df, grain_keys)

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
    # where_not_null legitimately produces fewer rows (e.g. students who never
    # graduated). Unexpected mismatches on other strategies are logged as warnings.
    if result_cols:
        lengths = {t: len(s) for t, s in result_cols.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) > 1:
            target_len = max(unique_lengths, key=list(lengths.values()).count)
            expected_short = {RowSelectionStrategy.where_not_null}

            for t, l in lengths.items():
                if l == target_len:
                    continue
                rec = manifest_index.get(t)
                strategy = (
                    rec.row_selection.strategy
                    if rec and rec.row_selection else None
                )
                if strategy in expected_short:
                    logger.debug(
                        f"[{t}] Fewer rows ({l}) due to {strategy} — "
                        f"reindexing to {target_len}"
                    )
                else:
                    logger.warning(
                        f"[{t}] Unexpected length mismatch: {l} != {target_len}. "
                        f"Strategy: {strategy}. Check join keys and row selection."
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

    Most steps are pure Series → Series and delegate to dispatch_step().
    Steps that declare extra_columns resolve additional Series from base_df
    generically and pass them as kwargs to the utility function.
    """
    from edvise.data_audit.genai.schema_mapping_agent import transformation_utilities as u
    from edvise.data_audit.genai.schema_mapping_agent.step_dispatcher import dispatch_step

    fn = step.function_name

    if fn == "NEW_UTILITY_NEEDED":
        raise ExecutionGapError(
            f"NEW_UTILITY_NEEDED: {getattr(step, 'description', '(no description)')}"
        )

    # Resolve extra_columns from base_df generically
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
        utility_fn = getattr(u, fn, None)
        if utility_fn is None:
            raise ExecutionError(
                f"No utility function '{fn}' found in transformation_utilities."
            )
        return utility_fn(s, **extra_kwargs)

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