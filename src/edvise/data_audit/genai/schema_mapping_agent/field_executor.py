"""
Field executor for SchemaMappingAgent pipeline.

Executes a TransformationMap against resolved DataFrames using the manifest
as the complete sourcing specification.

Execution model (per field):
    1. Read FieldMappingRecord from manifest — complete sourcing spec
    2. Resolve source Series — always returns len(base_df) rows:
       a. Same-table: direct column access
       b. Cross-table: merge base ← lookup, select correct value per base row
    3. Run transformation steps (pure Series → Series)
    4. Reduce to one value per entity using row_selection + entity_keys
    5. Assemble output DataFrame — all Series guaranteed same length

Key design principles:
    - resolve_source_series always returns a Series aligned to base_df (full length)
    - entity_keys are derived from schema.Config.unique + manifest mappings —
      source column names in base_df that correspond to the target grain
    - Grain reduction is per-field, operating on base_df with entity_keys as
      the groupby key — strategies are applied before assembly
    - where_not_null preserves all entities, producing NA for non-matching rows
    - All Series are guaranteed the same length after reduction — no post-assembly
      alignment needed
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

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
# Entity key derivation
# =============================================================================

def _derive_entity_keys(
    manifest: FieldMappingManifest,
    schema: Type,
) -> list[str]:
    """
    Derive source-space entity keys from schema.Config.unique + manifest mappings.

    For each target field in the schema's uniqueness constraint, resolves the
    corresponding source column name via the manifest. The resulting list of
    source column names is used as the groupby key for all row_selection
    grain reduction strategies.

    Args:
        manifest: FieldMappingManifest for this entity type
        schema: Pandera schema class — schema.Config.unique defines target grain

    Returns:
        Source column names in base_df corresponding to schema.Config.unique

    Raises:
        ValueError: If any field in schema.Config.unique has no source column
                    mapping in the manifest
    """
    target_to_source = {
        m.target_field: m.source_column
        for m in manifest.mappings
        if m.source_column is not None
    }

    entity_keys = []
    for target_field in schema.Config.unique:
        source_col = target_to_source.get(target_field)
        if source_col is None:
            raise ValueError(
                f"Target unique key '{target_field}' has no source column mapping "
                f"in the manifest. All fields in schema.Config.unique must be "
                f"mapped to a source column in the base table."
            )
        entity_keys.append(source_col)

    # Deduplicate while preserving order — multiple target fields may map to
    # the same source column (e.g. academic_year and academic_term both from "term")
    return list(dict.fromkeys(entity_keys))


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
    Grain reduction happens separately after transformation steps have run.

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
        return _resolve_same_table_series(record, base_df)


def _resolve_same_table_series(
    record: FieldMappingRecord,
    base_df: pd.DataFrame,
) -> pd.Series:
    """
    Direct column access from base_df.

    Reads from the already-cleaned base_df rather than re-fetching from
    dataframes — ensures alignment with the dropna/reset_index applied at
    the top of execute_transformation_map.

    Returns Series aligned to base_df — full base length, no grain reduction.
    """
    if record.source_column not in base_df.columns:
        raise KeyError(
            f"Column '{record.source_column}' not found in '{record.source_table}'. "
            f"Available: {list(base_df.columns)}"
        )

    return base_df[record.source_column].reset_index(drop=True)


def _resolve_cross_table_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
    base_df: pd.DataFrame,
) -> pd.Series:
    """
    Resolve a cross-table field via merge.

    Returns a Series of len(base_df) — one value per base row.
    Grain reduction happens separately after transformation steps have run.

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

    value_col = record.source_column
    rs = record.row_selection
    extra_cols = [rs.order_by] if rs and rs.order_by else []
    lookup_cols_needed = list(dict.fromkeys(lookup_join_cols + [value_col] + extra_cols))
    lookup_df = dataframes[join.lookup_table][lookup_cols_needed].copy()

    if rs and rs.filter:
        pre_len = len(lookup_df)
        lookup_df = _apply_filter(lookup_df, rs.filter)
        logger.debug(
            f"[{record.target_field}] Filter on '{rs.filter.column}': "
            f"{pre_len} → {len(lookup_df)} rows"
        )

    if rs and rs.order_by:
        if rs.order_by not in lookup_df.columns:
            raise KeyError(
                f"[{record.target_field}] order_by column '{rs.order_by}' "
                f"not found in '{join.lookup_table}'"
            )
        lookup_df = lookup_df.sort_values(rs.order_by, ascending=True)

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
# Per-field grain reduction — operates on base_df in source space
# =============================================================================

def _apply_grain_reduction(
    s: pd.Series,
    record: FieldMappingRecord,
    base_df: pd.DataFrame,
    entity_keys: list[str],
    entity_index: pd.DataFrame,
) -> pd.Series:
    """
    Reduce a transformed Series to one value per entity.

    Operates on base_df in source space — order_by and condition_col are source
    column names that exist in base_df. entity_keys are the source column names
    corresponding to schema.Config.unique, derived via _derive_entity_keys.

    Every strategy merges back against entity_index at the end, guaranteeing
    all fields produce Series with identical length and row ordering.

    For where_not_null: entities with no non-null row produce NA rather than
    being dropped — all entities are preserved in the output.

    Args:
        s: Transformed Series of len(base_df) — values in target space
        record: FieldMappingRecord with row_selection config
        base_df: Base DataFrame — source space, used for order_by / condition_col
        entity_keys: Source column names in base_df identifying one target entity.
                     Derived from schema.Config.unique via manifest mappings.
        entity_index: Canonical entity order — one row per unique entity_keys
                      combination in base_df order. All strategies merge back
                      to this index to guarantee consistent row ordering.
    """
    rs = record.row_selection
    if not rs or rs.strategy == RowSelectionStrategy.constant:
        # Constant fields produce identical values for all rows — slice to
        # entity grain length so all Series assemble to the same length.
        return s.iloc[:len(entity_index)].reset_index(drop=True)

    def _merge_back(reduced: pd.DataFrame) -> pd.Series:
        """Left merge reduced rows back to canonical entity_index order."""
        return (
            entity_index
            .merge(reduced, on=entity_keys, how="left")["_s"]
            .reset_index(drop=True)
        )

    if rs.strategy in (RowSelectionStrategy.any_row, RowSelectionStrategy.nth):
        reduced = (
            base_df.assign(_s=s.values)
            .drop_duplicates(subset=entity_keys, keep="first")[entity_keys + ["_s"]]
            .reset_index(drop=True)
        )
        return _merge_back(reduced)

    if rs.strategy == RowSelectionStrategy.first_by:
        if record.join:
            # Cross-table: ordering already applied during lookup dedup in
            # _resolve_cross_table_series — just reduce to entity grain
            reduced = (
                base_df.assign(_s=s.values)
                .drop_duplicates(subset=entity_keys, keep="first")[entity_keys + ["_s"]]
                .reset_index(drop=True)
            )
            return _merge_back(reduced)
        if rs.order_by not in base_df.columns:
            raise ExecutionError(
                f"first_by order_by '{rs.order_by}' not found in base DataFrame "
                f"for field '{record.target_field}'"
            )
        reduced = (
            base_df.assign(_s=s.values)
            .sort_values(rs.order_by, ascending=True)
            .drop_duplicates(subset=entity_keys, keep="first")[entity_keys + ["_s"]]
            .reset_index(drop=True)
        )
        return _merge_back(reduced)

    if rs.strategy == RowSelectionStrategy.where_not_null:
        if rs.condition_col not in base_df.columns:
            raise ExecutionError(
                f"where_not_null condition_col '{rs.condition_col}' not found "
                f"in base DataFrame for field '{record.target_field}'"
            )
        reduced = (
            base_df.assign(_s=s.values)
            .loc[base_df[rs.condition_col].notna()]
            .drop_duplicates(subset=entity_keys, keep="first")[entity_keys + ["_s"]]
            .reset_index(drop=True)
        )
        return _merge_back(reduced)

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
    schema: Type,
    raise_on_gap: bool = False,
    spark_session: Optional[Any] = None,
) -> ExecutionResult:
    """
    Execute a TransformationMap against resolved DataFrames.

    For each field plan:
        1. Resolve source Series — always len(base_df)
        2. Run transformation steps (pure Series → Series)
        3. Reduce to one value per entity using row_selection + entity_keys

    entity_keys are derived from schema.Config.unique resolved to source column
    names via the manifest. This guarantees all Series are the same length after
    reduction — assembly into a DataFrame is always clean.

    Args:
        transformation_map: Approved TransformationMap
        manifest: Approved FieldMappingManifest (same entity type)
        dataframes: Dict of dataset_name -> DataFrame
        schema: Pandera schema class for the target entity type
                (e.g. RawEdviseCourseDataSchemaFlexible).
                schema.Config.unique defines the target grain.
        raise_on_gap: If True, raise ExecutionGapError on first NEW_UTILITY_NEEDED
        spark_session: Optional Spark session (reserved for future use)

    Returns:
        ExecutionResult with assembled target DataFrame and execution metadata
    """
    alias_map = _build_alias_map(manifest)
    manifest_index = {m.target_field: m for m in manifest.mappings}
    base_table = _infer_base_table(manifest)
    base_df = dataframes[base_table]
    entity_keys = _derive_entity_keys(manifest, schema)

    # Drop rows with null entity keys and reset index — clean RangeIndex is
    # required since we use .values to align Series during grain reduction.
    base_df = base_df.dropna(subset=entity_keys).reset_index(drop=True)

    # Canonical entity order — all strategies merge back to this index so
    # every field's reduced Series has the same row ordering.
    entity_index = (
        base_df
        .drop_duplicates(subset=entity_keys, keep="first")[entity_keys]
        .reset_index(drop=True)
    )

    logger.debug(
        f"[{transformation_map.entity_type}] Base table: '{base_table}', "
        f"base rows: {len(base_df)}, entity_keys: {entity_keys}, "
        f"unique entities: {len(entity_index)}"
    )

    result_cols: dict[str, pd.Series] = {}
    gaps: list[str] = []
    skipped: list[str] = []
    executed: list[str] = []

    n_plans = len(transformation_map.plans)
    logger.info(
        f"[{transformation_map.entity_type}] Starting execution — "
        f"{n_plans} fields, {len(base_df)} base rows, {len(entity_index)} entities"
    )

    for i, plan in enumerate(transformation_map.plans, 1):
        target = plan.target_field
        record = manifest_index.get(target)

        if not record:
            logger.warning(f"[{i}/{n_plans}] No manifest record for '{target}' — skipping")
            continue

        if not plan.steps and not record.source_column:
            logger.debug(f"[{i}/{n_plans}] {target} — unmappable, skipping")
            skipped.append(target)
            continue

        gap_steps = [s for s in plan.steps if s.function_name == "NEW_UTILITY_NEEDED"]
        if gap_steps:
            msg = f"Field '{target}' has {len(gap_steps)} NEW_UTILITY_NEEDED step(s)"
            logger.warning(f"[{i}/{n_plans}] {target} — {msg}")
            if raise_on_gap:
                raise ExecutionGapError(msg)
            gaps.append(target)
            continue

        try:
            logger.debug(f"[{i}/{n_plans}] {target} — resolving source series")

            # --- 1. Resolve source Series (always len(base_df)) ---
            s = resolve_source_series(record, dataframes, alias_map, base_df)

            if s is None:
                s = pd.Series(
                    [pd.NA] * len(base_df),
                    index=base_df.index,
                    dtype="object",
                )

            # --- 2. Run transformation steps (pure Series → Series) ---
            for j, step in enumerate(plan.steps, 1):
                logger.debug(
                    f"[{i}/{n_plans}] {target} — step {j}/{len(plan.steps)}: "
                    f"{step.function_name}"
                )
                s = _execute_step(step, s, base_df)

            # --- 3. Reduce to one value per entity ---
            if record.row_selection is not None:
                strategy = record.row_selection.strategy
                logger.debug(
                    f"[{i}/{n_plans}] {target} — reducing via {strategy}"
                )
                s = _apply_grain_reduction(s, record, base_df, entity_keys, entity_index)

            result_cols[target] = s
            executed.append(target)
            logger.info(f"[{i}/{n_plans}] ✓ {target} — {len(s)} rows")

        except ExecutionGapError:
            gaps.append(target)
            if raise_on_gap:
                raise
        except Exception as e:
            raise ExecutionError(f"Failed executing '{target}': {e}") from e

    logger.info(
        f"[{transformation_map.entity_type}] Execution complete — "
        f"{len(executed)} executed, {len(skipped)} skipped, {len(gaps)} gaps"
    )

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