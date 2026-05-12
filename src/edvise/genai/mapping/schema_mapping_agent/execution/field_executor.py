"""
Field executor for SchemaMappingAgent pipeline.

Executes a TransformationMap against resolved DataFrames using the manifest
as the complete sourcing specification.

Execution model (per field):
    1. Read FieldMappingRecord from manifest — complete sourcing spec
    2. Resolve source Series — always returns len(base_df) rows:
       a. Same-table: direct column access; optional row_selection.filter masks
          non-matching rows to NA for this field only
       b. Cross-table: merge base ← lookup, select correct value per base row
    3. Run transformation steps (pure Series → Series)
    4. Reduce to one value per entity using row_selection + entity_keys
    5. Assemble output DataFrame — all Series guaranteed same length

Key design principles:
    - resolve_source_series always returns a Series aligned to base_df (full length)
    - entity_keys are derived from the manifest entity grain + manifest mappings —
      source column names in base_df that correspond to the target grain; for
      course entities, mapped ``course_section_id`` is appended when its source
      column exists in ``base_df`` (section-level grain; see ``_derive_entity_keys``)
    - Grain reduction is per-field, operating on base_df with entity_keys as
      the groupby key — strategies are applied before assembly
    - where_not_null preserves all entities, producing NA for non-matching rows
    - All Series are guaranteed the same length after reduction — no post-assembly
      alignment needed
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Type

import pandas as pd

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    JoinFilter,
    RowSelectionStrategy,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    infer_manifest_base_table,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    FieldTransformationPlan,
    TransformationMap,
    TransformationStep,
)
from .step_dispatcher import (
    ExecutionGapError,
    ExecutionError,
    ExecutionResult,
    dispatch_step,
)

logger = logging.getLogger(__name__)

SPARK_THRESHOLD = 500_000


class GrainReconciliationRequired(Exception):
    """Raised when base rows exceed unique manifest-grain entities; caller runs grain HITL gate."""

    def __init__(
        self,
        *,
        institution_id: str,
        dataset: str,
        base_rows: int,
        entity_rows: int,
        manifest_source_keys: list[str],
        mapped_source_columns: list[str],
        ia_source_keys: list[str] | None,
        hitl_output_path: Path,
        entity_type: str,
        sma_manifest_path: Path | None = None,
    ) -> None:
        self.institution_id = institution_id
        self.dataset = dataset
        self.base_rows = base_rows
        self.entity_rows = entity_rows
        self.manifest_source_keys = manifest_source_keys
        self.mapped_source_columns = mapped_source_columns
        self.ia_source_keys = ia_source_keys
        self.hitl_output_path = hitl_output_path
        self.entity_type = entity_type
        self.sma_manifest_path = sma_manifest_path
        super().__init__(
            "Grain reconciliation required: "
            f"{base_rows} base rows vs {entity_rows} unique entities on keys {manifest_source_keys}. "
            f"Run edvise.genai.mapping.schema_mapping_agent.execution.grain_reconciliation."
            f"run_grain_reconciliation_gate(...) writing to {hitl_output_path!r}, resolve HITL, "
            "then re-run with sma_grain_resolution_path when applicable."
        )


# =============================================================================
# SMA grain resolution (resume after HITL)
# =============================================================================


def _mapped_non_key_source_columns_for_variance(
    manifest: FieldMappingManifest,
    entity_keys: list[str],
    base_table: str,
) -> list[str]:
    """Non-key base-table source columns that participate in field mappings (variance scoping)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in manifest.mappings:
        col = _effective_source_column(m)
        if not col or not m.source_table:
            continue
        if m.source_table != base_table:
            continue
        if col in entity_keys:
            continue
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def _apply_sma_grain_resolution_payload(
    base_df: pd.DataFrame,
    entity_keys: list[str],
    payload: dict[str, Any],
) -> pd.DataFrame:
    """Shrink ``base_df`` using a resolver-written ``sma_grain_resolution*.json`` payload."""
    gr = payload.get("grain_resolution") or payload
    strategy = gr.get("dedup_strategy")
    if strategy in (None, "suffix_identifier"):
        return base_df
    if strategy == "intentional_step_down":
        return base_df.drop_duplicates(subset=entity_keys, keep="first").reset_index(
            drop=True
        )
    if strategy == "true_duplicate":
        return base_df.drop_duplicates().reset_index(drop=True)
    if strategy == "temporal_collapse":
        sort_by = gr.get("dedup_sort_by")
        asc = gr.get("dedup_sort_ascending")
        if not sort_by or asc is None:
            logger.warning(
                "sma_grain_resolution: temporal_collapse missing sort fields — no row reduction"
            )
            return base_df
        if sort_by not in base_df.columns:
            logger.warning(
                "sma_grain_resolution: sort_by %r not in base_df — no row reduction",
                sort_by,
            )
            return base_df
        return (
            base_df.sort_values(sort_by, ascending=bool(asc))
            .drop_duplicates(subset=entity_keys, keep="first")
            .reset_index(drop=True)
        )
    logger.warning("sma_grain_resolution: unknown dedup_strategy %r — ignoring", strategy)
    return base_df


def _maybe_apply_sma_grain_resolution_file(
    base_df: pd.DataFrame,
    entity_keys: list[str],
    *,
    sma_grain_resolution_path: Path | None,
    institution_id: str | None,
    dataset: str,
) -> pd.DataFrame:
    if sma_grain_resolution_path is None:
        return base_df
    path = Path(sma_grain_resolution_path)
    if not path.is_file():
        logger.warning("sma_grain_resolution_path %s not found — skipping", path)
        return base_df
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read sma grain resolution %s: %s", path, e)
        return base_df
    if institution_id and payload.get("institution_id") not in (None, institution_id):
        logger.warning(
            "sma_grain_resolution institution_id mismatch (file=%r run=%r) — skipping",
            payload.get("institution_id"),
            institution_id,
        )
        return base_df
    if payload.get("dataset") not in (None, dataset):
        logger.warning(
            "sma_grain_resolution dataset mismatch (file=%r run=%r) — skipping",
            payload.get("dataset"),
            dataset,
        )
        return base_df
    keys_file = payload.get("manifest_source_keys") or []
    if keys_file and list(keys_file) != list(entity_keys):
        logger.warning(
            "sma_grain_resolution manifest_source_keys %s != executor entity_keys %s — skipping",
            keys_file,
            entity_keys,
        )
        return base_df
    return _apply_sma_grain_resolution_payload(base_df, entity_keys, payload)


# =============================================================================
# Entity key derivation
# =============================================================================


def _effective_source_column(record: FieldMappingRecord) -> Optional[str]:
    """Prefer HITL-corrected column when set; else manifest source_column."""
    if record.corrected_source_column:
        return record.corrected_source_column
    return record.source_column


def _derive_entity_keys(
    manifest: FieldMappingManifest,
    schema: Type,
    base_df: Optional[pd.DataFrame] = None,
) -> list[str]:
    """
    Derive source-space entity keys from the manifest entity grain + manifest mappings.

    For each target field in the grain (``COURSE_MANIFEST_GRAIN_KEYS`` for course,
    ``Config.unique`` for cohort), resolves the corresponding source column name via
    the manifest. The resulting list of source column names is used as the groupby key
    for all row_selection grain reduction strategies.

    Course schema may define module-level ``COURSE_OPTIONAL_GRAIN_TARGETS`` (see
    ``raw_edvise_course``): those target fields are appended when they have a manifest
    mapping — used for optional disambiguators (e.g. ``source_term_key``) that direct
    Edvise uploads omit.

    ``course_section_id`` is optional in the manifest when an institution has no section
    column, but when it is mapped and its source column is present in ``base_df``, that
    column is appended so section-level rows are not collapsed together.

    If required grain fields are unmapped, logs a warning and uses only the keys that
    are mapped (may be empty). Unmapped optional grain targets are omitted without
    warning. Full required-grain compliance is enforced by
    :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.validation.validate_manifest`.

    Args:
        manifest: FieldMappingManifest for this entity type
        schema: Pandera schema class — course uses ``COURSE_MANIFEST_GRAIN_KEYS``; cohort
            uses ``Config.unique`` as the required target grain
        base_df: Optional base DataFrame — used to detect presence of the section source
            column for conditional grain

    Returns:
        Source column names in base_df for mapped subset of required + optional grain
        (deduped, order preserved).

    Raises:
        ValueError: If ``Config.unique`` is missing (``None``)
    """
    target_to_source = {
        m.target_field: col
        for m in manifest.mappings
        if (col := _effective_source_column(m)) is not None
    }

    cfg = getattr(schema, "Config", object)
    raw_unique = getattr(cfg, "unique", None)
    if raw_unique is None:
        raise ValueError(f"Schema '{schema.__name__}' must define Config.unique.")
    if schema.__name__ != "RawEdviseCourseDataSchema" and not raw_unique:
        raise ValueError(
            f"Schema '{schema.__name__}' must define a non-empty Config.unique."
        )

    optional_extra: tuple[str, ...] = ()
    if schema.__name__ == "RawEdviseCourseDataSchema":
        from edvise.data_audit.schemas.raw_edvise_course import (
            COURSE_MANIFEST_GRAIN_KEYS,
            COURSE_OPTIONAL_GRAIN_TARGETS,
        )

        optional_extra = COURSE_OPTIONAL_GRAIN_TARGETS
        required_keys = list(COURSE_MANIFEST_GRAIN_KEYS)
    else:
        required_keys = list(raw_unique)

    combined: list[str] = list(required_keys)
    for tf in optional_extra:
        if tf not in combined:
            combined.append(tf)

    missing_required = [tf for tf in required_keys if target_to_source.get(tf) is None]
    if missing_required:
        logger.warning(
            "Target entity key(s) %s have no source column mapping in the manifest; "
            "execution will use a reduced grain (or row fallback). "
            "Expected keys for %s entity grain: %s. "
            "validate_manifest() should flag missing grain keys before approval.",
            missing_required,
            schema.__name__,
            list(required_keys),
        )

    entity_keys = [
        target_to_source[tf] for tf in combined if target_to_source.get(tf) is not None
    ]

    # Conditional section key: optional in the manifest grain; include when mapped.
    section_source_col = target_to_source.get("course_section_id")
    if (
        section_source_col is not None
        and section_source_col not in entity_keys
        and base_df is not None
        and section_source_col in base_df.columns
    ):
        entity_keys.append(section_source_col)
        logger.info(
            "Conditional section key detected: adding '%s' (course_section_id) to "
            "entity keys — institution provides section-level data.",
            section_source_col,
        )

    return list(dict.fromkeys(entity_keys))


# =============================================================================
# Series resolution — always returns len(base_df) rows
# =============================================================================


def resolve_source_series(
    record: FieldMappingRecord,
    dataframes: dict[str, pd.DataFrame],
    alias_map: dict[str, dict[str, str]],
    base_df: pd.DataFrame,
    base_table: str,
) -> Optional[pd.Series]:
    """
    Resolve the source Series for a field mapping record.

    Always returns a Series of len(base_df) — aligned to base DataFrame index.
    Grain reduction happens separately after transformation steps have run.

    Three cases:
        1. Unmappable / constant — source_column is None → return None
        2. Same-table — direct column access, returns len(base_df) rows; when
           row_selection.filter is set, values are NA where the filter does not pass
        3. Cross-table — merge base ← lookup, selects correct value per base row,
                         returns len(base_df) rows

    Args:
        record: Approved FieldMappingRecord
        dataframes: Dict of dataset_name -> DataFrame
        alias_map: {table: {source_col: canonical_col}} from manifest column_aliases
        base_df: Base DataFrame — used for length alignment validation
        base_table: Name of the driving table (must match infer_manifest_base_table)

    Returns:
        Resolved pd.Series of len(base_df) or None if unmappable/constant
    """
    if not record.source_column or not record.source_table:
        return None

    if record.join:
        return _resolve_cross_table_series(record, dataframes, alias_map, base_df)
    else:
        return _resolve_same_table_series(record, base_df, base_table)


def _resolve_same_table_series(
    record: FieldMappingRecord,
    base_df: pd.DataFrame,
    base_table: str,
) -> pd.Series:
    """
    Direct column access from base_df.

    Reads from the already-cleaned base_df rather than re-fetching from
    dataframes — ensures alignment with the dropna/reset_index applied at
    the top of execute_transformation_map.

    Returns Series aligned to base_df — full base length, no grain reduction.
    """
    if record.source_table != base_table:
        raise KeyError(
            f"Field '{record.target_field}': source_table '{record.source_table}' "
            f"does not match execution base table '{base_table}' while join is null. "
            "Declare join with join.base_table matching the base table and "
            "join.lookup_table set to the table that contains source_column."
        )
    if record.source_column not in base_df.columns:
        raise KeyError(
            f"Column '{record.source_column}' not found in '{record.source_table}' "
            f"(base table '{base_table}'). Available: {list(base_df.columns)}. "
            "If this column exists on another dataset, use a cross-table join."
        )

    s = base_df[record.source_column]
    rs = record.row_selection
    if rs and rs.filter:
        fc = rs.filter.column
        if fc not in base_df.columns:
            raise KeyError(
                f"row_selection.filter column '{fc}' not found in '{record.source_table}' "
                f"(base table '{base_table}'). Available: {list(base_df.columns)}."
            )
        mask = _joinfilter_pass_mask(base_df, rs.filter)
        s = s.where(mask, other=pd.NA)

    return s.reset_index(drop=True)


def _coerce_join_frames_for_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: list[str],
    right_on: list[str],
    *,
    log_context: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align incompatible join-key dtypes before ``DataFrame.merge``.

    Pandas rejects merges when one key column is string-like and the other is
    numeric or boolean (e.g. ``course_number`` as object vs nullable ``Int64``).
    For identifier-style keys, casting both sides to pandas ``StringDtype`` makes
    ``123`` and ``\"123\"`` compare equal while preserving nulls as ``<NA>``.

    When no key pair needs bridging, returns ``(left, right)`` unchanged — no
    extra copies and no ``astype``.
    """

    def _is_textualish(series: pd.Series) -> bool:
        dt = series.dtype
        return bool(
            pd.api.types.is_string_dtype(dt)
            or pd.api.types.is_object_dtype(dt)
            or pd.api.types.is_categorical_dtype(dt)
        )

    def _needs_string_bridge(a: pd.Series, b: pd.Series) -> bool:
        if a.dtype == b.dtype:
            return False
        a_num = pd.api.types.is_numeric_dtype(a.dtype)
        b_num = pd.api.types.is_numeric_dtype(b.dtype)
        a_bool = pd.api.types.is_bool_dtype(a.dtype)
        b_bool = pd.api.types.is_bool_dtype(b.dtype)
        a_txt = _is_textualish(a)
        b_txt = _is_textualish(b)
        if (a_txt and b_num) or (b_txt and a_num):
            return True
        if (a_txt and b_bool) or (b_txt and a_bool):
            return True
        if (a_bool and b_num) or (b_bool and a_num):
            return True
        return False

    pairs_to_coerce = [
        (lc, rc)
        for lc, rc in zip(left_on, right_on)
        if _needs_string_bridge(left[lc], right[rc])
    ]
    if not pairs_to_coerce:
        return left, right

    left_c = left.copy()
    right_c = right.copy()
    coerced: list[str] = []
    for lcol, rcol in pairs_to_coerce:
        left_c[lcol] = left_c[lcol].astype("string")
        right_c[rcol] = right_c[rcol].astype("string")
        coerced.append(f"{lcol}↔{rcol}")

    logger.info(
        "[%s] Join keys coerced to string for merge compatibility: %s",
        log_context,
        coerced,
    )
    return left_c, right_c


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
    if join is None:
        raise ValueError(
            f"[{record.target_field}] join is required for cross-table field resolution"
        )
    _validate_table(join.base_table, dataframes)
    _validate_table(join.lookup_table, dataframes)

    lookup_join_cols = _resolve_join_keys(join.join_keys, join.lookup_table, alias_map)
    base_join_cols = _resolve_join_keys(join.join_keys, join.base_table, alias_map)

    value_col = record.source_column
    if value_col is None:
        raise ValueError(
            f"[{record.target_field}] source_column is required for cross-table field resolution"
        )
    rs = record.row_selection
    extra_cols = [rs.order_by] if rs and rs.order_by else []
    filter_cols = [rs.filter.column] if rs and rs.filter else []
    lookup_cols_needed = list(
        dict.fromkeys(lookup_join_cols + [value_col] + extra_cols + filter_cols)
    )
    lookup_df = dataframes[join.lookup_table][lookup_cols_needed].copy()

    # Log initial state
    initial_rows = len(lookup_df)
    initial_students = (
        lookup_df[lookup_join_cols[0]].nunique()
        if lookup_join_cols and initial_rows > 0
        else 0
    )
    base_students = (
        base_df[base_join_cols[0]].nunique() if base_join_cols else len(base_df)
    )
    logger.info(
        f"[{record.target_field}] Initial lookup_df: {initial_rows} rows, "
        f"{initial_students} unique students. Base_df: {len(base_df)} rows, "
        f"{base_students} unique students."
    )

    if rs and rs.filter:
        pre_len = len(lookup_df)
        pre_students = (
            lookup_df[lookup_join_cols[0]].nunique()
            if lookup_join_cols and pre_len > 0
            else 0
        )

        # Log filter details before applying
        filter_col = rs.filter.column
        if filter_col in lookup_df.columns:
            sample_values = lookup_df[filter_col].value_counts().head(10).to_dict()
            logger.debug(
                f"[{record.target_field}] Filter column '{filter_col}' sample values: {sample_values}"
            )

        lookup_df = _apply_filter(lookup_df, rs.filter)

        post_len = len(lookup_df)
        post_students = (
            lookup_df[lookup_join_cols[0]].nunique()
            if lookup_join_cols and post_len > 0
            else 0
        )

        logger.info(
            f"[{record.target_field}] Filter '{rs.filter.operator}' on '{rs.filter.column}' "
            f"(value: {rs.filter.value}): {pre_len} → {post_len} rows "
            f"({pre_students} → {post_students} students)"
        )

        if post_len == 0:
            logger.warning(
                f"[{record.target_field}] Filter removed ALL rows! All students will get NaN. "
                f"Check filter criteria: {rs.filter.operator}('{rs.filter.column}', {rs.filter.value})"
            )

    if rs and rs.order_by:
        if rs.order_by not in lookup_df.columns:
            raise KeyError(
                f"[{record.target_field}] order_by column '{rs.order_by}' "
                f"not found in '{join.lookup_table}'"
            )

        # Check for nulls in order_by column
        null_order_count = lookup_df[rs.order_by].isna().sum()
        if null_order_count > 0:
            logger.warning(
                f"[{record.target_field}] order_by column '{rs.order_by}' has {null_order_count} "
                f"null values. These rows will sort last."
            )

        lookup_df = lookup_df.sort_values(rs.order_by, ascending=True)
        logger.debug(
            f"[{record.target_field}] Sorted by '{rs.order_by}' (ascending). "
            f"Range: {lookup_df[rs.order_by].min()} to {lookup_df[rs.order_by].max()}"
        )

    if rs and rs.strategy == RowSelectionStrategy.nth and rs.n is not None:
        # Log row counts per student before nth selection
        if lookup_join_cols and len(lookup_df) > 0:
            student_counts = lookup_df.groupby(lookup_join_cols[0]).size()
            students_with_n_or_more = (student_counts >= rs.n).sum()
            students_with_fewer = (student_counts < rs.n).sum()
            logger.info(
                f"[{record.target_field}] Before nth({rs.n}) selection: "
                f"{students_with_n_or_more} students have ≥{rs.n} rows, "
                f"{students_with_fewer} students have <{rs.n} rows "
                f"(will get NaN). Row count distribution: {student_counts.describe().to_dict()}"
            )

        lookup_df = (
            lookup_df.groupby(lookup_join_cols, sort=False).nth(rs.n - 1).reset_index()
        )

        post_nth_rows = len(lookup_df)
        post_nth_students = (
            lookup_df[lookup_join_cols[0]].nunique()
            if lookup_join_cols and post_nth_rows > 0
            else 0
        )
        logger.info(
            f"[{record.target_field}] After nth({rs.n}) selection: {post_nth_rows} rows, "
            f"{post_nth_students} students"
        )
    else:
        pre_dedup_rows = len(lookup_df)
        pre_dedup_students = (
            lookup_df[lookup_join_cols[0]].nunique()
            if lookup_join_cols and pre_dedup_rows > 0
            else 0
        )
        lookup_df = lookup_df.drop_duplicates(subset=lookup_join_cols, keep="first")
        post_dedup_rows = len(lookup_df)
        post_dedup_students = (
            lookup_df[lookup_join_cols[0]].nunique()
            if lookup_join_cols and post_dedup_rows > 0
            else 0
        )
        logger.debug(
            f"[{record.target_field}] Deduplication: {pre_dedup_rows} → {post_dedup_rows} rows "
            f"({pre_dedup_students} → {post_dedup_students} students)"
        )

    _validate_columns(base_join_cols, base_df, join.base_table)
    _validate_columns(lookup_join_cols, lookup_df, join.lookup_table)

    left_merge, lookup_merge = _coerce_join_frames_for_merge(
        base_df[base_join_cols],
        lookup_df,
        base_join_cols,
        lookup_join_cols,
        log_context=record.target_field,
    )

    merged = left_merge.merge(
        lookup_merge,
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

    # Log final merge results
    result_series = merged[value_col].reset_index(drop=True)
    non_null_count = result_series.notna().sum()
    null_count = result_series.isna().sum()
    null_pct = (null_count / len(result_series) * 100) if len(result_series) > 0 else 0

    post_dedup_students_final = (
        lookup_df[lookup_join_cols[0]].nunique()
        if lookup_join_cols and len(lookup_df) > 0
        else 0
    )
    logger.info(
        f"[{record.target_field}] Merge complete: {non_null_count} non-null values, "
        f"{null_count} null values ({null_pct:.1f}% NaN). "
        f"Lookup had {len(lookup_df)} rows for {post_dedup_students_final} students."
    )

    if null_pct > 50:
        logger.warning(
            f"[{record.target_field}] High NaN rate ({null_pct:.1f}%)! "
            f"Check filter criteria and data coverage."
        )

    return result_series


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
    for the entity grain, derived via _derive_entity_keys.

    Every strategy merges back against entity_index at the end, guaranteeing
    all fields produce Series with identical length and row ordering.

    For where_not_null: entities with no non-null row produce NA rather than
    being dropped — all entities are preserved in the output.

    Same-table ``first_by`` with ``row_selection.filter``: rows failing the filter
    are excluded **before** sort/dedup so ordering picks the first passing row
    per entity (matches cross-table lookup filter semantics).

    Args:
        s: Transformed Series of len(base_df) — values in target space
        record: FieldMappingRecord with row_selection config
        base_df: Base DataFrame — source space, used for order_by / condition_col
        entity_keys: Source column names in base_df identifying one target entity.
                     Derived from the manifest entity grain via manifest mappings.
        entity_index: Canonical entity order — one row per unique entity_keys
                      combination in base_df order. All strategies merge back
                      to this index to guarantee consistent row ordering.
    """
    rs = record.row_selection
    if not rs or rs.strategy == RowSelectionStrategy.constant:
        # Constant fields produce identical values for all rows — slice to
        # entity grain length so all Series assemble to the same length.
        return s.iloc[: len(entity_index)].reset_index(drop=True)

    def _merge_back(reduced: pd.DataFrame) -> pd.Series:
        """Left merge reduced rows back to canonical entity_index order."""
        return entity_index.merge(reduced, on=entity_keys, how="left")[
            "_s"
        ].reset_index(drop=True)

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
        df_work = base_df.assign(_s=s.values)
        if rs.filter:
            df_work = df_work.loc[_joinfilter_pass_mask(base_df, rs.filter)]
        reduced = (
            df_work.sort_values(rs.order_by, ascending=True)
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


def _accumulate_required_source_columns_for_plan(
    record: FieldMappingRecord,
    plan: FieldTransformationPlan,
    required: dict[str, set[str]],
    *,
    base_table: str,
    alias_map: dict[str, dict[str, str]],
) -> None:
    """
    Collect dataset columns the field executor reads for this manifest record + plan.

    Mirrors skip rules and sourcing paths in :func:`execute_transformation_map`.
    """
    for step in plan.steps:
        extra = getattr(step, "extra_columns", None)
        if extra:
            for col in extra.values():
                if col:
                    required[base_table].add(col)

    eff = _effective_source_column(record)
    if not eff or not record.source_table:
        return

    rs = record.row_selection

    if record.join:
        j = record.join
        base_side = j.base_table
        lookup_side = j.lookup_table
        base_keys = _resolve_join_keys(j.join_keys, base_side, alias_map)
        lookup_keys = _resolve_join_keys(j.join_keys, lookup_side, alias_map)
        required[base_side].update(base_keys)
        required[lookup_side].update(lookup_keys)
        required[lookup_side].add(eff)
        if rs:
            if rs.filter:
                required[lookup_side].add(rs.filter.column)
            if rs.order_by:
                required[lookup_side].add(rs.order_by)
            if rs.strategy == RowSelectionStrategy.where_not_null and rs.condition_col:
                required[base_side].add(rs.condition_col)
        return

    st = record.source_table
    required[st].add(eff)
    if not rs:
        return
    if rs.filter:
        required[st].add(rs.filter.column)
    if rs.strategy == RowSelectionStrategy.first_by and rs.order_by:
        required[st].add(rs.order_by)
    if rs.strategy == RowSelectionStrategy.where_not_null and rs.condition_col:
        required[st].add(rs.condition_col)
    if rs.strategy == RowSelectionStrategy.nth and rs.order_by:
        required[st].add(rs.order_by)


def validate_source_columns_for_execute(
    transformation_map: TransformationMap,
    manifest: FieldMappingManifest,
    schema: Type,
    dataframes: dict[str, pd.DataFrame],
) -> None:
    """
    Ensure cleaned dataframes contain every column SMA reads for this entity.

    Raises ``ExecutionError`` with a concise report if any required column is missing.
    Call using the same ``dataframes`` passed to :func:`execute_transformation_map`
    (before the executor mutates the base frame).
    """
    alias_map = _build_alias_map(manifest)
    manifest_index = {m.target_field: m for m in manifest.mappings}
    base_table = infer_manifest_base_table(manifest)
    if base_table not in dataframes:
        raise ExecutionError(
            f"Base table '{base_table}' not found in dataframes. "
            f"Available: {sorted(dataframes.keys())}"
        )
    base_df_in = dataframes[base_table]
    entity_keys_pre = _derive_entity_keys(manifest, schema, base_df=base_df_in)

    required: dict[str, set[str]] = defaultdict(set)
    for ek in entity_keys_pre:
        required[base_table].add(ek)

    for plan in transformation_map.plans:
        record = manifest_index.get(plan.target_field)
        if not record:
            continue
        if plan.hook_required:
            continue
        if not plan.steps and not record.source_column:
            continue
        _accumulate_required_source_columns_for_plan(
            record,
            plan,
            required,
            base_table=base_table,
            alias_map=alias_map,
        )

    problems: list[str] = []
    for ds in sorted(required.keys()):
        cols = required[ds]
        if ds not in dataframes:
            problems.append(f"{ds}: missing dataset (required columns: {sorted(cols)})")
            continue
        have = set(dataframes[ds].columns)
        missing = sorted(c for c in cols if c not in have)
        if missing:
            problems.append(f"{ds}: missing columns {missing}")

    if problems:
        raise ExecutionError(
            "Cleaned data is missing columns required by the active field mapping / "
            "transformation plans — " + "; ".join(problems)
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
    *,
    institution_id: str | None = None,
    dataset: str | None = None,
    hitl_output_path: Path | None = None,
    ia_source_keys: list[str] | None = None,
    sma_grain_resolution_path: Path | None = None,
    sma_manifest_path: Path | None = None,
) -> ExecutionResult:
    """
    Execute a TransformationMap against resolved DataFrames.

    For each field plan:
        1. Resolve source Series — always len(base_df)
        2. Run transformation steps (pure Series → Series)
        3. Reduce to one value per entity using row_selection + entity_keys

    entity_keys are derived from the manifest entity grain resolved to source column
    names via the manifest, plus optional grain fields (e.g. mapped
    ``course_section_id`` when its source column exists on ``base_df``). This
    keeps all reduced Series aligned — assembly into a DataFrame is always clean.

    Args:
        transformation_map: Approved TransformationMap
        manifest: Approved FieldMappingManifest (same entity type)
        dataframes: Dict of dataset_name -> DataFrame
        schema: Pandera schema class for the target entity type
                (e.g. RawEdviseCourseDataSchema).
                Cohort: ``Config.unique``; course: ``COURSE_MANIFEST_GRAIN_KEYS`` plus
                optional targets (see ``_derive_entity_keys``).
        raise_on_gap: If True, raise ExecutionGapError on first plan with hook_required
        spark_session: Optional Spark session (reserved for future use)
        institution_id: Required when manifest grain is stricter than row count (grain gate).
        dataset: Logical dataset label for HITL artifacts (defaults to manifest base table).
        hitl_output_path: Where to write ``sma_grain_hitl.json`` when grain reconciliation triggers.
        ia_source_keys: IdentityAgent ``post_clean_primary_key`` in source space; when omitted,
            all grain mismatches are treated as within-grain multiplicity.
        sma_grain_resolution_path: Optional resolver output to shrink ``base_df`` before execution.
        sma_manifest_path: Optional manifest file path stored into HITL metadata for resolver suffix.

    Returns:
        ExecutionResult with assembled target DataFrame and execution metadata
    """
    validate_source_columns_for_execute(
        transformation_map, manifest, schema, dataframes
    )

    alias_map = _build_alias_map(manifest)
    manifest_index = {m.target_field: m for m in manifest.mappings}
    base_table = infer_manifest_base_table(manifest)
    base_df = dataframes[base_table]
    entity_keys = _derive_entity_keys(manifest, schema, base_df=base_df)
    if not entity_keys:
        logger.warning(
            "[%s] No entity keys resolved from manifest; using one row per base row "
            "as the execution grain (_sma_fallback_entity_row).",
            transformation_map.entity_type,
        )
        base_df = base_df.copy()
        base_df["_sma_fallback_entity_row"] = range(len(base_df))
        entity_keys = ["_sma_fallback_entity_row"]

    # Drop rows with null entity keys and reset index — clean RangeIndex is
    # required since we use .values to align Series during grain reduction.
    base_df = base_df.dropna(subset=entity_keys).reset_index(drop=True)

    dataset_label = dataset if dataset is not None else base_table
    base_df = _maybe_apply_sma_grain_resolution_file(
        base_df,
        entity_keys,
        sma_grain_resolution_path=sma_grain_resolution_path,
        institution_id=institution_id,
        dataset=dataset_label,
    )

    # Canonical entity order — all strategies merge back to this index so
    # every field's reduced Series has the same row ordering.
    entity_index = base_df.drop_duplicates(subset=entity_keys, keep="first")[
        entity_keys
    ].reset_index(drop=True)

    n_unique = len(entity_index)
    if n_unique < len(base_df):
        mapped_source_columns = _mapped_non_key_source_columns_for_variance(
            manifest, entity_keys, base_table
        )
        if not institution_id:
            raise ValueError(
                "Grain mismatch detected (more base rows than manifest-grain entities). "
                "Pass institution_id=... to execute_transformation_map."
            )
        if hitl_output_path is None:
            raise ValueError(
                "Grain mismatch requires human review. Pass hitl_output_path=Path(...) "
                "to execute_transformation_map so the caller can write sma_grain_hitl.json "
                "via run_grain_reconciliation_gate after catching GrainReconciliationRequired."
            )
        raise GrainReconciliationRequired(
            institution_id=institution_id,
            dataset=dataset_label,
            base_rows=len(base_df),
            entity_rows=n_unique,
            manifest_source_keys=list(entity_keys),
            mapped_source_columns=mapped_source_columns,
            ia_source_keys=ia_source_keys,
            hitl_output_path=Path(hitl_output_path),
            entity_type=str(transformation_map.entity_type),
            sma_manifest_path=sma_manifest_path,
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
            logger.warning(
                f"[{i}/{n_plans}] No manifest record for '{target}' — skipping"
            )
            continue

        if plan.hook_required:
            msg = f"Field '{target}' — hook_required (no covering utility chain; see reviewer_notes)"
            logger.warning(f"[{i}/{n_plans}] {target} — {msg}")
            if raise_on_gap:
                raise ExecutionGapError(msg)
            gaps.append(target)
            continue

        if not plan.steps and not record.source_column:
            logger.debug(f"[{i}/{n_plans}] {target} — unmappable, skipping")
            skipped.append(target)
            continue

        try:
            logger.debug(f"[{i}/{n_plans}] {target} — resolving source series")

            # --- 1. Resolve source Series (always len(base_df)) ---
            s = resolve_source_series(
                record, dataframes, alias_map, base_df, base_table
            )

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
                logger.debug(f"[{i}/{n_plans}] {target} — reducing via {strategy}")
                s = _apply_grain_reduction(
                    s, record, base_df, entity_keys, entity_index
                )

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
    from edvise.genai.mapping.schema_mapping_agent.transformation import utilities as u

    fn = step.function_name

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
        alias_map.setdefault(alias.table, {})[alias.source_column] = (
            alias.canonical_column
        )
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


def _joinfilter_pass_mask(df: pd.DataFrame, f: JoinFilter) -> pd.Series:
    """Boolean mask — True where ``df`` satisfies ``f`` (same semantics as lookup filtering)."""
    col = df[f.column].astype("string")
    if f.operator == "contains":
        return col.str.contains(str(f.value), na=False, regex=False)
    if f.operator == "equals":
        return col == str(f.value)
    if f.operator == "startswith":
        return col.str.startswith(str(f.value), na=False)
    if f.operator == "isin":
        return col.isin([str(v) for v in f.value])
    raise ValueError(f"Unknown filter operator: {f.operator}")


def _apply_filter(df: pd.DataFrame, f: JoinFilter) -> pd.DataFrame:
    """Apply a structured JoinFilter to a DataFrame."""
    return df[_joinfilter_pass_mask(df, f)].copy()
