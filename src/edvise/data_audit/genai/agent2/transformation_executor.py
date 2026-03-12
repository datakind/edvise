"""
Transformation map executor for Agent 2c.

Executes a TransformationMap deterministically against a pre-joined DataFrame.
The executor:
  1. Reads schema_contract.unique_keys to determine groupby keys for collapse
  2. Applies CollapseConfig per field to reduce grain before transformation
  3. Dispatches each TransformationStep to the corresponding utility function
  4. Assembles the final target DataFrame

Assumptions:
  - Input df is already joined across source tables (pre-processing layer handles joins)
  - schema_contract provides unique_keys per source dataset
  - Any plan with NEW_UTILITY_NEEDED steps raises ExecutionGapError — must be resolved
    before executor can run end-to-end
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from edvise.data_audit.genai.agent2.transformation_utilities import (
    cast_boolean,
    cast_datetime,
    cast_nullable_float,
    cast_nullable_int,
    cast_string,
    coerce_datetime,
    coerce_numeric,
    combine_columns,
    deduplicate_rows,
    fill_nulls,
    lowercase,
    map_values,
    normalize_credential,
    normalize_enrollment,
    normalize_grade,
    normalize_pell,
    normalize_student_age,
    normalize_term_code,
    replace_null_tokens,
    replace_values_with_null,
    strip_trailing_decimal,
    strip_whitespace,
    uppercase,
)
from edvise.data_audit.genai.agent2.mapping_schemas import (
    CollapseStrategy,
    FieldTransformationPlan,
    TransformationMap,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class ExecutionGapError(Exception):
    """Raised when a plan contains a NEW_UTILITY_NEEDED step."""
    pass


class ExecutionError(Exception):
    """Raised when a transformation step fails."""
    pass


# -----------------------------------------------------------------------------
# Execution result
# -----------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Result of executing a TransformationMap."""
    df: pd.DataFrame
    gaps: list[str] = field(default_factory=list)       # target fields with NEW_UTILITY_NEEDED
    skipped: list[str] = field(default_factory=list)    # unmappable fields (empty steps)
    executed: list[str] = field(default_factory=list)   # successfully transformed fields

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)


# -----------------------------------------------------------------------------
# Collapse helpers
# -----------------------------------------------------------------------------

def _collapse_field(
    df: pd.DataFrame,
    plan: FieldTransformationPlan,
    unique_keys: list[str],
) -> pd.Series:
    """
    Apply CollapseConfig to reduce student-term grain to student grain for one field.

    Returns a Series aligned to the collapsed DataFrame index — one row per
    unique key combination, index reset to RangeIndex for safe DataFrame assembly.
    """
    collapse = plan.collapse
    col = plan.source_columns[0] if plan.source_columns else None

    if collapse.strategy == CollapseStrategy.none:
        return df[col].reset_index(drop=True) if col else pd.Series(dtype="object")

    if collapse.strategy == CollapseStrategy.constant:
        return pd.Series(dtype="object")

    if collapse.strategy == CollapseStrategy.any_row:
        # Invariant — deduplicate to one row per unique key, take first
        return (
            df.drop_duplicates(subset=unique_keys, keep="first")[col]
            .reset_index(drop=True)
        )

    if collapse.strategy == CollapseStrategy.first_by:
        if not collapse.order_by:
            raise ExecutionError(
                f"first_by strategy requires order_by for field '{plan.target_field}'"
            )
        # Sort ascending then deduplicate — first row per unique key is the term-1 row
        return (
            df.sort_values(collapse.order_by, ascending=True)
            .drop_duplicates(subset=unique_keys, keep="first")[col]
            .reset_index(drop=True)
        )

    if collapse.strategy == CollapseStrategy.where_not_null:
        if not collapse.condition_col:
            raise ExecutionError(
                f"where_not_null strategy requires condition_col for field '{plan.target_field}'"
            )
        # Filter to rows where condition_col is non-null, then take first per unique key
        return (
            df[df[collapse.condition_col].notna()]
            .drop_duplicates(subset=unique_keys, keep="first")[col]
            .reset_index(drop=True)
        )

    raise ExecutionError(f"Unknown collapse strategy: {collapse.strategy}")


# -----------------------------------------------------------------------------
# Step dispatcher
# -----------------------------------------------------------------------------

def _dispatch_step(s: pd.Series, step: Any) -> pd.Series:
    """
    Dispatch a single TransformationStep to its utility function.

    Args:
        s: Input Series
        step: Typed TransformationStep model

    Returns:
        Transformed Series
    """
    fn = step.function_name

    if fn == "NEW_UTILITY_NEEDED":
        raise ExecutionGapError(
            f"NEW_UTILITY_NEEDED: {step.description}"
        )

    dispatch = {
        "cast_nullable_int":    lambda: cast_nullable_int(s),
        "cast_nullable_float":  lambda: cast_nullable_float(s),
        "cast_string":          lambda: cast_string(s),
        "cast_boolean":         lambda: cast_boolean(s, step.boolean_map),
        "cast_datetime":        lambda: cast_datetime(s),
        "coerce_numeric":       lambda: coerce_numeric(s),
        "coerce_datetime":      lambda: coerce_datetime(s, fmt=step.fmt),
        "strip_whitespace":     lambda: strip_whitespace(s),
        "lowercase":            lambda: lowercase(s),
        "uppercase":            lambda: uppercase(s),
        "map_values":           lambda: map_values(s, step.mapping),
        "normalize_term_code":  lambda: normalize_term_code(s),
        "normalize_grade":      lambda: normalize_grade(s),
        "normalize_enrollment": lambda: normalize_enrollment(s),
        "normalize_pell":       lambda: normalize_pell(s),
        "normalize_credential": lambda: normalize_credential(s),
        "normalize_student_age": lambda: normalize_student_age(s),
        "fill_nulls":           lambda: fill_nulls(s, step.value),
        "replace_null_tokens":  lambda: replace_null_tokens(s, step.null_tokens),
        "replace_values_with_null": lambda: replace_values_with_null(s, step.to_replace),
        "strip_trailing_decimal": lambda: strip_trailing_decimal(s),
    }

    if fn not in dispatch:
        raise ExecutionError(f"Unknown function_name: '{fn}'")

    return dispatch[fn]()


# -----------------------------------------------------------------------------
# DataFrame-level steps (combine_columns, deduplicate_rows)
# These operate on the full DataFrame rather than a single Series.
# -----------------------------------------------------------------------------

def _dispatch_df_step(df: pd.DataFrame, step: Any) -> pd.DataFrame:
    """Dispatch steps that operate on the full DataFrame."""
    fn = step.function_name

    if fn == "combine_columns":
        return combine_columns(df, step.cols, step.output_col, step.sep)

    if fn == "deduplicate_rows":
        return deduplicate_rows(df, subset=step.subset, keep=step.keep)

    raise ExecutionError(f"'{fn}' is not a DataFrame-level step")


_DF_LEVEL_STEPS = {"combine_columns", "deduplicate_rows"}


def execute_transformation_map(
    df: pd.DataFrame,
    transformation_map: TransformationMap,
    unique_keys: list[str],
    raise_on_gap: bool = False,
) -> ExecutionResult:
    """
    Execute a TransformationMap against a pre-joined DataFrame.

    Args:
        df: Pre-joined source DataFrame. For cohort maps this is student-term grain;
            for course maps this is already at course grain.
        transformation_map: Validated TransformationMap from Agent 2b.
        unique_keys: Groupby keys for collapse, from schema_contract.unique_keys
                     for the TARGET schema entity. For cohort: ["student_id"].
                     For course: ["student_id", "academic_term", "course_prefix", "course_number"].
        raise_on_gap: If True, raise ExecutionGapError on first NEW_UTILITY_NEEDED.
                      If False, skip gap fields and record them in result.gaps.

    Returns:
        ExecutionResult with assembled target DataFrame and execution metadata.
    """
    result_cols: dict[str, pd.Series] = {}
    gaps: list[str] = []
    skipped: list[str] = []
    executed: list[str] = []

    # --- Pre-collapse: DataFrame-level grain reduction before field plans run ---
    if transformation_map.pre_collapse:
        pc = transformation_map.pre_collapse
        if pc.order_by:
            df = df.sort_values(pc.order_by)
        df = df.drop_duplicates(subset=pc.subset, keep=pc.keep)
        logger.debug(
            f"Pre-collapse applied: deduped to {len(df)} rows "
            f"on {pc.subset} keep='{pc.keep}'"
            + (f" order_by='{pc.order_by}'" if pc.order_by else "")
        )

    # --- Determine the expected number of output rows ---
    # This is the number of unique key combinations in the input DataFrame
    # after any pre_collapse. All collapsed Series must align to this length.
    #
    # pre_collapse.subset uses SOURCE column names (pre-transformation).
    # unique_keys uses TARGET column names (post-transformation).
    # We must use source names here since transformations haven't run yet.
    #
    # TODO(performance): This runs a full drop_duplicates just to count rows.
    # For performance improvement, pre-compute the collapsed base DataFrame once
    # upfront and derive expected_n_rows from len(collapsed_base) instead of
    # running a separate deduplication pass here.
    dedup_subset = (
        transformation_map.pre_collapse.subset
        if transformation_map.pre_collapse
        else unique_keys
    )
    expected_n_rows = df.drop_duplicates(subset=dedup_subset).shape[0]
    logger.debug(
        f"Expected output rows: {expected_n_rows} "
        f"(dedup on {dedup_subset})"
    )

    for plan in transformation_map.plans:
        target = plan.target_field

        # --- Unmappable field: no steps, no source columns ---
        if not plan.steps and not plan.source_columns:
            logger.debug(f"Skipping unmappable field: {target}")
            skipped.append(target)
            continue

        # --- Check for gaps before executing ---
        gap_steps = [s for s in plan.steps if s.function_name == "NEW_UTILITY_NEEDED"]
        if gap_steps:
            msg = f"Field '{target}' has {len(gap_steps)} NEW_UTILITY_NEEDED step(s)"
            logger.warning(msg)
            if raise_on_gap:
                raise ExecutionGapError(msg)
            gaps.append(target)
            continue

        try:
            # --- Collapse: reduce to correct grain if needed ---
            if plan.collapse and plan.collapse.strategy not in (
                CollapseStrategy.none,
                CollapseStrategy.constant,
            ):
                s = _collapse_field(df, plan, unique_keys)

                # TODO(performance): _collapse_field calls drop_duplicates independently
                # per field, meaning the same deduplication runs once per collapsible field.
                # For performance improvement, pre-compute a collapsed base DataFrame once
                # per strategy+order_by combination before the plan loop, then each field
                # just selects its column from the pre-collapsed base rather than
                # re-running deduplication independently.

                # --- Reindex to expected_n_rows ---
                # where_not_null may produce fewer rows than expected (e.g. students
                # who never graduated are absent). Reindex to full row count so all
                # Series in result_cols have the same length for DataFrame assembly.
                # Missing rows get NaN/NA which is correct — null for unmapped students.
                if len(s) < expected_n_rows:
                    logger.debug(
                        f"Field '{target}': collapsed to {len(s)} rows "
                        f"(expected {expected_n_rows}) — reindexing with nulls for missing rows."
                    )
                    s = s.reindex(range(expected_n_rows))

            else:
                # No collapse — work directly on the source column
                col = plan.source_columns[0] if plan.source_columns else None
                s = df[col].reset_index(drop=True).copy() if col else pd.Series(
                    dtype="object", index=range(expected_n_rows)
                )

            # --- Apply transformation steps ---
            # Note: DataFrame-level steps (combine_columns, deduplicate_rows) operate
            # on df directly without copying. This is intentional for performance —
            # these steps are rare and the caller should not rely on df being unmodified
            # after execute_transformation_map returns.
            for step in plan.steps:
                if step.function_name in _DF_LEVEL_STEPS:
                    df = _dispatch_df_step(df, step)
                    s = df[step.output_col].reset_index(drop=True) if hasattr(step, "output_col") else s
                else:
                    s = _dispatch_step(s, step)

            result_cols[target] = s
            executed.append(target)
            logger.debug(f"Executed field: {target}")

        except ExecutionGapError:
            gaps.append(target)
            if raise_on_gap:
                raise
        except Exception as e:
            raise ExecutionError(f"Failed executing field '{target}': {e}") from e

    # --- Assemble target DataFrame ---
    # All Series in result_cols should have the same length (expected_n_rows)
    # due to reset_index(drop=True) and reindex above.
    # Validate lengths before assembly to catch any remaining mismatches.
    mismatched = {
        t: len(s) for t, s in result_cols.items()
        if len(s) != expected_n_rows
    }
    if mismatched:
        logger.warning(
            f"Series length mismatch before assembly — expected {expected_n_rows} rows. "
            f"Mismatched fields: {mismatched}. These fields may cause alignment issues."
        )

    target_df = pd.DataFrame(result_cols)

    return ExecutionResult(
        df=target_df,
        gaps=gaps,
        skipped=skipped,
        executed=executed,
    )