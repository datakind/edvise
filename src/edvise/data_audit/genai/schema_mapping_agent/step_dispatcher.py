"""
Transformation step dispatcher for SchemaMappingAgent.

Responsibilities (narrowed after field executor refactor):
  - Dispatch individual TransformationStep models to utility functions
  - Apply CollapseConfig to reduce student-term grain to student grain
  - Surface NEW_UTILITY_NEEDED gaps

What this module no longer does (moved to field_executor.py):
  - Join resolution — handled by JoinConfig on FieldMappingRecord
  - Cross-table lookups — handled by field_executor.resolve_source_series()
  - Pre-collapse DataFrame grain reduction — removed (JoinConfig.keep handles it)
  - TransformationMap orchestration — moved to field_executor.execute_transformation_map()

All transformation steps are pure Series → Series.
The two exceptions handled by field_executor before steps run:
  - birthyear_to_age_bucket: reference_year_column resolved from base_df,
    passed as second Series argument
  - conditional_credits: grade_column resolved from base_df,
    passed as second Series argument
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from edvise.data_audit.genai.schema_mapping_agent.transformation_utilities import (
    birthyear_to_age_bucket,
    cast_boolean,
    cast_datetime,
    cast_nullable_float,
    cast_nullable_int,
    cast_string,
    conditional_credits,
    coerce_datetime,
    coerce_numeric,
    extract_year,
    fill_constant,
    fill_nulls,
    lowercase,
    map_values,
    normalize_credential,
    normalize_enrollment,
    normalize_grade,
    normalize_pell,
    normalize_student_age,
    normalize_term_code,
    normalize_year_range,
    parse_term_description,
    parse_yyyymm,
    replace_null_tokens,
    replace_values_with_null,
    strip_trailing_decimal,
    strip_whitespace,
    uppercase,
)
from edvise.data_audit.genai.schema_mapping_agent.mapping_schemas import (
    CollapseStrategy,
    FieldTransformationPlan,
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
    gaps: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)

    @property
    def has_gaps(self) -> bool:
        return bool(self.gaps)


# -----------------------------------------------------------------------------
# Collapse
# -----------------------------------------------------------------------------

def apply_collapse(
    s: pd.Series,
    plan: FieldTransformationPlan,
    base_df: pd.DataFrame,
    unique_keys: list[str],
    expected_n_rows: int,
) -> pd.Series:
    """
    Apply CollapseConfig to reduce student-term grain to student grain.

    Args:
        s: Resolved source Series (pre-transformation, aligned to base_df index)
        plan: FieldTransformationPlan with collapse config
        base_df: Base DataFrame — used for order_by and condition_col access
        unique_keys: Schema contract unique keys — groupby keys for collapse
        expected_n_rows: Expected output row count after collapse

    Returns:
        Series collapsed to student grain, reset to RangeIndex
    """
    collapse = plan.collapse

    if collapse.strategy == CollapseStrategy.none:
        return s.reset_index(drop=True)

    if collapse.strategy == CollapseStrategy.constant:
        # fill_constant step produces the correct length — no collapse needed
        return s.reset_index(drop=True)

    if collapse.strategy == CollapseStrategy.any_row:
        result = (
            base_df.assign(_s=s.values)
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )
        return _reindex_to_expected(result, expected_n_rows, plan.target_field)

    if collapse.strategy == CollapseStrategy.first_by:
        order_col = collapse.order_by
        if order_col not in base_df.columns:
            raise ExecutionError(
                f"first_by order_by column '{order_col}' not found in base DataFrame "
                f"for field '{plan.target_field}'"
            )
        result = (
            base_df.assign(_s=s.values)
            .sort_values(order_col, ascending=True)
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )
        return _reindex_to_expected(result, expected_n_rows, plan.target_field)

    if collapse.strategy == CollapseStrategy.where_not_null:
        cond_col = collapse.condition_col
        if cond_col not in base_df.columns:
            raise ExecutionError(
                f"where_not_null condition_col '{cond_col}' not found in base DataFrame "
                f"for field '{plan.target_field}'"
            )
        result = (
            base_df.assign(_s=s.values)
            .loc[base_df[cond_col].notna()]
            .drop_duplicates(subset=unique_keys, keep="first")["_s"]
            .reset_index(drop=True)
        )
        # where_not_null may produce fewer rows than expected — reindex with nulls
        return _reindex_to_expected(result, expected_n_rows, plan.target_field)

    raise ExecutionError(
        f"Unknown collapse strategy: {collapse.strategy} for field '{plan.target_field}'"
    )


def _reindex_to_expected(
    s: pd.Series,
    expected_n_rows: int,
    target_field: str,
) -> pd.Series:
    """Reindex Series to expected_n_rows, filling gaps with NA."""
    if len(s) < expected_n_rows:
        logger.debug(
            f"Field '{target_field}': collapsed to {len(s)} rows "
            f"(expected {expected_n_rows}) — reindexing with nulls for missing rows"
        )
        return s.reindex(range(expected_n_rows))
    return s


# -----------------------------------------------------------------------------
# Step dispatcher
# -----------------------------------------------------------------------------

def dispatch_step(
    s: pd.Series,
    step: Any,
) -> pd.Series:
    """
    Dispatch a single TransformationStep to its utility function.

    All steps are pure Series → Series. Steps that require a second Series
    (birthyear_to_age_bucket, conditional_credits) are handled upstream in
    field_executor._execute_step() before this dispatcher is called.

    Args:
        s: Input Series from prior step in the chain
        step: Typed TransformationStep model

    Returns:
        Transformed Series
    """
    fn = step.function_name

    if fn == "NEW_UTILITY_NEEDED":
        raise ExecutionGapError(
            f"NEW_UTILITY_NEEDED: {getattr(step, 'description', '(no description)')}"
        )

    dispatch = {
        "cast_nullable_int":        lambda: cast_nullable_int(s),
        "cast_nullable_float":      lambda: cast_nullable_float(s),
        "cast_string":              lambda: cast_string(s),
        "cast_boolean":             lambda: cast_boolean(
                                        s,
                                        step.boolean_map if hasattr(step, "boolean_map") else None,
                                    ),
        "cast_datetime":            lambda: cast_datetime(s),
        "coerce_numeric":           lambda: coerce_numeric(s),
        "coerce_datetime":          lambda: coerce_datetime(
                                        s,
                                        fmt=step.fmt if hasattr(step, "fmt") else None,
                                    ),
        "strip_whitespace":         lambda: strip_whitespace(s),
        "lowercase":                lambda: lowercase(s),
        "uppercase":                lambda: uppercase(s),
        "map_values":               lambda: map_values(
                                        s,
                                        step.mapping,
                                        default=step.default,
                                    ),
        "normalize_term_code":      lambda: normalize_term_code(s),
        "normalize_grade":          lambda: normalize_grade(s),
        "normalize_enrollment":     lambda: normalize_enrollment(s),
        "normalize_pell":           lambda: normalize_pell(s),
        "normalize_credential":     lambda: normalize_credential(s),
        "normalize_student_age":    lambda: normalize_student_age(s),
        "fill_nulls":               lambda: fill_nulls(s, step.value),
        "replace_null_tokens":      lambda: replace_null_tokens(s, step.null_tokens),
        "replace_values_with_null": lambda: replace_values_with_null(s, step.to_replace),
        "strip_trailing_decimal":   lambda: strip_trailing_decimal(s),
        "fill_constant":            lambda: fill_constant(s, step.value),
        "normalize_year_range":     lambda: normalize_year_range(s),
        "extract_year":             lambda: extract_year(s),
        "parse_yyyymm":             lambda: parse_yyyymm(s),
        "parse_term_description":   lambda: parse_term_description(s),
        # birthyear_to_age_bucket and conditional_credits are NOT in this dispatch
        # table — they require a second Series and are handled in
        # field_executor._execute_step() before the step chain reaches here.
    }

    if fn in ("birthyear_to_age_bucket", "conditional_credits"):
        raise ExecutionError(
            f"Step '{fn}' requires a second Series and must be handled by "
            f"field_executor._execute_step(), not dispatch_step(). "
            f"Check that field_executor is calling dispatch_step correctly."
        )

    if fn not in dispatch:
        raise ExecutionError(
            f"Unknown function_name: '{fn}'. "
            f"Add it to the dispatch table in transformation_executor.dispatch_step()."
        )

    return dispatch[fn]()