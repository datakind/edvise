"""
Transformation step dispatcher for SchemaMappingAgent.

Pure step dispatch — receives a pre-resolved Series and applies a single
TransformationStep to it. No sourcing, no joining, no row selection.

All of those concerns live in:
    - manifest.schemas.FieldMappingRecord (sourcing spec)
    - execution.field_executor.execute_transformation_map (orchestration)

The two steps that need a second Series (birthyear_to_age_bucket,
conditional_credits) are handled in execution.field_executor._execute_step()
before reaching this dispatcher.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from edvise.feature_generation.term_code_display import (
    academic_term_category_from_term_code_display,
    academic_year_from_term_code_display,
)

from ..transformation.utilities import (
    cast_boolean,
    cast_datetime,
    cast_nullable_float,
    cast_nullable_int,
    cast_string,
    coerce_datetime,
    coerce_numeric,
    extract_academic_year_from_term_code,
    extract_term_season_from_term_code,
    extract_year,
    format_academic_year_from_calendar_year,
    parse_term_code_to_datetime,
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
    substring_after_first_delimiter,
    term_season_from_datetime,
    uppercase,
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
# Step dispatcher
# -----------------------------------------------------------------------------


def dispatch_step(
    s: pd.Series,
    step: Any,
) -> pd.Series:
    """
    Dispatch a single TransformationStep to its utility function.

    All steps here are pure Series → Series.
    birthyear_to_age_bucket and conditional_credits are NOT in this dispatch
    table — they require a second Series and are handled upstream in
    execution.field_executor._execute_step().

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

    if fn in ("birthyear_to_age_bucket", "conditional_credits"):
        raise ExecutionError(
            f"Step '{fn}' requires a second Series and must be handled by "
            f"execution.field_executor._execute_step() before reaching dispatch_step()."
        )

    dispatch = {
        "cast_nullable_int": lambda: cast_nullable_int(s),
        "cast_nullable_float": lambda: cast_nullable_float(s),
        "cast_string": lambda: cast_string(s),
        "cast_boolean": lambda: cast_boolean(
            s,
            step.boolean_map if hasattr(step, "boolean_map") else None,
        ),
        "cast_datetime": lambda: cast_datetime(s),
        "coerce_numeric": lambda: coerce_numeric(s),
        "coerce_datetime": lambda: coerce_datetime(
            s,
            fmt=step.fmt if hasattr(step, "fmt") else None,
        ),
        "strip_whitespace": lambda: strip_whitespace(s),
        "lowercase": lambda: lowercase(s),
        "uppercase": lambda: uppercase(s),
        "map_values": lambda: map_values(
            s,
            step.mapping,
            default=step.default,
        ),
        "normalize_term_code": lambda: normalize_term_code(s),
        "normalize_grade": lambda: normalize_grade(s),
        "normalize_enrollment": lambda: normalize_enrollment(s),
        "normalize_pell": lambda: normalize_pell(s),
        "normalize_credential": lambda: normalize_credential(s),
        "normalize_student_age": lambda: normalize_student_age(s),
        "fill_nulls": lambda: fill_nulls(s, step.value),
        "replace_null_tokens": lambda: replace_null_tokens(s, step.null_tokens),
        "replace_values_with_null": lambda: replace_values_with_null(
            s, step.to_replace
        ),
        "strip_trailing_decimal": lambda: strip_trailing_decimal(s),
        "fill_constant": lambda: fill_constant(s, step.value),
        "normalize_year_range": lambda: normalize_year_range(s),
        "extract_year": lambda: extract_year(s),
        "format_academic_year_from_calendar_year": lambda: format_academic_year_from_calendar_year(
            s
        ),
        "term_season_from_datetime": lambda: term_season_from_datetime(s),
        "substring_after_first_delimiter": lambda: substring_after_first_delimiter(
            s,
            delimiter=step.delimiter,
        ),
        "parse_yyyymm": lambda: parse_yyyymm(s),
        "parse_term_description": lambda: parse_term_description(s),
        "extract_academic_year_from_term_code": lambda: extract_academic_year_from_term_code(
            s
        ),
        "extract_term_season_from_term_code": lambda: extract_term_season_from_term_code(
            s
        ),
        "academic_year_from_term_code_display": lambda: academic_year_from_term_code_display(
            s
        ),
        "academic_term_category_from_term_code_display": lambda: academic_term_category_from_term_code_display(
            s
        ),
        "parse_term_code_to_datetime": lambda: parse_term_code_to_datetime(s),
    }

    if fn not in dispatch:
        raise ExecutionError(
            f"Unknown function_name: '{fn}'. "
            f"Add it to the dispatch table in execution.step_dispatcher.dispatch_step()."
        )

    return dispatch[fn]()
