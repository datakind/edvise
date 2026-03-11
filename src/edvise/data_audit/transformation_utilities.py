"""
Basic transformation utilities for Agent 2 transformation maps.

This module provides atomic, reusable transformation functions that can be chained
together in transformation maps. It reuses existing utilities from the codebase to
maintain DRY principles - most functions are thin wrappers or direct re-exports.

All utilities operate on pandas Series or DataFrames and return transformed Series/DataFrames.
"""

import typing as t
import warnings

import numpy as np
import pandas as pd

# Import existing utilities to reuse (DRY)
from edvise.data_audit.custom_cleaning import (
    _cast_series_to_nullable_dtype,  # Will expose as public
    assign_numeric_grade,
    normalize_columns,
)
from edvise.data_audit.schemas._edvise_shared import (
    credential_degree_series_to_canonical,
    enrollment_series_to_pdp,
    grade_series_normalized,
    pell_series_to_pdp,
    student_age_series_to_pdp,
    term_series_to_pdp,
)

# Re-export existing utilities with their original names
__all__ = [
    # Column operations
    "normalize_columns",
    # Type casting (exposed from private function)
    "cast_nullable_dtype",
    "cast_nullable_int",
    "cast_nullable_float",
    "cast_string",
    "cast_boolean",
    "cast_datetime",
    # Coercion
    "coerce_numeric",
    "coerce_datetime",
    # String operations
    "strip_whitespace",
    "lowercase",
    "uppercase",
    # Value mapping
    "map_values",
    # Domain-specific normalization (re-exported with aliases)
    "normalize_term_code",
    "normalize_grade",
    "normalize_enrollment",
    "normalize_pell",
    "normalize_credential",
    "normalize_student_age",
    # Null handling
    "fill_nulls",
    "replace_null_tokens",
    "replace_values_with_null",
    # Column combination
    "combine_columns",
    # Deduplication
    "deduplicate_rows",
    # Specialized
    "assign_numeric_grade",
    "strip_trailing_decimal",
]


# ============================================================================
# Type Casting Utilities
# ============================================================================
# These expose the private _cast_series_to_nullable_dtype as public utilities

def cast_nullable_dtype(
    s: pd.Series,
    dtype_str: str,
    boolean_map: dict[str, bool] | None = None,
) -> pd.Series:
    """
    Cast a Series to one of our supported nullable dtypes.
    
    This is a public wrapper around the existing _cast_series_to_nullable_dtype
    to maintain DRY - we don't duplicate the casting logic.
    
    Args:
        s: Series to cast
        dtype_str: Target dtype ("Int64", "Float64", "boolean", "string", "datetime64[ns]")
        boolean_map: Optional mapping for boolean conversion (defaults to standard map)
    
    Returns:
        Series with the specified nullable dtype
    """
    if boolean_map is None:
        boolean_map = {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        }
    return _cast_series_to_nullable_dtype(s, dtype_str, boolean_map)


def cast_nullable_int(s: pd.Series) -> pd.Series:
    """Cast Series to nullable Int64. Thin wrapper around cast_nullable_dtype."""
    return cast_nullable_dtype(s, "Int64", {})


def cast_nullable_float(s: pd.Series) -> pd.Series:
    """Cast Series to nullable Float64. Thin wrapper around cast_nullable_dtype."""
    return cast_nullable_dtype(s, "Float64", {})


def cast_string(s: pd.Series) -> pd.Series:
    """Cast Series to nullable string. Thin wrapper around cast_nullable_dtype."""
    return cast_nullable_dtype(s, "string", {})


def cast_boolean(s: pd.Series, boolean_map: dict[str, bool] | None = None) -> pd.Series:
    """Cast Series to nullable boolean. Thin wrapper around cast_nullable_dtype."""
    return cast_nullable_dtype(s, "boolean", boolean_map)


def cast_datetime(s: pd.Series) -> pd.Series:
    """Cast Series to datetime64[ns]. Thin wrapper around cast_nullable_dtype."""
    return cast_nullable_dtype(s, "datetime64[ns]", {})


# ============================================================================
# Coercion Utilities
# ============================================================================

def coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Coerce Series to numeric, returning Int64 if all values are integers, else Float64.
    
    Reuses logic from _cast_series_to_nullable_dtype to maintain consistency.
    """
    num = pd.to_numeric(s, errors="coerce")
    non_na = num.dropna()
    if len(non_na) and np.all(np.isclose(non_na % 1, 0)):
        return num.astype("Int64")
    return num.astype("Float64")


def coerce_datetime(s: pd.Series, fmt: str | None = None) -> pd.Series:
    """
    Coerce Series to datetime.
    
    This is a Series-level wrapper around pandas to_datetime.
    The existing parse_dttm_values works on DataFrame+col, but we need Series-level.
    """
    if fmt:
        return pd.to_datetime(s, format=fmt, errors="coerce")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually",
            category=UserWarning,
        )
        return pd.to_datetime(s, errors="coerce")


# ============================================================================
# String Operations
# ============================================================================
# These are simple pandas operations - no need to duplicate logic

def strip_whitespace(s: pd.Series) -> pd.Series:
    """Strip whitespace from string Series. Simple pandas wrapper."""
    return s.astype("string").str.strip()


def lowercase(s: pd.Series) -> pd.Series:
    """Convert string Series to lowercase. Simple pandas wrapper."""
    return s.astype("string").str.lower()


def uppercase(s: pd.Series) -> pd.Series:
    """Convert string Series to uppercase. Simple pandas wrapper."""
    return s.astype("string").str.upper()


# ============================================================================
# Value Mapping
# ============================================================================

def map_values(s: pd.Series, mapping: dict) -> pd.Series:
    """Map values in Series using provided mapping. Simple pandas wrapper."""
    return s.map(mapping)


# ============================================================================
# Domain-Specific Normalization
# ============================================================================
# Re-export existing functions with shorter aliases for transformation maps

normalize_term_code = term_series_to_pdp
normalize_grade = grade_series_normalized
normalize_enrollment = enrollment_series_to_pdp
normalize_pell = pell_series_to_pdp
normalize_credential = credential_degree_series_to_canonical
normalize_student_age = student_age_series_to_pdp


# ============================================================================
# Null Handling
# ============================================================================

def fill_nulls(s: pd.Series, value: t.Any) -> pd.Series:
    """Fill null values in Series. Simple pandas wrapper."""
    return s.fillna(value)


def replace_null_tokens(s: pd.Series, null_tokens: list[str]) -> pd.Series:
    """
    Replace null token strings with pd.NA.
    
    Extracts the logic from clean_dataset() to make it reusable.
    This is the only place we extract logic (not duplicate) - it's a simple replace.
    """
    return s.replace(null_tokens, pd.NA)


def replace_values_with_null(s: pd.Series, to_replace: str | list[str]) -> pd.Series:
    """
    Replace specified values with None/null.
    
    This is a Series-level wrapper. The existing replace_values_with_null in
    utils/data_cleaning.py works on DataFrame+col, but we need Series-level.
    """
    return s.replace(to_replace=to_replace, value=None)


# ============================================================================
# Column Operations
# ============================================================================

def combine_columns(
    df: pd.DataFrame, cols: list[str], output_col: str, sep: str = ""
) -> pd.DataFrame:
    """
    Combine multiple columns into a single column.
    
    This is a new utility, but uses standard pandas operations (no duplication).
    """
    df = df.copy()
    df[output_col] = df[cols].astype("string").agg(sep.join, axis=1)
    return df


# ============================================================================
# Deduplication
# ============================================================================

def deduplicate_rows(
    df: pd.DataFrame, subset: list[str] | None = None, keep: str = "first"
) -> pd.DataFrame:
    """Remove duplicate rows. Simple pandas wrapper."""
    return df.drop_duplicates(subset=subset, keep=keep)


# ============================================================================
# Specialized Utilities
# ============================================================================
# Re-export existing specialized functions

# assign_numeric_grade is already imported and re-exported in __all__


def strip_trailing_decimal(s: pd.Series) -> pd.Series:
    """
    Strip trailing ".0" from string Series.
    
    This is a Series-level version of strip_trailing_decimal_strings.
    The existing function works on DataFrame with specific columns, but we need Series-level.
    """
    return s.astype("string").str.replace(r"\.0$", "", regex=True)
