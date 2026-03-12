"""
Basic transformation utilities for Agent 2 transformation maps.

This module provides atomic, reusable transformation functions that can be chained
together in transformation maps. It reuses existing utilities from the codebase to
maintain DRY principles - most functions are thin wrappers or direct re-exports.

All utilities operate on pandas Series or DataFrames and return transformed Series/DataFrames.
"""

import typing as t
import warnings
from datetime import datetime

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
    # New transformation utilities
    "fill_constant",
    "extract_year",
    "parse_yyyymm",
    "birthyear_to_age_bucket",
    "conditional_credits",
    "stems_lookup",
    "cross_table_lookup",
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
#
# IMPORTANT: These functions transform data to Edvise schema format!
# The Edvise schemas (RawEdviseStudentDataSchema, RawEdviseCourseDataSchema)
# use PDP categories as their canonical values:
#   - Terms: FALL, WINTER, SPRING, SUMMER
#   - Enrollment: FIRST-TIME, RE-ADMIT, TRANSFER-IN
#   - Pell: Y, N
#   - Credentials: Bachelor's, Associate's, Certificate
#
# These functions are called automatically by schema.validate(), so using them
# in transformation maps ensures data matches Edvise schema requirements.

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



# =============================================================================
# fill_constant
# =============================================================================

def fill_constant(s: pd.Series, value: str) -> pd.Series:
    """
    Fill all rows with a constant string value, regardless of existing content.

    Used for fields that can be safely derived as a constant for all rows
    based on institutional context — e.g. credential_type_sought_year_1
    at  where ugrd_grad_flag confirms all students are Bachelor's seekers.

    Args:
        s: Input Series (ignored — output is constant for all rows)
        value: Constant string value to fill

    Returns:
        Series of same length as s, all values equal to value
    """
    return pd.Series([value] * len(s), dtype="string")


# =============================================================================
# extract_year
# =============================================================================

def extract_year(s: pd.Series) -> pd.Series:
    """
    Extract the first 4-digit year from a string Series.

    Handles formats like:
      - "2018-2019" -> "2018"
      - "2018-19"   -> "2018"
      - "2018"      -> "2018"

    Args:
        s: String Series containing year range or year string

    Returns:
        String Series with first 4-digit year extracted, null where no match
    """
    return (
        s.astype("string")
        .str.extract(r"(\d{4})", expand=False)
        .astype("string")
    )


# =============================================================================
# parse_yyyymm
# =============================================================================

def parse_yyyymm(s: pd.Series) -> pd.Series:
    """
    Parse a YYYYMM string Series to datetime, using the first day of the month.

    Expects values like "202301" (after strip_trailing_decimal removes ".0").
    Nulls and unparseable values become NaT.

    Args:
        s: String Series in YYYYMM format

    Returns:
        datetime64[ns] Series, first day of each month
    """
    return pd.to_datetime(
        s.astype("string"),
        format="%Y%m",
        errors="coerce",
    )


# =============================================================================
# birthyear_to_age_bucket
# =============================================================================

def birthyear_to_age_bucket(s: pd.Series) -> pd.Series:
    """
    Convert a birthyear Series to PDP age bucket strings using current year
    as the reference year.

    PDP age buckets:
      - "20 AND YOUNGER"  : age <= 20
      - ">20 - 24"        : 21 <= age <= 24
      - "OLDER THAN 24"   : age >= 25

    Null birthyears produce null bucket values.

    Args:
        s: Nullable Int64 Series of birth years (e.g. 1999, 2002)

    Returns:
        String Series with PDP age bucket values
    """
    reference_year = datetime.now().year
    age = pd.to_numeric(s, errors="coerce")
    age = reference_year - age

    def _bucket(a: float) -> str | None:
        if pd.isna(a):
            return None
        if a <= 20:
            return "20 AND YOUNGER"
        if a <= 24:
            return ">20 - 24"
        return "OLDER THAN 24"

    return age.map(_bucket).astype("string")


# =============================================================================
# conditional_credits
# =============================================================================

# Passing grades per Edvise schema ALLOWED_GRADES
_PASSING_GRADES = frozenset([
    "A", "A+", "A-",
    "B", "B+", "B-",
    "C", "C+", "C-",
    "D", "D+", "D-",
    "P", "PASS",
    "S", "SAT",
])


def conditional_credits(
    grade_series: pd.Series,
    credits_series: pd.Series,
) -> pd.Series:
    """
    Calculate credits earned based on whether the grade is passing.

    If grade is in the passing set → earned = attempted (credits_series value)
    Otherwise (F, W, WD, U, UNSAT, I, IP, etc.) → earned = 0.0

    Args:
        grade_series: String Series of normalized grade values
        credits_series: Numeric Series of credits attempted

    Returns:
        Float64 Series of credits earned
    """
    grade_upper = grade_series.astype("string").str.strip().str.upper()
    credits_numeric = pd.to_numeric(credits_series, errors="coerce").astype("Float64")

    earned = credits_numeric.where(grade_upper.isin(_PASSING_GRADES), other=0.0)
    return earned.astype("Float64")


# =============================================================================
# stems_lookup
# =============================================================================

def stems_lookup(
    course_cip_series: pd.Series,
    stems_df: pd.DataFrame,
    cip_col: str = "cip",
    area_col: str = "area",
    stem_value: str = "STEM",
    stem_output: str = "STEM",
    non_stem_output: str = "Non-STEM",
) -> pd.Series:
    """
    Classify courses as STEM or Non-STEM by joining to a STEM definition lookup table.

    Handles the specific issue where course_cip is inferred as datetime64[ns]
    by pandas — recovers the numeric CIP code before joining.

    Args:
        course_cip_series: Series of CIP codes (may be datetime64[ns] due to dtype misfire)
        stems_df: STEM definition DataFrame with cip and area columns
        cip_col: Column name in stems_df to join on (default: "cip")
        area_col: Column name in stems_df containing STEM classification (default: "area")
        stem_value: Value in area_col that indicates STEM (default: "STEM")
        stem_output: Output value for STEM courses (default: "STEM")
        non_stem_output: Output value for non-STEM courses (default: "Non-STEM")

    Returns:
        String Series with stem_output or non_stem_output values, null where CIP is null
    """
    # --- Fix datetime dtype misfire ---
    # pandas infers course_cip as datetime64[ns] because CIP codes like
    # 40.0801 get parsed as dates. Recover the numeric value by extracting
    # the year component which encodes the original integer CIP code.
    if pd.api.types.is_datetime64_any_dtype(course_cip_series):
        # CIP codes like "2040-08-01" encode 40 (year=2040, month=08, day=01)
        # Recover by extracting year - 2000 to get the 2-digit CIP prefix
        # then reconstruct: year*100 + month gives approximate CIP integer
        dt = pd.to_datetime(course_cip_series, errors="coerce")
        # Reconstruct as float: year-2000 gives major, month/100 gives minor
        # e.g. 2040-08-01 -> (2040-2000)*100 + 8 = 4008 -> 40.08 as float
        cip_numeric = ((dt.dt.year - 2000) * 100 + dt.dt.month) / 100
        cip_numeric = cip_numeric.astype("Float64")
    else:
        cip_numeric = pd.to_numeric(course_cip_series, errors="coerce").astype("Float64")

    # --- Join to stems_def_df ---
    lookup = stems_df[[cip_col, area_col]].copy()
    lookup[cip_col] = pd.to_numeric(lookup[cip_col], errors="coerce").astype("Float64")

    merged = cip_numeric.rename("_cip").to_frame().merge(
        lookup,
        left_on="_cip",
        right_on=cip_col,
        how="left",
    )

    # --- Map to output values ---
    area = merged[area_col].astype("string")
    result = area.map(lambda a: stem_output if a == stem_value else (non_stem_output if pd.notna(a) else None))
    return result.astype("string")


# =============================================================================
# cross_table_lookup
# =============================================================================

def cross_table_lookup(
    base_series: pd.Series,
    base_join_keys: list[str],
    base_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    lookup_join_keys: list[str],
    lookup_value_col: str,
) -> pd.Series:
    """
    Pull a column from a lookup DataFrame by joining on specified keys.

    Used for cross-table fields where the value exists in a different table
    than the base — e.g. term_major in the course schema comes from student_df
    joined on student_id + term.

    Args:
        base_series: Not used directly — length reference for output alignment
        base_join_keys: Column names in base_df to join on
        base_df: The base DataFrame (already pre-collapsed/filtered)
        lookup_df: The DataFrame containing the value to pull
        lookup_join_keys: Column names in lookup_df to join on (same order as base_join_keys)
        lookup_value_col: Column in lookup_df to pull as the output value

    Returns:
        String Series aligned to base_df index with pulled values, null where no match
    """
    # Select only the join keys + value column from lookup to avoid column collisions
    lookup_slim = lookup_df[lookup_join_keys + [lookup_value_col]].drop_duplicates(
        subset=lookup_join_keys, keep="first"
    )

    merged = base_df[base_join_keys].merge(
        lookup_slim,
        left_on=base_join_keys,
        right_on=lookup_join_keys,
        how="left",
    )

    return merged[lookup_value_col].astype("string").reset_index(drop=True)