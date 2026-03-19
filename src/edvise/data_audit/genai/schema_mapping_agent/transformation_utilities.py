"""
Transformation utilities for SchemaMappingAgent transformation maps.

All utilities are pure Series → Series (or scalar → Series for fill_constant).
No DataFrame context, no cross-table joins — those are handled by field_executor.

Removed from previous version:
  - cross_table_lookup: join logic moved to JoinConfig on FieldMappingRecord
  - stems_lookup: STEM classification now handled via JoinConfig + map_values step
  - combine_columns: DataFrame-level operation, not compatible with Series-only model
  - deduplicate_rows: DataFrame-level operation, not compatible with Series-only model
"""

import re
import typing as t
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from edvise.data_audit.custom_cleaning import (
    _cast_series_to_nullable_dtype,
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

# =============================================================================
# Type Casting
# =============================================================================

def cast_nullable_dtype(
    s: pd.Series,
    dtype_str: str,
    boolean_map: dict[str, bool] | None = None,
) -> pd.Series:
    """
    Cast a Series to one of the supported nullable dtypes.
    Thin public wrapper around _cast_series_to_nullable_dtype.

    Args:
        s: Series to cast
        dtype_str: Target dtype — "Int64", "Float64", "boolean", "string", "datetime64[ns]"
        boolean_map: Optional mapping for boolean conversion
    """
    if boolean_map is None:
        boolean_map = {
            "true": True, "false": False,
            "yes": True, "no": False,
            "1": True, "0": False,
        }
    return _cast_series_to_nullable_dtype(s, dtype_str, boolean_map)


def cast_nullable_int(s: pd.Series) -> pd.Series:
    """
    Cast Series to nullable Int64.

    Handles float64 input (e.g. 3.0 → 3) by using pd.to_numeric + Series.astype
    rather than pd.array which enforces safe casting rules and rejects float64.
    """
    coerced = pd.to_numeric(s, errors="coerce")
    return coerced.astype("Int64")


def cast_nullable_float(s: pd.Series) -> pd.Series:
    """Cast Series to nullable Float64."""
    return cast_nullable_dtype(s, "Float64", {})


def cast_string(s: pd.Series) -> pd.Series:
    """Cast Series to nullable string."""
    return cast_nullable_dtype(s, "string", {})


def cast_boolean(s: pd.Series, boolean_map: dict[str, bool] | None = None) -> pd.Series:
    """
    Cast Series to nullable boolean.

    Default map: "true"/"false", "yes"/"no", "1"/"0" → True/False (case-insensitive).
    Pass boolean_map to override. Values not in the map → pd.NA.
    """
    return cast_nullable_dtype(s, "boolean", boolean_map)


def cast_datetime(s: pd.Series) -> pd.Series:
    """Cast Series to datetime64[ns]."""
    return cast_nullable_dtype(s, "datetime64[ns]", {})


# =============================================================================
# Coercion
# =============================================================================

def coerce_numeric(s: pd.Series) -> pd.Series:
    """
    Coerce Series to numeric, inferring Int64 or Float64.

    Returns Int64 if all non-null values are whole numbers, else Float64.
    Non-numeric values → pd.NA.

    Prefer cast_nullable_int or cast_nullable_float when the target dtype is known.
    Use this only when dtype should be inferred from the data.
    """
    num = pd.to_numeric(s, errors="coerce")
    non_na = num.dropna()
    if len(non_na) and np.all(np.isclose(non_na % 1, 0)):
        return num.astype("Int64")
    return num.astype("Float64")


def coerce_datetime(s: pd.Series, fmt: str | None = None) -> pd.Series:
    """
    Coerce Series to datetime. Series-level wrapper around pd.to_datetime.

    Args:
        s: Input Series
        fmt: strptime format string. None = pandas infers.
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


# =============================================================================
# String Operations
# =============================================================================

def strip_whitespace(s: pd.Series) -> pd.Series:
    """Strip leading/trailing whitespace from string Series."""
    return s.astype("string").str.strip()


def lowercase(s: pd.Series) -> pd.Series:
    """Convert string Series to lowercase."""
    return s.astype("string").str.lower()


def uppercase(s: pd.Series) -> pd.Series:
    """Convert string Series to uppercase."""
    return s.astype("string").str.upper()


# =============================================================================
# Null Handling (used by map_values)
# =============================================================================

def replace_values_with_null(
    s: pd.Series,
    to_replace: str | list[str],
) -> pd.Series:
    """Replace specified values with None/null."""
    return s.replace(to_replace=to_replace, value=None)


# =============================================================================
# Value Mapping
# =============================================================================

def map_values(
    s: pd.Series,
    mapping: dict,
    default: str | None = "passthrough",
) -> pd.Series:
    """
    Map values in Series using a dictionary.

    Args:
        s: Input Series
        mapping: {source_value: target_value} dict
        default: Behavior for unmapped entries.
            "passthrough" — keep original value (default)
            None — fill unmapped with NA
            Any other string — fill unmapped with that value
    
    Note:
        When a value is explicitly mapped to null (e.g., {"(Blank)": null}),
        that null is preserved even with default="passthrough". Only unmapped
        values are filled back. Null mappings are handled via replace_values_with_null
        internally for consistency.
    """
    # Split mapping into null and non-null mappings
    # Check for None, pd.NA, or NaN values
    null_mappings = [
        k for k, v in mapping.items()
        if v is None or v is pd.NA or (isinstance(v, float) and pd.isna(v))
    ]
    non_null_mapping = {k: v for k, v in mapping.items() if k not in null_mappings}
    
    # Apply null replacements first (using replace_values_with_null for consistency)
    if null_mappings:
        s = replace_values_with_null(s, null_mappings)
    
    # Apply non-null mappings
    if non_null_mapping:
        result = s.map(non_null_mapping)
    else:
        result = s.copy()
    
    # Handle default behavior for unmapped values
    if default == "passthrough":
        # Only fill nulls for values that were NOT in the mapping
        # Values explicitly mapped to null should stay null
        all_mapped_keys = set(mapping.keys())
        unmapped_mask = ~s.isin(all_mapped_keys)
        # Fill nulls only where original value was unmapped
        result = result.mask(unmapped_mask & result.isna(), s)
    elif default is not None:
        # Fill unmapped nulls with default value
        all_mapped_keys = set(mapping.keys())
        unmapped_mask = ~s.isin(all_mapped_keys)
        result = result.mask(unmapped_mask & result.isna(), default)
    
    return result


# =============================================================================
# Domain-Specific Normalization
# Re-exports from _edvise_shared with short aliases for use in transformation maps
# =============================================================================

# =============================================================================
# Domain-Specific Normalization
# Re-exports from _edvise_shared with short aliases for use in transformation maps
# =============================================================================

def normalize_term_code(s: pd.Series) -> pd.Series:
    """
    Normalize "Season YYYY" term description strings to canonical PDP term codes.

    Input:  "Spring 2020", "Fall 2019", "Summer 2021", "SP", "FA"
    Output: "SPRING",      "FALL",      "SUMMER",      "SPRING", "FALL"

    Unmapped values → pd.NA.

    Use this for institutions whose term columns contain natural-language season
    descriptions or short season codes (e.g. UCF's term_desc / term_descr columns).

    Do NOT use for YYYYTT format term codes (e.g. "2019SP", "2018FA") —
    use extract_term_season_from_term_code for those instead.
    """
    return term_series_to_pdp(s)


def normalize_grade(s: pd.Series) -> pd.Series:
    """
    Normalize raw grade strings for EDA: strip whitespace and uppercase.

    Input:  "a+", " B- ", "pass"
    Output: "A+", "B-",   "PASS"

    Does not map to canonical values or categorize pass/fail — use map_values
    for that. Use this as a cleaning step before map_values or conditional_credits.
    """
    return grade_series_normalized(s)


def normalize_enrollment(s: pd.Series) -> pd.Series:
    """
    Normalize enrollment_type strings to PDP categories.

    Input:  "First-time student", "Transfer",   "Re-Admit"
    Output: "FIRST-TIME",         "TRANSFER-IN", "RE-ADMIT"

    Matching is substring-based and case-insensitive:
        "First" / "Freshman" / "Time" → "FIRST-TIME"
        "Transfer"                    → "TRANSFER-IN"
        "Re-Admit" / "Readmit"        → "RE-ADMIT"

    Unmapped values → pd.NA.

    Use this when the source enrollment_type values already approximate PDP
    language. For institutions with bespoke codes (e.g. UCF's "Beginner - FTIC"),
    use map_values instead.
    """
    return enrollment_series_to_pdp(s)


def normalize_pell(s: pd.Series) -> pd.Series:
    """
    Normalize Pell status to PDP Y/N.

    Input:  "Yes", "No", "Y", "N", "yes"
    Output: "Y",   "N",  "Y", "N", "Y"

    Unmapped values → pd.NA.
    """
    return pell_series_to_pdp(s)


def normalize_credential(s: pd.Series) -> pd.Series:
    """
    Normalize credential/degree free text to canonical PDP-style values.

    Input:  "Bachelor's Degree", "BA",          "Associate of Arts", "Certification"
    Output: "Bachelor's",        "Bachelor's",  "Associate's",       "Certificate"

    Matching is substring-based and case-insensitive:
        "bachelor" / "ba" / "bs" → "Bachelor's"
        "associate" / "aa" / "as" / "aas" → "Associate's"
        "certificate" / "certification"   → "Certificate"

    Unmapped values → pd.NA.

    Use this when source values are free-text degree labels that loosely match
    canonical names. For institutions with exact known values, map_values gives
    more precise control.
    """
    return credential_degree_series_to_canonical(s)


def normalize_student_age(s: pd.Series) -> pd.Series:
    """
    Normalize student age to PDP-style buckets.

    Input:  18,                  22,        30,               "20 AND YOUNGER"
    Output: "20 AND YOUNGER",    ">20 - 24", "OLDER THAN 24", "20 AND YOUNGER"

    Accepts both numeric ages (13–100) and existing bucket phrase strings
    (case-insensitive passthrough). Unmapped values → pd.NA.

    Note: prefer birthyear_to_age_bucket when the source column is a birth year
    rather than a current age — it handles reference year extraction from
    YYYY-YY cohort strings internally.
    """
    return student_age_series_to_pdp(s)


# =============================================================================
# Null Handling
# =============================================================================

def fill_nulls(s: pd.Series, value: t.Any) -> pd.Series:
    """
    Fill existing null (pd.NA / NaN) values with a scalar.

    Use when you want a fallback value for missing data.
    To convert null-token strings (e.g. "(Blank)") to null first,
    use replace_null_tokens or replace_values_with_null before this step.
    """
    return s.fillna(value)


def replace_null_tokens(s: pd.Series, null_tokens: list[str]) -> pd.Series:
    """
    Replace null-token strings (e.g. "(Blank)", "N/A") with pd.NA.

    Use when the source data encodes missing values as a sentinel string
    rather than a true null. Accepts a list of tokens to replace.

    For a single token, replace_values_with_null is also available.
    """
    return s.replace(null_tokens, pd.NA)


# =============================================================================
# Specialized
# =============================================================================

def strip_trailing_decimal(s: pd.Series) -> pd.Series:
    """
    Strip trailing '.0' from string or numeric Series.

    Handles numeric inputs by converting to Int64 first to avoid
    float precision issues (e.g. 202101.0 → '202101').
    """
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64").astype("string")
    else:
        s = s.astype("string")
    return s.str.replace(r"\.0$", "", regex=True)


def fill_constant(s: pd.Series, value: str) -> pd.Series:
    """
    Fill all rows with a constant string value, ignoring existing content.

    Used for fields derivable as an institutional constant — e.g.
    credential_type_sought_year_1 at UCF where all students are Bachelor's seekers.

    Args:
        s: Input Series — used only for length
        value: Constant string value
    """
    return pd.Series([value] * len(s), dtype="string")


def normalize_year_range(s: pd.Series) -> pd.Series:
    """
    Normalize year range string to YYYY-YY format matching YEAR_PATTERN (^\\d{4}-\\d{2}$).

    Handles:
        "2018-2019" → "2018-19"
        "2018-19"   → "2018-19"  (passthrough)
        other       → null

    Args:
        s: String Series containing year range values
    """
    s = s.astype("string").str.strip()

    already_correct = s.str.match(r"^\d{4}-\d{2}$")
    full_range = s.str.extract(r"^(\d{4})-(\d{4})$")
    converted = (
        full_range[0] + "-" + full_range[1].str[-2:]
    ).where(full_range[0].notna())

    result = pd.Series(pd.NA, index=s.index, dtype="string")
    result = result.where(~already_correct, s)
    result = result.where(converted.isna(), converted)
    return result


def extract_year(s: pd.Series) -> pd.Series:
    """
    Extract the first 4-digit year from a string Series.

    Handles:
        "2018-2019" → "2018"
        "2018-19"   → "2018"
        "2018FA"    → "2018"
        "2018"      → "2018"
    """
    return (
        s.astype("string")
        .str.extract(r"(\d{4})", expand=False)
        .astype("string")
    )


def parse_yyyymm(s: pd.Series) -> pd.Series:
    """
    Parse YYYYMM string Series to datetime using the first day of the month.

    Expects values like "202301" (after strip_trailing_decimal removes ".0").
    Nulls and unparseable values become NaT.
    """
    return pd.to_datetime(
        s.astype("string"),
        format="%Y%m",
        errors="coerce",
    )


def parse_term_description(s: pd.Series) -> pd.Series:
    """
    Parse "Season YYYY" term description strings to datetime.

    Expects values like "Summer 2018", "Fall 2020", "Spring 2021".
    Uses the start month of each term season.
    Nulls and unparseable values become NaT.

    Season → start month mapping:
        FALL   → September (9)
        SPRING → February (2)
        SUMMER → June (6)
        WINTER → January (1)
    """
    TERM_MONTHS = {"FALL": 9, "SPRING": 2, "SUMMER": 6, "WINTER": 1, "JANUARY": 1}

    s_upper = s.astype("string").str.strip().str.upper()
    matches = s_upper.str.extract(
        r"^(SPRING|SUMMER|FALL|WINTER|JANUARY)\s+(\d{4})$",
        expand=True,
    )

    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    valid = matches[0].notna() & matches[1].notna()

    if valid.any():
        months = matches[0][valid].map(TERM_MONTHS)
        years = pd.to_numeric(matches[1][valid], errors="coerce")
        parseable = months.notna() & years.notna()

        if parseable.any():
            dates = pd.to_datetime(
                pd.DataFrame({
                    "year": years[parseable],
                    "month": months[parseable],
                    "day": 1,
                }),
                errors="coerce",
            )
            result[valid[valid].index[parseable]] = dates

    return result


def birthyear_to_age_bucket(
    s: pd.Series,
    reference_year_series: pd.Series | None = None,
) -> pd.Series:
    """
    Convert birthyear Series to PDP age bucket strings.

    PDP age buckets:
        "20 AND YOUNGER"  — age <= 20
        ">20 - 24"        — 21 <= age <= 24
        "OLDER THAN 24"   — age >= 25

    Args:
        s: Nullable Int64 Series of birth years
        reference_year_series: Optional Series of reference years. Accepts:
            - Integer years (e.g. 2023)
            - YYYY-YY format strings (e.g. "2023-24") — first 4 digits extracted
            If None, uses current year for all rows.
    """
    birthyear = pd.to_numeric(s, errors="coerce")

    if reference_year_series is not None:
        ref_numeric = pd.to_numeric(reference_year_series, errors="coerce")
        ref_from_str = pd.to_numeric(
            reference_year_series.astype("string").str.extract(r"(\d{4})", expand=False),
            errors="coerce",
        )
        reference_year = ref_numeric.fillna(ref_from_str).fillna(datetime.now().year)
    else:
        reference_year = datetime.now().year

    age = reference_year - birthyear

    def _bucket(a: float) -> str | None:
        if pd.isna(a):
            return None
        if a <= 20:
            return "20 AND YOUNGER"
        if a <= 24:
            return ">20 - 24"
        return "OLDER THAN 24"

    return age.map(_bucket).astype("string")


def extract_academic_year_from_term_code(s: pd.Series) -> pd.Series:
    """
    Extract academic year from YYYYTT format term codes and convert to YYYY-YY format.

    Academic year logic:
        - FA (Fall) terms start the academic year: '2018FA' -> '2018-19'
        - SP (Spring) terms end the prior academic year: '2019SP' -> '2018-19'
        - S1/S2 (Summer) terms are part of the prior academic year: '2018S1'/'2018S2' -> '2017-18'

    Args:
        s: String Series of term codes in YYYYTT format (e.g., '2018FA', '2019SP', '2018S1', '2018S2')

    Returns:
        String Series in YYYY-YY format matching YEAR_PATTERN (^\\d{4}-\\d{2}$), or pd.NA for invalid inputs
    """
    s = s.astype("string").str.strip().str.upper()

    # Extract year (first 4 digits) and season code (last 2 characters)
    year_match = s.str.extract(r"^(\d{4})", expand=False)
    season_match = s.str.extract(r"([A-Z0-9]{2})$", expand=False)

    # Convert year to numeric
    year_numeric = pd.to_numeric(year_match, errors="coerce")

    # Determine academic year start year based on season
    # FA (Fall) starts the academic year, so use the year as-is
    # SP (Spring) ends the prior academic year, so subtract 1
    # S1/S2 (Summer) are part of the prior academic year, so subtract 1
    academic_year_start = year_numeric.copy()
    is_spring_or_summer = season_match.isin(["SP", "S1", "S2"])
    academic_year_start = academic_year_start.where(
        ~is_spring_or_summer,
        academic_year_start - 1
    )

    # Calculate academic year end (last 2 digits of start year + 1)
    academic_year_end = (academic_year_start + 1).astype("Int64").astype("string").str[-2:]

    # Format as YYYY-YY
    result = (
        academic_year_start.astype("Int64").astype("string") + "-" + academic_year_end
    )

    # Return NA for invalid inputs (where year or season couldn't be extracted)
    return result.where(
        year_match.notna() & season_match.notna() & academic_year_start.notna(),
        pd.NA
    ).astype("string")


def extract_term_season_from_term_code(s: pd.Series) -> pd.Series:
    """
    Extract canonical term season from YYYYTT format term codes.

    Maps season codes to canonical PDP term categories:
        - 'FA' -> 'FALL'
        - 'SP' -> 'SPRING'
        - 'S1'/'S2' -> 'SUMMER'

    Args:
        s: String Series of term codes in YYYYTT format (e.g., '2018FA', '2019SP', '2018S1', '2018S2')

    Returns:
        String Series with values in ['FALL', 'SPRING', 'SUMMER'], or pd.NA for unmapped/invalid inputs
    """
    s = s.astype("string").str.strip().str.upper()

    # Extract season code (last 2 characters)
    season_match = s.str.extract(r"([A-Z0-9]{2})$", expand=False)

    # Map season codes to canonical terms
    season_mapping = {
        "FA": "FALL",
        "SP": "SPRING",
        "S1": "SUMMER",
        "S2": "SUMMER",
    }

    result = season_match.map(season_mapping)
    return result.astype("string")


def parse_term_code_to_datetime(s: pd.Series) -> pd.Series:
    """
    Parse YYYYTT format term codes to datetime using the first day of the term.

    Maps term codes to start dates:
        - '2018FA' -> 2018-09-01 (Fall starts in September)
        - '2019SP' -> 2019-01-01 (Spring starts in January)
        - '2018S1'/'2018S2' -> 2018-06-01 (Summer starts in June)

    Args:
        s: String Series of term codes in YYYYTT format (e.g., '2018FA', '2019SP', '2018S1', '2018S2')

    Returns:
        Datetime Series with first day of term, or pd.NaT for invalid inputs
    """
    s = s.astype("string").str.strip().str.upper()

    # Extract year (first 4 digits) and season code (last 2 characters)
    year_match = s.str.extract(r"^(\d{4})", expand=False)
    season_match = s.str.extract(r"([A-Z0-9]{2})$", expand=False)

    # Convert year to numeric
    year_numeric = pd.to_numeric(year_match, errors="coerce")

    # Map season codes to start months
    # FA (Fall) -> September (9)
    # SP (Spring) -> January (1)
    # S1/S2 (Summer) -> June (6)
    season_to_month = {
        "FA": 9,
        "SP": 1,
        "S1": 6,
        "S2": 6,
    }

    month_numeric = season_match.map(season_to_month)

    # Create datetime from year, month, day=1
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    valid = year_numeric.notna() & month_numeric.notna()

    if valid.any():
        dates = pd.to_datetime(
            pd.DataFrame({
                "year": year_numeric[valid],
                "month": month_numeric[valid],
                "day": 1,
            }),
            errors="coerce",
        )
        result[valid] = dates

    return result


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
    s: pd.Series,
    grade_series: pd.Series,
) -> pd.Series:
    """
    Calculate credits earned based on whether the grade is passing.

    Passing grade  → earned = s (credits_series value)
    Non-passing    → earned = 0.0

    Args:
        s: Numeric Series of credits attempted (already cast to Int64
           by a prior cast_nullable_int step in the transformation chain)
        grade_series: String Series of normalized grade values,
                      resolved from base_df via extra_columns

    Returns:
        Float64 Series of credits earned
    """
    grade_upper = grade_series.astype("string").str.strip().str.upper()
    credits_numeric = pd.to_numeric(s, errors="coerce").astype("Float64")
    return credits_numeric.where(
        grade_upper.isin(_PASSING_GRADES),
        other=0.0,
    ).astype("Float64")