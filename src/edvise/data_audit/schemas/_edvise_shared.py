# ruff: noqa: F821
# mypy: ignore-errors
"""Shared regex patterns, Field definitions, and PDP-compat transforms for Edvise raw schemas."""

import math
import re
from decimal import ROUND_FLOOR, Decimal, InvalidOperation
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import pandera as pda
except ModuleNotFoundError:
    import edvise.utils as utils

    utils.databricks.mock_pandera()
    import pandera as pda

# Year format YYYY-YY (cohort, academic_year)
YEAR_PATTERN = re.compile(r"^\d{4}-\d{2}$")

# Term name, e.g. Fall, Fall 2023, SP (cohort_term, academic_term)
TERM_PATTERN = re.compile(
    r"(?i)^(\d{4})?\s?(Fall|Winter|Spring|Summer|FA|WI|SP|SU|SM)\s?(\d{4})?$"
)

StudentIdField = pda.Field(nullable=False, str_length={"min_value": 1})

# ---------------------------------------------------------------------------
# Term
# ---------------------------------------------------------------------------

TERM_CATEGORIES = ["FALL", "WINTER", "SPRING", "SUMMER"]


def _term_to_pdp_val(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().upper()
    if "FALL" in s or s in ("FA", "FALL"):
        return "FALL"
    if "WINTER" in s or s in ("WI", "WINTER"):
        return "WINTER"
    if "SPRING" in s or s in ("SP", "SPRING"):
        return "SPRING"
    if "SUMMER" in s or s in ("SU", "SM", "SUMMER"):
        return "SUMMER"
    return None


def term_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map term strings to PDP categories: FALL, WINTER, SPRING, SUMMER.

    Args:
        series: Raw term labels (e.g. "Fall", "SP", "Fall 2023").

    Returns:
        String series with values in TERM_CATEGORIES, or pd.NA where unmapped.
    """
    return series.astype(str).apply(_term_to_pdp_val).astype(pd.StringDtype())


# ---------------------------------------------------------------------------
# Enrollment (kept for other consumers; not used in student schema validation)
# ---------------------------------------------------------------------------

ENROLLMENT_CATEGORIES = ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"]


def _enrollment_to_pdp_val(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return None
    s = s.strip().upper()
    if "FIRST" in s or "FRESHMAN" in s or "TIME" in s:
        return "FIRST-TIME"
    if "TRANSFER" in s:
        return "TRANSFER-IN"
    if "RE-ADMIT" in s or "READMIT" in s:
        return "RE-ADMIT"
    return None


def enrollment_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map enrollment_type strings to PDP categories: FIRST-TIME, RE-ADMIT, TRANSFER-IN.

    Args:
        series: Raw enrollment labels (e.g. "First-time student", "Transfer").

    Returns:
        String series with values in ENROLLMENT_CATEGORIES, or pd.NA where unmapped.
    """
    return series.astype(str).apply(_enrollment_to_pdp_val).astype(pd.StringDtype())


# ---------------------------------------------------------------------------
# Student age
# ---------------------------------------------------------------------------

STUDENT_AGE_20_AND_YOUNGER = "20 AND YOUNGER"
STUDENT_AGE_20_24 = ">20 - 24"
STUDENT_AGE_OLDER_THAN_24 = "OLDER THAN 24"

# Canonical labels for bias / fair-lending style analysis (use with schema isin).
LEARNER_AGE_BUCKETS: tuple[str, str, str] = (
    STUDENT_AGE_20_AND_YOUNGER,
    STUDENT_AGE_20_24,
    STUDENT_AGE_OLDER_THAN_24,
)


def _numeric_age_to_bucket(n: int) -> Optional[str]:
    if 13 <= n <= 20:
        return STUDENT_AGE_20_AND_YOUNGER
    if 21 <= n <= 24:
        return STUDENT_AGE_20_24
    if 25 <= n <= 100:
        return STUDENT_AGE_OLDER_THAN_24
    return None


def _learner_age_phrase_to_bucket(s: str) -> Optional[str]:
    """Map free-text age phrases to PDP buckets; s is non-empty stripped string."""
    lower = s.lower()
    if (
        "younger" in lower
        or "20 and" in lower
        or lower in ("<=20", "under 21", "under21")
    ):
        return STUDENT_AGE_20_AND_YOUNGER
    if "older" in lower and "24" in lower:
        return STUDENT_AGE_OLDER_THAN_24
    if ">20" in s or "20 - 24" in lower or "20-24" in lower or "21-24" in lower:
        return STUDENT_AGE_20_24
    return None


def _learner_age_raw_to_bucket(val: Any) -> Optional[str]:
    """
    Map a single raw learner_age cell to a PDP bucket, or None if unmappable.

    Accepts ints, floats (e.g. 21.0), numeric strings (including \"21.0\"),
    common phrases, and case variants of the canonical bucket labels.
    Unmapped values become null downstream so validation stays permissive.
    """
    if pd.isna(val):
        return None
    if isinstance(val, (bool, np.bool_)):
        return None
    if isinstance(val, (int, np.integer)):
        return _numeric_age_to_bucket(int(val))
    if isinstance(val, (float, np.floating)):
        if np.isnan(val):
            return None
        return _numeric_age_to_bucket(int(math.floor(float(val))))
    if isinstance(val, Decimal):
        try:
            i = int(val.to_integral_value(rounding=ROUND_FLOOR))
        except (InvalidOperation, ValueError, OverflowError):
            return None
        return _numeric_age_to_bucket(i)
    if not isinstance(val, str):
        val = str(val).strip()
    else:
        val = val.strip()
    if not val or val.lower() == "nan":
        return None
    collapsed = " ".join(val.split())
    for label in LEARNER_AGE_BUCKETS:
        if collapsed.casefold() == label.casefold():
            return label
    phrase = _learner_age_phrase_to_bucket(collapsed)
    if phrase is not None:
        return phrase
    try:
        x = float(collapsed)
        if not math.isfinite(x):
            return None
        return _numeric_age_to_bucket(int(math.floor(x)))
    except (ValueError, TypeError, OverflowError):
        return None


def student_age_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map learner_age to PDP-style buckets (20 AND YOUNGER, >20 - 24, OLDER THAN 24).

    Optional field: values that cannot be interpreted are set to pd.NA so rows still
    validate; use buckets where possible for bias analysis.

    Args:
        series: Raw age values (numeric 13-100, floats, numeric strings, phrases).

    Returns:
        String series with bucket labels, or pd.NA where unmapped.
    """
    return series.apply(_learner_age_raw_to_bucket).astype(pd.StringDtype())


# ---------------------------------------------------------------------------
# Pell
# ---------------------------------------------------------------------------

PELL_CATEGORIES = ["Y", "N"]


def _pell_to_pdp_val(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s in ("Y", "YES"):
        return "Y"
    if s in ("N", "NO"):
        return "N"
    return None


def pell_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map pell status to PDP Y/N.

    Args:
        series: Raw values (e.g. "Yes", "No", "Y", "N").

    Returns:
        String series with "Y" or "N", or pd.NA where unmapped.
    """
    return series.astype(str).apply(_pell_to_pdp_val).astype(pd.StringDtype())


# ---------------------------------------------------------------------------
# Credential/degree (kept for other consumers; not used in raw schema validation)
# ---------------------------------------------------------------------------

CREDENTIAL_CANONICAL_BACHELORS = "Bachelor's"
CREDENTIAL_CANONICAL_ASSOCIATES = "Associate's"
CREDENTIAL_CANONICAL_CERTIFICATE = "Certificate"


def _credential_degree_to_canonical(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return None
    lower = s.strip().lower()
    if "bachelor" in lower or lower in ("ba", "bs") or "ba " in lower or "bs " in lower:
        return CREDENTIAL_CANONICAL_BACHELORS
    if (
        "associate" in lower
        or lower in ("aa", "as", "aas")
        or "aa " in lower
        or "as " in lower
        or "aas" in lower
    ):
        return CREDENTIAL_CANONICAL_ASSOCIATES
    if "certificate" in lower or "certification" in lower:
        return CREDENTIAL_CANONICAL_CERTIFICATE
    return None


def credential_degree_series_to_canonical(series: pd.Series) -> pd.Series:
    """
    Map credential/degree free text to canonical PDP-style values.

    Args:
        series: Raw labels (e.g. "Bachelor's Degree", "BA", "Associate's").

    Returns:
        String series with "Bachelor's", "Associate's", or "Certificate"; pd.NA where unmapped.
    """
    return (
        series.astype(str)
        .apply(_credential_degree_to_canonical)
        .astype(pd.StringDtype())
    )


# ---------------------------------------------------------------------------
# Grade
# ---------------------------------------------------------------------------


def grade_series_normalized(series: pd.Series) -> pd.Series:
    """
    Normalize grade for EDA: strip whitespace and uppercase.

    Args:
        series: Raw grade values.

    Returns:
        String series with stripped, uppercased grades.
    """
    return series.astype(str).str.strip().str.upper().astype(pd.StringDtype())


# ---------------------------------------------------------------------------
# Student schema transforms (entry_term, learner_age, pell_recipient_year1)
# ---------------------------------------------------------------------------


def _apply_student_schema_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-validation transforms for RawEdviseStudentDataSchema.

    Normalizes only the fields that require it for Pandera coercion:
    - entry_term: mapped to FALL/WINTER/SPRING/SUMMER for categorical coercion
    - learner_age: bucketed into PDP-style age ranges when parseable; else null
    - pell_recipient_year1: normalized to Y/N

    Does not touch enrollment_type, intended_program_type, or
    conferred_credential_type — those are free-form strings in the raw schema.
    """
    df = df.copy()
    if "entry_term" in df.columns:
        df["entry_term"] = term_series_to_pdp(df["entry_term"])
    if "learner_age" in df.columns:
        df["learner_age"] = student_age_series_to_pdp(df["learner_age"])
    if "pell_recipient_year1" in df.columns:
        df["pell_recipient_year1"] = pell_series_to_pdp(df["pell_recipient_year1"])
    return df


# ---------------------------------------------------------------------------
# Course schema transforms (academic_term, term_pell_recipient)
# ---------------------------------------------------------------------------


def _apply_course_schema_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-validation transforms for RawEdviseCourseDataSchema.

    Normalizes only the fields that require it for Pandera coercion:
    - academic_term: mapped to FALL/WINTER/SPRING/SUMMER for categorical coercion
    - term_pell_recipient: normalized to Y/N

    Does not touch term_degree or grade — those are free-form/custom-checked
    in the raw schema.
    """
    df = df.copy()
    if "academic_term" in df.columns:
        df["academic_term"] = term_series_to_pdp(df["academic_term"])
    if "term_pell_recipient" in df.columns:
        df["term_pell_recipient"] = pell_series_to_pdp(df["term_pell_recipient"])
    return df
