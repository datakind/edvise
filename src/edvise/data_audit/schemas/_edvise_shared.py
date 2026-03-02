# ruff: noqa: F821
# mypy: ignore-errors
"""Shared regex patterns, Field definitions, and PDP-compat transforms for Edvise raw schemas."""

import re
from typing import Optional

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

# Credential/degree type (student + course optional fields)
CREDENTIAL_DEGREE_PATTERN = re.compile(
    r"(?i).*(bachelor|ba|bs|associate|aa|as|aas|certificate|certification).*"
)

StudentIdField = pda.Field(nullable=False, str_length={"min_value": 1})

# ---------------------------------------------------------------------------
# PDP-compat transformation helpers (for EDA / pipeline compatibility)
# ---------------------------------------------------------------------------

# Term: PDP uses categorical FALL, WINTER, SPRING, SUMMER


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


# PDP term categories (ordered: FALL, WINTER, SPRING, SUMMER)
TERM_CATEGORIES = ["FALL", "WINTER", "SPRING", "SUMMER"]


def term_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map term strings to PDP categories: FALL, WINTER, SPRING, SUMMER.

    Args:
        series: Raw term labels (e.g. "Fall", "SP", "Fall 2023").

    Returns:
        String series with values in TERM_CATEGORIES, or pd.NA where unmapped.
    """
    return series.astype(str).apply(_term_to_pdp_val).astype("string")


# Enrollment: PDP uses FIRST-TIME, RE-ADMIT, TRANSFER-IN


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


# PDP enrollment categories
ENROLLMENT_CATEGORIES = ["FIRST-TIME", "RE-ADMIT", "TRANSFER-IN"]


def enrollment_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map enrollment_type strings to PDP categories: FIRST-TIME, RE-ADMIT, TRANSFER-IN.

    Args:
        series: Raw enrollment labels (e.g. "First-time student", "Transfer").

    Returns:
        String series with values in ENROLLMENT_CATEGORIES, or pd.NA where unmapped.
    """
    return series.astype(str).apply(_enrollment_to_pdp_val).astype("string")


# Student age: PDP-style buckets (synth uses these exact strings)
STUDENT_AGE_20_AND_YOUNGER = "20 AND YOUNGER"
STUDENT_AGE_20_24 = ">20 - 24"
STUDENT_AGE_OLDER_THAN_24 = "OLDER THAN 24"


def _student_age_to_pdp_val(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None
    # Already one of the three phrases (case-insensitive)
    lower = s.lower()
    if "20" in lower and ("younger" in lower or "and" in lower):
        return STUDENT_AGE_20_AND_YOUNGER
    if "older" in lower and "24" in lower:
        return STUDENT_AGE_OLDER_THAN_24
    if ">20" in s or "20 - 24" in lower or "20-24" in lower:
        return STUDENT_AGE_20_24
    # Numeric 13–100: map to bucket
    try:
        n = int(s)
        if 13 <= n <= 20:
            return STUDENT_AGE_20_AND_YOUNGER
        if 21 <= n <= 24:
            return STUDENT_AGE_20_24
        if 25 <= n <= 100:
            return STUDENT_AGE_OLDER_THAN_24
    except (ValueError, TypeError):
        pass
    return None


def student_age_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map student_age to PDP-style buckets (20 AND YOUNGER, >20 - 24, OLDER THAN 24).

    Args:
        series: Raw age values (numeric 13-100 or phrase strings).

    Returns:
        String series with bucket labels, or pd.NA where unmapped.
    """
    return series.astype(str).apply(_student_age_to_pdp_val).astype("string")


# Pell: PDP uses Y, N


def _pell_to_pdp_val(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s in ("Y", "YES"):
        return "Y"
    if s in ("N", "NO"):
        return "N"
    return None


# PDP pell categories
PELL_CATEGORIES = ["Y", "N"]


def pell_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map pell status to PDP Y/N.

    Args:
        series: Raw values (e.g. "Yes", "No", "Y", "N").

    Returns:
        String series with "Y" or "N", or pd.NA where unmapped.
    """
    return series.astype(str).apply(_pell_to_pdp_val).astype("string")


# Credential/degree: canonical values for PDP compatibility (Bachelor's, Associate's, Certificate)
CREDENTIAL_CANONICAL_BACHELORS = "Bachelor's"
CREDENTIAL_CANONICAL_ASSOCIATES = "Associate's"
CREDENTIAL_CANONICAL_CERTIFICATE = "Certificate"


def _credential_degree_to_canonical(s: str) -> Optional[str]:
    if pd.isna(s) or not isinstance(s, str) or not s.strip():
        return None
    lower = s.strip().lower()
    # Bachelor's: full word or BA/BS (with or without trailing space to avoid false positives like "base")
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
    return series.astype(str).apply(_credential_degree_to_canonical).astype("string")


# Grade: EDA uses strip/upper for consistency checks


def grade_series_normalized(series: pd.Series) -> pd.Series:
    """
    Normalize grade for EDA: strip whitespace and uppercase.

    Args:
        series: Raw grade values.

    Returns:
        String series with stripped, uppercased grades.
    """
    return series.astype(str).str.strip().str.upper().astype("string")


def _apply_edvise_pdp_transforms_student(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal: PDP-compat transforms for cohort data (raw_* copies + normalized values).
    Used by RawEdviseStudentDataSchema.validate() so coercion sees FALL, FIRST-TIME, etc.
    Accepts "cohort_year" as alias for "cohort".
    """
    df = df.copy()
    # Schema expects "cohort"; allow uploads that use "cohort_year" (e.g. from API/JSON alias).
    if "cohort" not in df.columns and "cohort_year" in df.columns:
        df["cohort"] = df["cohort_year"].astype("string")
    if "enrollment_type" in df.columns:
        df["raw_enrollment_type"] = df["enrollment_type"].astype("string")
        df["enrollment_type"] = enrollment_series_to_pdp(df["enrollment_type"])
    if "student_age" in df.columns:
        df["raw_student_age"] = df["student_age"].astype("string")
        df["student_age"] = student_age_series_to_pdp(df["student_age"])
    if "credential_type_sought_year_1" in df.columns:
        df["raw_credential_type_sought_year_1"] = df[
            "credential_type_sought_year_1"
        ].astype("string")
        df["credential_type_sought_year_1"] = credential_degree_series_to_canonical(
            df["credential_type_sought_year_1"]
        )
    if "degree_grad" in df.columns:
        df["raw_degree_grad"] = df["degree_grad"].astype("string")
        df["degree_grad"] = credential_degree_series_to_canonical(df["degree_grad"])
    if "cohort_term" in df.columns:
        df["cohort_term"] = term_series_to_pdp(df["cohort_term"])
    if "pell_status_first_year" in df.columns:
        df["pell_status_first_year"] = pell_series_to_pdp(df["pell_status_first_year"])
    return df


def _apply_edvise_pdp_transforms_course(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal: PDP-compat transforms for course data. Used by RawEdviseCourseDataSchema.validate().
    """
    df = df.copy()
    if "academic_term" in df.columns:
        df["academic_term"] = term_series_to_pdp(df["academic_term"])
    if "term_degree" in df.columns:
        df["raw_term_degree"] = df["term_degree"].astype("string")
        df["term_degree"] = credential_degree_series_to_canonical(df["term_degree"])
    if "grade" in df.columns:
        df["raw_grade"] = df["grade"].astype("string")
        df["grade"] = grade_series_normalized(df["grade"])
    return df
