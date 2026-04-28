"""Unit tests for Edvise shared transform helpers (_edvise_shared)."""

import numpy as np
import pandas as pd

from edvise.data_audit.schemas._edvise_shared import (
    credential_degree_series_to_canonical,
    enrollment_series_to_pdp,
    grade_series_normalized,
    pell_series_to_pdp,
    student_age_series_to_pdp,
    term_series_to_pdp,
)


# ---- term_series_to_pdp ----


def test_term_series_to_pdp_fall_variants() -> None:
    """Term transform maps Fall, FA, Fall 2023 to FALL."""
    series = pd.Series(["Fall", "FA", "Fall 2023", "FALL"])
    result = term_series_to_pdp(series)
    assert result.tolist() == ["FALL", "FALL", "FALL", "FALL"]


def test_term_series_to_pdp_all_seasons_and_abbrevs() -> None:
    """Term transform maps Winter/WI, Spring/SP, Summer/SU/SM to PDP categories."""
    series = pd.Series(["Winter", "WI", "Spring", "SP", "Summer", "SU", "SM"])
    result = term_series_to_pdp(series)
    assert result.tolist() == [
        "WINTER",
        "WINTER",
        "SPRING",
        "SPRING",
        "SUMMER",
        "SUMMER",
        "SUMMER",
    ]


def test_term_series_to_pdp_unmapped_returns_null() -> None:
    """Term transform returns pd.NA for unmapped values."""
    series = pd.Series(["InvalidTerm", "Q1", np.nan])
    result = term_series_to_pdp(series)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


# ---- enrollment_series_to_pdp ----


def test_enrollment_series_to_pdp_first_time_transfer_readmit() -> None:
    """Enrollment transform maps First-time, Transfer, Re-admit to PDP categories."""
    series = pd.Series(
        [
            "First-time student",
            "Transfer",
            "Re-admit",
            "Freshman",
            "RE-ADMIT",
            "Transfer-in",
        ]
    )
    result = enrollment_series_to_pdp(series)
    assert result.tolist() == [
        "FIRST-TIME",
        "TRANSFER-IN",
        "RE-ADMIT",
        "FIRST-TIME",
        "RE-ADMIT",
        "TRANSFER-IN",
    ]


def test_enrollment_series_to_pdp_unmapped_returns_null() -> None:
    """Enrollment transform returns pd.NA for unmapped values."""
    series = pd.Series(["Unknown", "Other"])
    result = enrollment_series_to_pdp(series)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


# ---- student_age_series_to_pdp ----


def test_student_age_series_to_pdp_numeric_buckets() -> None:
    """Student age transform maps numeric 13-20, 21-24, 25-100 to buckets."""
    series = pd.Series([13, 20, 21, 24, 25, 100])
    result = student_age_series_to_pdp(series)
    assert result.iloc[0] == "20 AND YOUNGER"
    assert result.iloc[1] == "20 AND YOUNGER"
    assert result.iloc[2] == ">20 - 24"
    assert result.iloc[3] == ">20 - 24"
    assert result.iloc[4] == "OLDER THAN 24"
    assert result.iloc[5] == "OLDER THAN 24"


def test_student_age_series_to_pdp_phrase_strings() -> None:
    """Student age transform maps phrase strings to PDP buckets."""
    series = pd.Series(["20 and younger", ">20 - 24", "OLDER THAN 24"])
    result = student_age_series_to_pdp(series)
    assert result.iloc[0] == "20 AND YOUNGER"
    assert result.iloc[1] == ">20 - 24"
    assert result.iloc[2] == "OLDER THAN 24"


def test_student_age_series_to_pdp_out_of_range_returns_null() -> None:
    """Student age outside 13-100 or unknown phrase returns pd.NA."""
    series = pd.Series([12, 101, "Unknown"])
    result = student_age_series_to_pdp(series)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_student_age_series_to_pdp_float_and_numeric_string() -> None:
    """Float ages and numeric strings (e.g. 21.0) bucket like integer ages."""
    series = pd.Series([21.0, "21.0", "24.9"])
    result = student_age_series_to_pdp(series)
    assert result.iloc[0] == ">20 - 24"
    assert result.iloc[1] == ">20 - 24"
    assert result.iloc[2] == ">20 - 24"


def test_student_age_series_to_pdp_canonical_label_whitespace() -> None:
    """Already-canonical bucket labels with odd spacing still normalize."""
    series = pd.Series(["  older than 24  ", "OLDER THAN 24"])
    result = student_age_series_to_pdp(series)
    assert result.iloc[0] == "OLDER THAN 24"
    assert result.iloc[1] == "OLDER THAN 24"


# ---- pell_series_to_pdp ----


def test_pell_series_to_pdp_yes_no_variants() -> None:
    """Pell transform maps Yes/No, Y/N to Y/N."""
    series = pd.Series(["Yes", "No", "Y", "N", "YES", "NO"])
    result = pell_series_to_pdp(series)
    assert result.tolist() == ["Y", "N", "Y", "N", "Y", "N"]


def test_pell_series_to_pdp_unmapped_returns_null() -> None:
    """Pell transform returns pd.NA for unmapped values."""
    series = pd.Series(["Maybe", "Unknown"])
    result = pell_series_to_pdp(series)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


# ---- credential_degree_series_to_canonical ----


def test_credential_degree_series_to_canonical_bachelors_variants() -> None:
    """Credential transform maps Bachelor's, BA, BS to Bachelor's."""
    series = pd.Series(["Bachelor's Degree", "BA", "BS", "bachelor"])
    result = credential_degree_series_to_canonical(series)
    assert result.tolist() == ["Bachelor's", "Bachelor's", "Bachelor's", "Bachelor's"]


def test_credential_degree_series_to_canonical_associate_certificate() -> None:
    """Credential transform maps Associate's, AA, AS, AAS, Certificate to canonical."""
    series = pd.Series(
        ["Associate's", "AA", "AS", "AAS", "Certificate", "certification"]
    )
    result = credential_degree_series_to_canonical(series)
    assert result.iloc[0] == "Associate's"
    assert result.iloc[1] == "Associate's"
    assert result.iloc[2] == "Associate's"
    assert result.iloc[3] == "Associate's"
    assert result.iloc[4] == "Certificate"
    assert result.iloc[5] == "Certificate"


def test_credential_degree_series_to_canonical_unmapped_returns_null() -> None:
    """Credential transform returns pd.NA for unmapped values."""
    series = pd.Series(["Unknown", "Other"])
    result = credential_degree_series_to_canonical(series)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])


# ---- grade_series_normalized ----


def test_grade_series_normalized_strip_uppercase() -> None:
    """Grade transform strips whitespace and uppercases."""
    series = pd.Series(["  a  ", "b", "B-", "  A+  "])
    result = grade_series_normalized(series)
    assert result.tolist() == ["A", "B", "B-", "A+"]


def test_grade_series_normalized_preserves_valid_grade() -> None:
    """Grade transform preserves already-normalized grades."""
    series = pd.Series(["A", "B", "F", "W"])
    result = grade_series_normalized(series)
    assert result.tolist() == ["A", "B", "F", "W"]
