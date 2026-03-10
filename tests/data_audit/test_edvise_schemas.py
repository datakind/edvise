"""Unit tests for Edvise raw schemas (RawEdviseStudentDataSchema, RawEdviseCourseDataSchema)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaError, SchemaErrors

from edvise.data_audit.schemas import (
    RawEdviseCourseDataSchema,
    RawEdviseStudentDataSchema,
)

# All column names from the schemas (for tests that include optional columns).
STUDENT_COLUMNS = list(RawEdviseStudentDataSchema.to_schema().columns.keys())
COURSE_COLUMNS = list(RawEdviseCourseDataSchema.to_schema().columns.keys())

# Required-only columns (optional columns may be missing from the DataFrame).
STUDENT_REQUIRED_COLUMNS = [
    "student_id",
    "enrollment_type",
    "credential_type_sought_year_1",
    "program_of_study_term_1",
    "cohort",
    "cohort_term",
]
COURSE_REQUIRED_COLUMNS = [
    "student_id",
    "academic_year",
    "academic_term",
    "course_prefix",
    "course_number",
    "course_name",
    "grade",
    "course_credits_attempted",
    "course_credits_earned",
    "pass_fail_flag",
]


def _minimal_valid_student_row() -> dict[str, Any]:
    """One row of valid student data; required fields set, optionals as None/NaN."""
    return {
        "student_id": "s1",
        "enrollment_type": "First-time student",
        "credential_type_sought_year_1": "Bachelor's Degree",
        "program_of_study_term_1": "Biology",
        "cohort": "2023-24",
        "cohort_term": "Fall 2023",
        "first_enrollment_date": pd.NaT,
        "student_age": None,
        "race": None,
        "ethnicity": None,
        "gender": None,
        "first_gen": None,
        "pell_status_first_year": None,
        "incarcerated_status": None,
        "military_status": None,
        "employment_status": None,
        "disability_status": None,
        "first_bachelors_grad_date": pd.NaT,
        "first_associates_grad_date": pd.NaT,
        "degree_grad": None,
        "major_grad": None,
        "certificate1_date": pd.NaT,
        "certificate2_date": pd.NaT,
        "certificate3_date": pd.NaT,
        "credits_earned_ap": np.nan,
        "credits_earned_dual_enrollment": np.nan,
    }


def _minimal_valid_course_row() -> dict[str, Any]:
    """One row of valid course data; required fields set, optionals as None/NaN."""
    return {
        "student_id": "s1",
        "academic_year": "2024-25",
        "academic_term": "Fall",
        "course_prefix": "MATH",
        "course_number": "101",
        "course_name": "Calculus I",
        "grade": "B",
        "course_credits_attempted": 3.0,
        "course_credits_earned": 3.0,
        "pass_fail_flag": "Pass",
        "department": None,
        "course_classification": None,
        "course_type": None,
        "course_begin_date": pd.NaT,
        "course_end_date": pd.NaT,
        "delivery_method": None,
        "core_course": None,
        "prerequisite_course_flag": None,
        "course_instructor_employment_status": None,
        "gateway_or_development_flag": None,
        "course_section_size": np.nan,
        "term_enrollment_intensity": None,
        "term_degree": None,
        "term_major": None,
        "intent_to_transfer_flag": None,
        "term_pell_recipient": None,
    }


def test_raw_edvise_student_schema_empty_dataframe_passes() -> None:
    """Empty DataFrame with all schema columns passes (no uniqueness or cardinality violations)."""
    df = pd.DataFrame(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 0


def test_raw_edvise_student_schema_extra_columns_allowed() -> None:
    """Schema has strict=False; DataFrame with extra columns still validates."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    df["extra_notes"] = "optional column"
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 1
    assert validated_df["student_id"].iloc[0] == "s1"


def test_raw_edvise_student_schema_valid_minimal() -> None:
    """Valid minimal student row (required only, optionals null) passes."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row])
    # Ensure all schema columns exist (order/columns match)
    df = df.reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    assert validated_df["student_id"].iloc[0] == "s1"


def test_raw_edvise_student_schema_validate_lazy_false_raises_first_error() -> None:
    """Validate with lazy=False raises on first error (no collection of all errors)."""
    row = _minimal_valid_student_row()
    row["student_id"] = ""  # invalid
    row["enrollment_type"] = "Unknown"  # also invalid
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=False)


def test_raw_edvise_student_schema_required_columns_only_passes() -> None:
    """DataFrame with only required columns (no optional columns) passes."""
    row = {
        "student_id": "s1",
        "enrollment_type": "First-time student",
        "credential_type_sought_year_1": "Bachelor's Degree",
        "program_of_study_term_1": "Biology",
        "cohort": "2023-24",
        "cohort_term": "Fall 2023",
    }
    df = pd.DataFrame([row], columns=STUDENT_REQUIRED_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    for col in STUDENT_REQUIRED_COLUMNS:
        assert col in validated_df.columns
    # PDP-compat transforms add raw_* columns and normalize
    assert validated_df["enrollment_type"].iloc[0] == "FIRST-TIME"
    assert "raw_enrollment_type" in validated_df.columns
    assert "raw_credential_type_sought_year_1" in validated_df.columns
    assert validated_df["raw_enrollment_type"].iloc[0] == "First-time student"


def test_raw_edvise_student_schema_valid_with_optionals() -> None:
    """Valid student row with optional fields filled passes."""
    row = _minimal_valid_student_row()
    row["cohort"] = "2023-24"
    row["cohort_term"] = "Fall 2023"
    row["student_age"] = "22"
    row["pell_status_first_year"] = "Y"
    row["credits_earned_ap"] = 6.0
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1


def test_raw_edvise_student_schema_empty_student_id_fails() -> None:
    """Empty student_id fails validation."""
    row = _minimal_valid_student_row()
    row["student_id"] = ""
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_bad_enrollment_type_fails() -> None:
    """Invalid enrollment_type fails validation."""
    row = _minimal_valid_student_row()
    row["enrollment_type"] = "Unknown"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_bad_cohort_fails() -> None:
    """Invalid cohort format fails when column is present (must be YYYY-YY)."""
    row = _minimal_valid_student_row()
    row["cohort"] = "2023"  # must be YYYY-YY
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_valid_cohort() -> None:
    """Valid cohort YYYY-YY passes."""
    row = _minimal_valid_student_row()
    row["cohort"] = "2024-25"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["cohort"].iloc[0] == "2024-25"


def test_raw_edvise_student_schema_cohort_year_alias_passes() -> None:
    """DataFrame with cohort_year but no cohort column passes; transform creates cohort from cohort_year."""
    row = _minimal_valid_student_row()
    # Use cohort_year as the column name (e.g. from API/JSON alias); no "cohort" column.
    row["cohort_year"] = "2024-25"
    if "cohort" in row:
        del row["cohort"]
    df = pd.DataFrame([row]).reindex(
        columns=[c for c in STUDENT_COLUMNS if c != "cohort"] + ["cohort_year"]
    )
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert "cohort" in validated_df.columns
    assert validated_df["cohort"].iloc[0] == "2024-25"


def test_raw_edvise_student_schema_missing_required_column_fails() -> None:
    """DataFrame missing a required column fails validation."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    df = df.drop(columns=["enrollment_type"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_missing_cohort_fails() -> None:
    """DataFrame missing required cohort column (and no cohort_year alias) fails validation."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    df = df.drop(columns=["cohort"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_enrollment_type_transfer_and_readmit() -> None:
    """Enrollment_type Transfer and Re-admit map to TRANSFER-IN and RE-ADMIT."""
    for raw_val, expected in [("Transfer", "TRANSFER-IN"), ("Re-admit", "RE-ADMIT")]:
        row = _minimal_valid_student_row()
        row["enrollment_type"] = raw_val
        df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
        validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
        assert validated_df["enrollment_type"].iloc[0] == expected


def test_raw_edvise_student_schema_credential_type_unmappable_maps_to_null_passes() -> (
    None
):
    """credential_type_sought_year_1 that does not map to canonical becomes null; column is nullable so validation passes."""
    row = _minimal_valid_student_row()
    row["credential_type_sought_year_1"] = "Unknown Degree"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["credential_type_sought_year_1"].iloc[0])
    assert validated_df["raw_credential_type_sought_year_1"].iloc[0] == "Unknown Degree"


def test_raw_edvise_student_schema_validated_output_categorical_dtypes() -> None:
    """Validated student DataFrame has categorical dtypes for cohort_term and enrollment_type (schema coercion)."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert hasattr(validated_df["cohort_term"].dtype, "categories")
    assert hasattr(validated_df["enrollment_type"].dtype, "categories")
    assert list(validated_df["cohort_term"].dtype.categories) == [
        "FALL",
        "WINTER",
        "SPRING",
        "SUMMER",
    ]


def test_raw_edvise_student_schema_valid_cohort_term_passes() -> None:
    """Valid cohort_term (e.g. Fall 2023, SP) is normalized to PDP (FALL, WINTER, SPRING, SUMMER)."""
    row = _minimal_valid_student_row()
    row["cohort_term"] = "Spring 2024"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["cohort_term"].iloc[0] == "SPRING"


def test_raw_edvise_student_schema_bad_cohort_term_fails() -> None:
    """Invalid cohort_term is mapped to null by transform; validation fails (cohort_term is required)."""
    row = _minimal_valid_student_row()
    row["cohort_term"] = "InvalidTerm"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_bad_student_age_maps_to_null() -> None:
    """Student_age outside 13-100 / phrase range is mapped to null by PDP-compat transform (optional column)."""
    row = _minimal_valid_student_row()
    row["student_age"] = "12"  # below 13
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["student_age"].iloc[0])


def test_raw_edvise_student_schema_valid_student_age_passes() -> None:
    """Valid student_age (numeric or phrase) is normalized to PDP buckets (e.g. 20 AND YOUNGER)."""
    row = _minimal_valid_student_row()
    row["student_age"] = "20 and younger"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["student_age"].iloc[0] == "20 AND YOUNGER"


def test_raw_edvise_student_schema_bad_pell_maps_to_null() -> None:
    """Invalid pell_status_first_year (e.g. Maybe) is mapped to null by PDP-compat transform (optional column)."""
    row = _minimal_valid_student_row()
    row["pell_status_first_year"] = "Maybe"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["pell_status_first_year"].iloc[0])


def test_raw_edvise_student_schema_bad_degree_grad_maps_to_null() -> None:
    """degree_grad that doesn't match bachelor/associate/certificate maps to null (optional column)."""
    row = _minimal_valid_student_row()
    row["degree_grad"] = "Unknown"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["degree_grad"].iloc[0])
    assert validated_df["raw_degree_grad"].iloc[0] == "Unknown"


def test_raw_edvise_student_schema_unique_student_id() -> None:
    """Duplicate student_id fails uniqueness check."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row, row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_gender_max_5_values() -> None:
    """gender allows at most 5 distinct values."""
    rows = [
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
    ]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["gender"] = ["A", "B", "C", "D", "E", "F"][i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_gender_5_values_passes() -> None:
    """gender with exactly 5 distinct values passes."""
    rows = [_minimal_valid_student_row() for _ in range(5)]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["gender"] = ["A", "B", "C", "D", "E"][i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 5


def test_raw_edvise_student_schema_first_gen_max_3_values() -> None:
    """first_gen allows at most 3 distinct values."""
    rows = [
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
        _minimal_valid_student_row(),
    ]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["first_gen"] = ["Y", "N", "Unknown", "Other"][i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_first_gen_3_values_passes() -> None:
    """first_gen with exactly 3 distinct values passes."""
    rows = [_minimal_valid_student_row() for _ in range(3)]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["first_gen"] = ["Y", "N", "Unknown"][i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 3


def test_raw_edvise_student_schema_credential_type_max_5_values() -> None:
    """After transform credential_type_sought_year_1 has at most 3 canonical values; 6 raw values map to 2–3, so passes."""
    valid_creds = [
        "Bachelor's Degree",
        "Associate's Degree",
        "Certificate",
        "AA",
        "AS",
        "AAS",
    ]
    rows = [_minimal_valid_student_row() for _ in range(6)]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["credential_type_sought_year_1"] = valid_creds[i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 6
    # Canonical column has at most 3 distinct (Bachelor's, Associate's, Certificate)
    assert validated_df["credential_type_sought_year_1"].nunique() <= 3


def test_raw_edvise_student_schema_credential_type_5_values_passes() -> None:
    """credential_type_sought_year_1 with exactly 5 distinct values passes."""
    valid_creds = ["Bachelor's Degree", "Associate's Degree", "Certificate", "AA", "AS"]
    rows = [_minimal_valid_student_row() for _ in range(5)]
    for i, r in enumerate(rows):
        r["student_id"] = f"s{i}"
        r["credential_type_sought_year_1"] = valid_creds[i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 5


def test_raw_edvise_course_schema_empty_dataframe_passes() -> None:
    """Empty DataFrame with all schema columns passes."""
    df = pd.DataFrame(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 0


def test_raw_edvise_course_schema_extra_columns_allowed() -> None:
    """Schema has strict=False; DataFrame with extra columns still validates."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    df["extra_notes"] = "optional column"
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 1
    assert validated_df["student_id"].iloc[0] == "s1"


def test_raw_edvise_course_schema_missing_required_column_fails() -> None:
    """DataFrame missing a required column fails validation."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    df = df.drop(columns=["academic_year"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_academic_term_fails() -> None:
    """Invalid academic_term format fails."""
    row = _minimal_valid_course_row()
    row["academic_term"] = "InvalidTerm"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_valid_academic_term_passes() -> None:
    """Valid academic_term (e.g. Winter, SP 2024) is normalized to PDP (FALL, WINTER, SPRING, SUMMER)."""
    row = _minimal_valid_course_row()
    row["academic_term"] = "Winter"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert validated_df["academic_term"].iloc[0] == "WINTER"


def test_raw_edvise_course_schema_academic_term_abbreviations() -> None:
    """Academic_term abbreviations FA, WI, SP, SU map to FALL, WINTER, SPRING, SUMMER."""
    for raw_val, expected in [
        ("FA", "FALL"),
        ("WI", "WINTER"),
        ("SP", "SPRING"),
        ("SU", "SUMMER"),
    ]:
        row = _minimal_valid_course_row()
        row["academic_term"] = raw_val
        df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
        validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
        assert validated_df["academic_term"].iloc[0] == expected


def test_raw_edvise_course_schema_grade_normalization() -> None:
    """Grade is normalized (strip, uppercase); original preserved in raw_grade."""
    row = _minimal_valid_course_row()
    row["grade"] = "  b  "
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert validated_df["grade"].iloc[0] == "B"
    assert validated_df["raw_grade"].iloc[0] == "  b  "


def test_raw_edvise_course_schema_validated_output_academic_term_categorical() -> None:
    """Validated course DataFrame has categorical dtype for academic_term (schema coercion)."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert hasattr(validated_df["academic_term"].dtype, "categories")
    assert list(validated_df["academic_term"].dtype.categories) == [
        "FALL",
        "WINTER",
        "SPRING",
        "SUMMER",
    ]


def test_raw_edvise_course_schema_bad_term_enrollment_intensity_fails() -> None:
    """Invalid term_enrollment_intensity fails when present (must contain full-time or part-time)."""
    row = _minimal_valid_course_row()
    row["term_enrollment_intensity"] = "Half-time"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_term_pell_recipient_fails() -> None:
    """Invalid term_pell_recipient fails when present (must be Y/Yes/N/No)."""
    row = _minimal_valid_course_row()
    row["term_pell_recipient"] = "Maybe"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_term_degree_maps_to_null() -> None:
    """term_degree that doesn't match bachelor/associate/certificate maps to null; original in raw_term_degree."""
    row = _minimal_valid_course_row()
    row["term_degree"] = "Unknown"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["term_degree"].iloc[0])
    assert validated_df["raw_term_degree"].iloc[0] == "Unknown"


def test_raw_edvise_course_schema_credits_zero_passes() -> None:
    """course_credits_attempted and course_credits_earned can be 0 (schema ge=0 allows zero)."""
    row = _minimal_valid_course_row()
    row["course_credits_attempted"] = 0.0
    row["course_credits_earned"] = 0.0
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 1
    # Validated validated_dfput may reorder/drop columns; ensure validation accepted zeros
    assert validated_df["course_credits_earned"].iloc[0] == 0.0


def test_raw_edvise_course_schema_valid_minimal() -> None:
    """Valid minimal course row passes."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    assert validated_df["student_id"].iloc[0] == "s1"
    assert validated_df["grade"].iloc[0] == "B"


def test_raw_edvise_course_schema_required_columns_only_passes() -> None:
    """DataFrame with only required columns (no optional columns) passes."""
    row = {
        "student_id": "s1",
        "academic_year": "2024-25",
        "academic_term": "Fall",
        "course_prefix": "MATH",
        "course_number": "101",
        "course_name": "Calculus I",
        "grade": "B",
        "course_credits_attempted": 3.0,
        "course_credits_earned": 3.0,
        "pass_fail_flag": "Pass",
    }
    df = pd.DataFrame([row], columns=COURSE_REQUIRED_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    for col in COURSE_REQUIRED_COLUMNS:
        assert col in validated_df.columns
    # PDP-compat: academic_term normalized to FALL; grade normalized, original in raw_grade
    assert validated_df["academic_term"].iloc[0] == "FALL"
    assert "raw_grade" in validated_df.columns
    assert validated_df["grade"].iloc[0] == "B"
    assert validated_df["raw_grade"].iloc[0] == "B"


def test_raw_edvise_course_schema_valid_with_optionals() -> None:
    """Valid course row with optional fields passes."""
    row = _minimal_valid_course_row()
    row["term_enrollment_intensity"] = "Full-time"
    row["term_pell_recipient"] = "Y"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1


def test_raw_edvise_course_schema_bad_grade_fails() -> None:
    """Invalid grade value fails validation."""
    row = _minimal_valid_course_row()
    row["grade"] = "X"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_academic_year_fails() -> None:
    """Invalid academic_year format fails."""
    row = _minimal_valid_course_row()
    row["academic_year"] = "2024"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_pass_fail_flag_fails() -> None:
    """Invalid pass_fail_flag fails validation."""
    row = _minimal_valid_course_row()
    row["pass_fail_flag"] = "Yes"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_valid_grades_and_pass_fail() -> None:
    """Various allowed grades and pass/fail values pass."""
    for grade in ["A", "B-", "4", "P", "W"]:
        for pf in ["P", "F", "Pass", "Fail"]:
            row = _minimal_valid_course_row()
            row["grade"] = grade
            row["pass_fail_flag"] = pf
            df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
            validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
            assert validated_df["grade"].iloc[0] == grade
            assert validated_df["pass_fail_flag"].iloc[0] == pf


def test_raw_edvise_course_schema_multiple_rows() -> None:
    """Multiple valid course rows pass."""
    rows = [
        _minimal_valid_course_row(),
        _minimal_valid_course_row(),
    ]
    rows[1]["student_id"] = "s2"
    rows[1]["course_prefix"] = "ENGL"
    rows[1]["course_number"] = "101"
    df = pd.DataFrame(rows).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 2


def test_raw_edvise_course_schema_duplicate_composite_key_fails() -> None:
    """Duplicate (student_id, academic_year, academic_term, course_prefix, course_number) fails."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row, row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_section_size_negative_fails() -> None:
    """Negative course_section_size fails validation."""
    row = _minimal_valid_course_row()
    row["course_section_size"] = -1.0
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)
