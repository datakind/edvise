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

# Columns that must exist on the DataFrame (Pandera); nullable columns may be all-null.
# See RawEdviseStudentDataSchema: learner_id/entry_year/entry_term are non-null;
# enrollment_type, intended_program_type, declared_major_at_entry must be present but may be null.
STUDENT_REQUIRED_COLUMNS = [
    "learner_id",
    "entry_year",
    "entry_term",
    "enrollment_type",
    "intended_program_type",
    "declared_major_at_entry",
]
# course_title is optional (may be omitted from the DataFrame or null when present).
COURSE_REQUIRED_COLUMNS = [
    "learner_id",
    "academic_year",
    "academic_term",
    "course_prefix",
    "course_number",
    "grade",
    "course_credits_attempted",
    "course_credits_earned",
]


def _minimal_valid_student_row() -> dict[str, Any]:
    """One row of valid student data; required fields set, optionals as None/NaN."""
    return {
        "learner_id": "s1",
        "entry_year": "2023-24",
        "entry_term": "Fall 2023",
        "enrollment_type": "First-time student",
        "intended_program_type": "Bachelor's Degree",
        "declared_major_at_entry": "Biology",
        "matriculation_date": pd.NaT,
        "learner_age": None,
        "race": None,
        "ethnicity": None,
        "gender": None,
        "first_generation_status": None,
        "pell_recipient_year1": None,
        "incarcerated_status": None,
        "military_status": None,
        "employment_status": None,
        "disability_status": None,
        "bachelors_degree_conferral_date": pd.NaT,
        "associates_degree_conferral_date": pd.NaT,
        "conferred_credential_type": None,
        "major_at_completion": None,
        "certificate1_date": pd.NaT,
        "certificate2_date": pd.NaT,
        "certificate3_date": pd.NaT,
        "credits_earned_ap": np.nan,
        "credits_earned_dual_enrollment": np.nan,
    }


def _minimal_valid_course_row() -> dict[str, Any]:
    """One row of valid course data; required fields set, optionals as None/NaN."""
    return {
        "learner_id": "s1",
        "academic_year": "2024-25",
        "academic_term": "Fall",
        "course_prefix": "MATH",
        "course_number": "101",
        "course_title": "Calculus I",
        "course_section_id": "001",
        "grade": "B",
        "course_credits_attempted": 3.0,
        "course_credits_earned": 3.0,
        "department": None,
        "instructional_format": None,
        "academic_level": None,
        "course_begin_date": pd.NaT,
        "course_end_date": pd.NaT,
        "instructional_modality": None,
        "gen_ed_flag": None,
        "prerequisite_flag": None,
        "instructor_appointment_status": None,
        "gateway_or_developmental_flag": None,
        "course_section_size": np.nan,
        "term_degree": None,
        "term_declared_major": None,
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
    assert validated_df["learner_id"].iloc[0] == "s1"


def test_raw_edvise_student_schema_valid_minimal() -> None:
    """Valid minimal student row (required only, optionals null) passes."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row])
    df = df.reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    assert validated_df["learner_id"].iloc[0] == "s1"


def test_raw_edvise_student_schema_validate_lazy_false_raises_first_error() -> None:
    """Validate with lazy=False raises on first error (no collection of all errors)."""
    row = _minimal_valid_student_row()
    row["learner_id"] = ""  # invalid
    row["entry_year"] = "bad"  # also invalid
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=False)


def test_raw_edvise_student_schema_required_columns_only_passes() -> None:
    """DataFrame with schema-required columns and no optional columns passes."""
    row = {
        "learner_id": "s1",
        "entry_year": "2023-24",
        "entry_term": "Fall 2023",
        "enrollment_type": pd.NA,
        "intended_program_type": pd.NA,
        "declared_major_at_entry": pd.NA,
    }
    df = pd.DataFrame([row], columns=STUDENT_REQUIRED_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    for col in STUDENT_REQUIRED_COLUMNS:
        assert col in validated_df.columns
    assert validated_df["entry_term"].iloc[0] == "FALL"


def test_raw_edvise_student_schema_valid_with_optionals() -> None:
    """Valid student row with optional fields filled passes."""
    row = _minimal_valid_student_row()
    row["entry_year"] = "2023-24"
    row["entry_term"] = "Fall 2023"
    row["learner_age"] = "22"
    row["pell_recipient_year1"] = "Y"
    row["credits_earned_ap"] = 6.0
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1


def test_raw_edvise_student_schema_empty_learner_id_fails() -> None:
    """Empty learner_id fails validation."""
    row = _minimal_valid_student_row()
    row["learner_id"] = ""
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_bad_entry_year_fails() -> None:
    """Invalid entry_year format (must be YYYY-YY) fails when column is present."""
    row = _minimal_valid_student_row()
    row["entry_year"] = "2023"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_valid_entry_year() -> None:
    """Valid entry_year YYYY-YY passes."""
    row = _minimal_valid_student_row()
    row["entry_year"] = "2024-25"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["entry_year"].iloc[0] == "2024-25"


def test_raw_edvise_student_schema_missing_required_column_fails() -> None:
    """DataFrame missing a non-optional schema column fails validation."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    df = df.drop(columns=["intended_program_type"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_missing_entry_year_fails() -> None:
    """DataFrame missing required entry_year fails validation."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    df = df.drop(columns=["entry_year"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_enrollment_type_preserved() -> None:
    """enrollment_type is free-form; raw values are not coerced in schema validation."""
    for raw_val in ("Transfer", "Re-admit", "First-time student"):
        row = _minimal_valid_student_row()
        row["enrollment_type"] = raw_val
        df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
        validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
        assert validated_df["enrollment_type"].iloc[0] == raw_val


def test_raw_edvise_student_schema_intended_program_free_text_passes() -> None:
    """intended_program_type accepts arbitrary text (no canonical null-mapping in raw schema)."""
    row = _minimal_valid_student_row()
    row["intended_program_type"] = "Unknown Degree"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["intended_program_type"].iloc[0] == "Unknown Degree"


def test_raw_edvise_student_schema_validated_output_categorical_dtypes() -> None:
    """Validated student DataFrame has categorical dtype for entry_term (schema coercion)."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert hasattr(validated_df["entry_term"].dtype, "categories")
    assert list(validated_df["entry_term"].dtype.categories) == [
        "FALL",
        "WINTER",
        "SPRING",
        "SUMMER",
    ]


def test_raw_edvise_student_schema_valid_entry_term_passes() -> None:
    """Valid entry_term (e.g. Spring 2024) is normalized to PDP (SPRING)."""
    row = _minimal_valid_student_row()
    row["entry_term"] = "Spring 2024"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["entry_term"].iloc[0] == "SPRING"


def test_raw_edvise_student_schema_bad_entry_term_fails() -> None:
    """Invalid entry_term maps to null by transform; non-nullable entry_term fails validation."""
    row = _minimal_valid_student_row()
    row["entry_term"] = "InvalidTerm"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_bad_learner_age_maps_to_null() -> None:
    """learner_age outside mappable range maps to null; optional column stays valid."""
    row = _minimal_valid_student_row()
    row["learner_age"] = "12"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["learner_age"].iloc[0])


def test_raw_edvise_student_schema_valid_learner_age_passes() -> None:
    """Valid learner_age phrase is normalized to PDP buckets (e.g. 20 AND YOUNGER)."""
    row = _minimal_valid_student_row()
    row["learner_age"] = "20 and younger"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["learner_age"].iloc[0] == "20 AND YOUNGER"


def test_raw_edvise_student_schema_bad_pell_maps_to_null() -> None:
    """Invalid pell_recipient_year1 maps to null; optional column stays valid."""
    row = _minimal_valid_student_row()
    row["pell_recipient_year1"] = "Maybe"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["pell_recipient_year1"].iloc[0])


def test_raw_edvise_student_schema_conferred_credential_free_text() -> None:
    """conferred_credential_type is free-form string in the raw schema."""
    row = _minimal_valid_student_row()
    row["conferred_credential_type"] = "Unknown"
    df = pd.DataFrame([row]).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert validated_df["conferred_credential_type"].iloc[0] == "Unknown"


def test_raw_edvise_student_schema_unique_learner_id() -> None:
    """Duplicate learner_id fails uniqueness check."""
    row = _minimal_valid_student_row()
    df = pd.DataFrame([row, row]).reindex(columns=STUDENT_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseStudentDataSchema.validate(df, lazy=True)


def test_raw_edvise_student_schema_gender_many_distinct_values_passes() -> None:
    """gender is unconstrained string; many distinct values still validate."""
    rows = [_minimal_valid_student_row() for _ in range(6)]
    for i, r in enumerate(rows):
        r["learner_id"] = f"s{i}"
        r["gender"] = ["A", "B", "C", "D", "E", "F"][i]
    df = pd.DataFrame(rows).reindex(columns=STUDENT_COLUMNS)
    validated_df = RawEdviseStudentDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 6


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
    assert validated_df["learner_id"].iloc[0] == "s1"


def test_raw_edvise_course_schema_missing_required_column_fails() -> None:
    """DataFrame missing a required column fails validation."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    df = df.drop(columns=["academic_year"])
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_bad_academic_term_fails() -> None:
    """Invalid academic_term fails."""
    row = _minimal_valid_course_row()
    row["academic_term"] = "InvalidTerm"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_valid_academic_term_passes() -> None:
    """Valid academic_term (e.g. Winter) is normalized to PDP (WINTER)."""
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


def test_raw_edvise_course_schema_grade_validation_accepts_trimmed_letter() -> None:
    """Grade check accepts letter grades after strip/upper internally; stored value unchanged."""
    row = _minimal_valid_course_row()
    row["grade"] = "  b  "
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert str(validated_df["grade"].iloc[0]).strip().upper() == "B"


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


def test_raw_edvise_course_schema_bad_term_pell_maps_to_null() -> None:
    """Invalid term_pell_recipient is mapped to null by transform; optional column validates."""
    row = _minimal_valid_course_row()
    row["term_pell_recipient"] = "Maybe"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert pd.isna(validated_df["term_pell_recipient"].iloc[0])


def test_raw_edvise_course_schema_term_degree_free_text() -> None:
    """term_degree is free-form in the raw schema (no raw_* alias column)."""
    row = _minimal_valid_course_row()
    row["term_degree"] = "Unknown"
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert validated_df["term_degree"].iloc[0] == "Unknown"


def test_raw_edvise_course_schema_credits_zero_passes() -> None:
    """course_credits_attempted and course_credits_earned can be 0 (schema ge=0 allows zero)."""
    row = _minimal_valid_course_row()
    row["course_credits_attempted"] = 0.0
    row["course_credits_earned"] = 0.0
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 1
    assert validated_df["course_credits_earned"].iloc[0] == 0.0


def test_raw_edvise_course_schema_valid_minimal() -> None:
    """Valid minimal course row passes."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    assert validated_df["learner_id"].iloc[0] == "s1"
    assert validated_df["grade"].iloc[0] == "B"


def test_raw_edvise_course_schema_required_columns_only_passes() -> None:
    """DataFrame with only required columns (no optional columns, including no course_title) passes."""
    row = {
        "learner_id": "s1",
        "academic_year": "2024-25",
        "academic_term": "Fall",
        "course_prefix": "MATH",
        "course_number": "101",
        "grade": "B",
        "course_credits_attempted": 3.0,
        "course_credits_earned": 3.0,
    }
    df = pd.DataFrame([row], columns=COURSE_REQUIRED_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert isinstance(validated_df, pd.DataFrame)
    assert len(validated_df) == 1
    for col in COURSE_REQUIRED_COLUMNS:
        assert col in validated_df.columns
    assert "course_title" not in validated_df.columns
    assert validated_df["academic_term"].iloc[0] == "FALL"
    assert validated_df["grade"].iloc[0] == "B"


def test_raw_edvise_course_schema_course_title_null_passes() -> None:
    """course_title may be present and null when other fields are valid."""
    row = _minimal_valid_course_row()
    row["course_title"] = pd.NA
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 1
    assert pd.isna(validated_df["course_title"].iloc[0])


def test_raw_edvise_course_schema_valid_with_optionals() -> None:
    """Valid course row with optional fields passes."""
    row = _minimal_valid_course_row()
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


def test_raw_edvise_course_schema_valid_letter_and_numeric_grades() -> None:
    """Various allowed letter and numeric grades pass."""
    for grade in ["A", "B-", "3.5", "P", "W"]:
        row = _minimal_valid_course_row()
        row["grade"] = grade
        df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
        validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
        assert validated_df["grade"].iloc[0] == grade


def test_raw_edvise_course_schema_multiple_rows() -> None:
    """Multiple valid course rows with distinct composite keys pass."""
    rows = [
        _minimal_valid_course_row(),
        _minimal_valid_course_row(),
    ]
    rows[1]["learner_id"] = "s2"
    rows[1]["course_prefix"] = "ENGL"
    rows[1]["course_number"] = "101"
    rows[1]["course_section_id"] = "002"
    df = pd.DataFrame(rows).reindex(columns=COURSE_COLUMNS)
    validated_df = RawEdviseCourseDataSchema.validate(df, lazy=True)
    assert len(validated_df) == 2


def test_raw_edvise_course_schema_duplicate_composite_key_fails() -> None:
    """Duplicate (learner_id, academic_year, academic_term, course_prefix, course_number) fails."""
    row = _minimal_valid_course_row()
    df = pd.DataFrame([row, row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)


def test_raw_edvise_course_schema_course_section_size_negative_fails() -> None:
    """Negative course_section_size fails validation."""
    row = _minimal_valid_course_row()
    row["course_section_size"] = -1.0
    df = pd.DataFrame([row]).reindex(columns=COURSE_COLUMNS)
    with pytest.raises((SchemaError, SchemaErrors)):
        RawEdviseCourseDataSchema.validate(df, lazy=True)
