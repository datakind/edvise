#!/usr/bin/env python3
"""
Generate synthetic student data that validates against RawEdviseStudentDataSchema.

Values and definitions align with the DataKind/Edvise student file spec:
- Required: student_id, enrollment_type, credential_type_sought_year_1, program_of_study_term_1, cohort_year, cohort_term
- Optional columns may be null; when present they follow the schema patterns and cardinality
  (e.g. gender max 5 values, first_gen max 3, credential_type_sought_year_1 max 5).
"""

import argparse
import logging
import pathlib
import random
import sys
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from edvise.data_audit.schemas import RawEdviseStudentDataSchema

LOGGER = logging.getLogger(__name__)

# Required columns only (no optional columns in the "required_only" output)
REQUIRED_STUDENT_COLUMNS = [
    "student_id",
    "enrollment_type",
    "credential_type_sought_year_1",
    "program_of_study_term_1",
    "cohort_year",
    "cohort_term",
]

# --- Value pools that satisfy schema and spec (cardinality limits) ---

# enrollment_type: must match ENROLLMENT_TYPE_PATTERN (first-time|freshman|transfer|re-admit|readmit)
# First Time most likely; Transfer and Readmit less
ENROLLMENT_TYPES = [
    "First Time in College",
    "Transfer",
    "Readmit",
]
ENROLLMENT_TYPE_WEIGHTS = [8, 1, 1]  # First Time, Transfer, Readmit

# credential_type_sought_year_1: must match CREDENTIAL_DEGREE_PATTERN; max 5 distinct
# Bachelor's and Associate more likely than Certificate types
CREDENTIAL_TYPES = [
    "Bachelor's Degree",
    "Associate Degree",
    "Certificate",
    "Undergraduate Certificate or Diploma Program",
]
CREDENTIAL_TYPE_WEIGHTS = [3, 3, 1, 1]  # Bachelor's, Associate, Certificate, Undergrad Cert

# program_of_study_term_1: any categorical (e.g. major names)
PROGRAMS_OF_STUDY = [
    "Biology",
    "Chemistry",
    "Liberal Arts",
    "Mathematics",
    "History",
    "Spanish",
    "Computer Science",
    "Psychology",
    "Business",
    "Nursing",
    "Undeclared",
]

# cohort_year: YYYY-YY
def _random_cohort_year(min_yr: int = 2018, max_yr: int = 2025) -> str:
    y = random.randint(min_yr, max_yr)
    return f"{y}-{str(y + 1)[2:]}"


# cohort_term: Fall, Winter, Spring, Summer or FA, WI, SP, SU/SM (TERM_PATTERN)
# Fall and Spring more likely than Winter and Summer
COHORT_TERMS = ["Fall", "Winter", "Spring", "Summer"]
COHORT_TERM_WEIGHTS = [3, 1, 3, 1]  # Fall, Winter, Spring, Summer

# student_age: binned; must match STUDENT_AGE_PATTERN. Weights: 20 and younger 7, 20-24 2, older than 24 1
STUDENT_AGE_VALUES = ["20 and younger", ">20 - 24", "older than 24"]
STUDENT_AGE_WEIGHTS = [7, 2, 1]

# race (optional); weighted distribution
RACE_VALUES = [
    "White",
    "Hispanic",
    "Black or African American",
    "Two or More Races",
    "Asian",
    "Native Hawaiian or Other Pacific Islander",
    "American Indian or Alaska Native",
    "Unknown",
]
RACE_WEIGHTS = [75, 8, 6, 3, 3, 2, 2, 1]

# ethnicity (optional); Hispanic if race is Hispanic, Unknown if race is Unknown; else Not Hispanic 89, Hispanic 8, Unknown 3
ETHNICITY_VALUES = ["Not Hispanic", "Hispanic", "Unknown"]
ETHNICITY_WEIGHTS = [89, 8, 3]

# gender: max 5 distinct values; Female ~54%, Male ~43%, others ~1% each
GENDER_VALUES = ["Female", "Male", "Non-Binary", "Unknown", "Prefer not to say"]
GENDER_WEIGHTS = [54, 43, 1, 1, 1]  # Female, Male, Non-Binary, Unknown, Prefer not to say

# first_gen: No 60%, Yes 38%, Unknown 2%
FIRST_GEN_VALUES = ["No", "Yes", "Unknown"]
FIRST_GEN_WEIGHTS = [60, 38, 2]

# pell_status_first_year: 66% Y, 33% N
PELL_VALUES = ["Y", "N"]
PELL_WEIGHTS = [66, 33]

# incarcerated_status: 98% No, 1% Yes, 1% Unknown
INCARCERATED_VALUES = ["No", "Yes", "Unknown"]
INCARCERATED_WEIGHTS = [98, 1, 1]
# military_status: 95% Never Served, 2% Veteran, 2% Unknown, 1% Active Duty
MILITARY_VALUES = ["Never Served", "Veteran", "Unknown", "Active Duty"]
MILITARY_WEIGHTS = [95, 2, 2, 1]
EMPLOYMENT_VALUES = ["Full-Time", "Part-Time", "Not employed", "Unknown"]
# disability_status: 95% No, 3% Yes, 2% Unknown
DISABILITY_VALUES = ["No", "Yes", "Unknown"]
DISABILITY_WEIGHTS = [95, 3, 2]

# degree_grad: must match CREDENTIAL_DEGREE_PATTERN when present
DEGREE_GRAD_VALUES = [
    "Bachelor's Degree",
    "Associate Degree",
    "Certificate",
]


def _random_date_in_range(
    start_year: int, end_year: int, allow_null: bool = True
) -> pd.Timestamp | None:
    if allow_null and random.random() < 0.3:
        return pd.NaT
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    delta = (end - start).days
    if delta <= 0:
        return pd.Timestamp(start)
    d = start + timedelta(days=random.randint(0, delta))
    return pd.Timestamp(d)


def generate_student_row(
    *,
    student_id: str,
    use_optionals: bool = True,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Generate one synthetic student record conforming to RawEdviseStudentDataSchema."""
    r = rng or random
    row: dict[str, Any] = {
        "student_id": student_id,
        "enrollment_type": r.choices(ENROLLMENT_TYPES, weights=ENROLLMENT_TYPE_WEIGHTS, k=1)[0],
        "credential_type_sought_year_1": r.choices(CREDENTIAL_TYPES, weights=CREDENTIAL_TYPE_WEIGHTS, k=1)[0],
        "program_of_study_term_1": r.choice(PROGRAMS_OF_STUDY),
    }
    if not use_optionals:
        # Required: cohort_year and cohort_term; rest optional (null)
        y = r.randint(2018, 2025)
        row["cohort_year"] = f"{y}-{str(y + 1)[2:]}"
        row["cohort_term"] = r.choices(COHORT_TERMS, weights=COHORT_TERM_WEIGHTS, k=1)[0]
        row["first_enrollment_date"] = pd.NaT
        row["student_age"] = None
        row["race"] = None
        row["ethnicity"] = None
        row["gender"] = None
        row["first_gen"] = None
        row["pell_status_first_year"] = None
        row["incarcerated_status"] = None
        row["military_status"] = None
        row["employment_status"] = None
        row["disability_status"] = None
        row["first_bachelors_grad_date"] = pd.NaT
        row["first_associates_grad_date"] = pd.NaT
        row["degree_grad"] = None
        row["major_grad"] = None
        row["certificate1_date"] = pd.NaT
        row["certificate2_date"] = pd.NaT
        row["certificate3_date"] = pd.NaT
        row["credits_earned_ap"] = np.nan
        row["credits_earned_dual_enrollment"] = np.nan
        return row

    # Optional fields: sometimes null, sometimes a valid value
    def opt(null_prob: float = 0.25):
        return r.random() < null_prob

    # cohort_year and cohort_term required (non-missing)
    row["cohort_year"] = _random_cohort_year()
    row["cohort_term"] = r.choices(COHORT_TERMS, weights=COHORT_TERM_WEIGHTS, k=1)[0]
    row["first_enrollment_date"] = _random_date_in_range(2018, 2025, allow_null=True)
    row["student_age"] = None if opt(0.15) else r.choices(STUDENT_AGE_VALUES, weights=STUDENT_AGE_WEIGHTS, k=1)[0]
    row["race"] = None if opt(0.15) else r.choices(RACE_VALUES, weights=RACE_WEIGHTS, k=1)[0]
    if opt(0.15):
        row["ethnicity"] = None
    elif row["race"] == "Hispanic":
        row["ethnicity"] = "Hispanic"
    elif row["race"] == "Unknown":
        row["ethnicity"] = "Unknown"
    else:
        row["ethnicity"] = r.choices(ETHNICITY_VALUES, weights=ETHNICITY_WEIGHTS, k=1)[0]
    row["gender"] = None if opt(0.1) else r.choices(GENDER_VALUES, weights=GENDER_WEIGHTS, k=1)[0]
    row["first_gen"] = None if opt(0.2) else r.choices(FIRST_GEN_VALUES, weights=FIRST_GEN_WEIGHTS, k=1)[0]
    row["pell_status_first_year"] = None if opt(0.2) else r.choices(PELL_VALUES, weights=PELL_WEIGHTS, k=1)[0]
    row["incarcerated_status"] = None if opt(0.4) else r.choices(INCARCERATED_VALUES, weights=INCARCERATED_WEIGHTS, k=1)[0]
    row["military_status"] = None if opt(0.3) else r.choices(MILITARY_VALUES, weights=MILITARY_WEIGHTS, k=1)[0]
    row["employment_status"] = None if opt(0.2) else r.choice(EMPLOYMENT_VALUES)
    row["disability_status"] = None if opt(0.3) else r.choices(DISABILITY_VALUES, weights=DISABILITY_WEIGHTS, k=1)[0]
    row["first_bachelors_grad_date"] = _random_date_in_range(2020, 2026, allow_null=True)
    row["first_associates_grad_date"] = _random_date_in_range(2020, 2026, allow_null=True)
    row["degree_grad"] = None if opt(0.5) else r.choice(DEGREE_GRAD_VALUES)
    row["major_grad"] = None if opt(0.5) else r.choice(PROGRAMS_OF_STUDY)
    row["certificate1_date"] = _random_date_in_range(2020, 2026, allow_null=True)
    row["certificate2_date"] = _random_date_in_range(2020, 2026, allow_null=True)
    row["certificate3_date"] = _random_date_in_range(2020, 2026, allow_null=True)
    if opt(0.5):
        row["credits_earned_ap"] = np.nan
    else:
        row["credits_earned_ap"] = round(r.uniform(0.0, 24.0), 1)
    if opt(0.5):
        row["credits_earned_dual_enrollment"] = np.nan
    else:
        row["credits_earned_dual_enrollment"] = round(r.uniform(0.0, 30.0), 1)
    return row


def generate_student_dataframe(
    n_rows: int,
    *,
    seed: int | None = None,
    use_optionals: bool = True,
    ensure_cardinality: bool = True,
) -> pd.DataFrame:
    """
    Build a DataFrame of n_rows synthetic students that validates against RawEdviseStudentDataSchema.

    When ensure_cardinality is True, gender, first_gen, and credential_type_sought_year_1
    are drawn from their full pools so that across the dataset we stay within schema
    cardinality limits (5, 3, 5 respectively).
    """
    rng = random.Random(seed)
    # Unique student IDs (schema: unique on student_id)
    used: set[str] = set()
    ids = []
    for _ in range(n_rows):
        while True:
            sid = str(rng.randint(100000, 999999999))
            if sid not in used:
                used.add(sid)
                ids.append(sid)
                break
    rows = [
        generate_student_row(student_id=sid, use_optionals=use_optionals, rng=rng)
        for sid in ids
    ]
    schema = RawEdviseStudentDataSchema.to_schema()
    columns = list(schema.columns.keys())
    df = pd.DataFrame(rows).reindex(columns=columns)
    # Ensure dtypes for optional numerics (NaN allowed)
    for col in ("credits_earned_ap", "credits_earned_dual_enrollment"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if ensure_cardinality:
        # Restrict so dataset has at most 5 genders, 3 first_gen, 5 credential types
        if "gender" in df.columns:
            mask = df["gender"].notna()
            df.loc[mask, "gender"] = rng.choices(
                GENDER_VALUES, weights=GENDER_WEIGHTS, k=mask.sum()
            )
        if "first_gen" in df.columns:
            mask = df["first_gen"].notna()
            df.loc[mask, "first_gen"] = rng.choices(
                FIRST_GEN_VALUES, weights=FIRST_GEN_WEIGHTS, k=mask.sum()
            )
        df["credential_type_sought_year_1"] = rng.choices(
            CREDENTIAL_TYPES, weights=CREDENTIAL_TYPE_WEIGHTS, k=len(df)
        )
    validated = RawEdviseStudentDataSchema.validate(df, lazy=True)
    return validated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic student data matching RawEdviseStudentDataSchema and DataKind spec.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num-rows",
        type=int,
        default=100,
        help="Number of student rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--no-optionals",
        action="store_true",
        help="When not using --output: generate only the required-only dataset. Ignored when --output is set.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Output path for CSV (or .parquet). When set, writes TWO files: one with all variables (including optionals) and one with only the required columns (optional columns do not exist in the file), so you can test with or without optional columns.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip schema validation (not recommended).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.output is not None:
        # Write two datasets: full (all variables) and required-only (required columns only, no optionals)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stem, suffix = args.output.stem, args.output.suffix
        if not stem:
            stem = "edvise_students"
        if not suffix:
            suffix = ".csv"
        out_full = args.output.parent / f"{stem}_full{suffix}"
        out_required_only = args.output.parent / f"{stem}_required_only{suffix}"

        df_full = generate_student_dataframe(
            args.num_rows,
            seed=args.seed,
            use_optionals=True,
            ensure_cardinality=True,
        )
        df_with_required_filled = generate_student_dataframe(
            args.num_rows,
            seed=args.seed,
            use_optionals=False,
            ensure_cardinality=True,
        )
        if not args.no_validate:
            RawEdviseStudentDataSchema.validate(df_full, lazy=True)
            RawEdviseStudentDataSchema.validate(df_with_required_filled, lazy=True)
        LOGGER.info("Generated %s rows; schema validation passed for full dataset.", args.num_rows)

        # Required-only file: only the required columns (optional columns do not exist)
        df_required_only = df_with_required_filled[REQUIRED_STUDENT_COLUMNS].copy()

        if out_full.suffix.lower() == ".parquet":
            df_full.to_parquet(out_full, index=False)
            df_required_only.to_parquet(out_required_only, index=False)
        else:
            df_full.to_csv(out_full, index=False)
            df_required_only.to_csv(out_required_only, index=False)
        LOGGER.info("Wrote full dataset (all variables): %s", out_full)
        LOGGER.info("Wrote required-only dataset (only %s columns, no optionals): %s", len(REQUIRED_STUDENT_COLUMNS), out_required_only)
        return 0

    # Single in-memory dataset when no --output
    df = generate_student_dataframe(
        args.num_rows,
        seed=args.seed,
        use_optionals=not args.no_optionals,
        ensure_cardinality=True,
    )
    if not args.no_validate:
        RawEdviseStudentDataSchema.validate(df, lazy=True)
    LOGGER.info("Generated %s rows; schema validation passed.", len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
