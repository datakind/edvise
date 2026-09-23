"""Shared eligible-inference-term enumeration after frames are joinable."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from edvise.shared.utils import cohort_pair_columns
from edvise.student_selection.filter_inference import exclude_training_cohort_students
from edvise.student_selection.select_students_attributes import (
    select_students_by_attributes,
)

LOGGER = logging.getLogger(__name__)

_ACADEMIC_TERM_ORDER = {"FALL": 1, "WINTER": 2, "SPRING": 3, "SUMMER": 4}
_STUDENT_ID_COLUMNS = ("student_id", "study_id", "learner_id")


@dataclass(frozen=True)
class EligibleInferenceTermRow:
    """An academic term with students eligible for inference."""

    term_label: str
    valid_student_count: int


@dataclass(frozen=True)
class EligibleInferenceTermsResult:
    """Eligible academic terms for a model and input batch."""

    status: Literal["valid", "invalid"]
    batch_name: str | None = None
    terms: tuple[EligibleInferenceTermRow, ...] = ()
    reason: str | None = None


def invalid_eligible_terms(
    reason: str, batch_name: str | None = None
) -> EligibleInferenceTermsResult:
    """Build a consistent invalid eligible-terms result."""
    LOGGER.warning("Eligible inference terms unavailable: %s", reason)
    return EligibleInferenceTermsResult(
        status="invalid",
        batch_name=batch_name,
        reason=reason,
    )


def academic_year_sort_value(value: object) -> int:
    """Return the ending year from an academic-year label."""
    years = re.findall(r"\d{2,4}", str(value))
    if not years:
        return -1
    year = int(years[-1])
    return year if year >= 100 else 2000 + year


def normalize_identifier_series(values: pd.Series) -> pd.Series:
    """Canonicalize identifiers so int, float, and string forms join."""
    stripped = values.astype("string").str.strip()
    numeric = pd.to_numeric(stripped, errors="coerce")
    is_whole_number = numeric.notna() & (numeric == numeric.round())
    result = stripped.fillna("")
    if bool(is_whole_number.any()):
        whole_as_int = numeric.loc[is_whole_number].astype("int64").astype(str)
        result = result.mask(is_whole_number, whole_as_int)
    return result.replace({"<NA>": "", "nan": "", "None": "", "NaN": ""})


def shared_student_id_column(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    preferred: str | None = None,
) -> str | None:
    """Return the config student id column, else a supported identifier in both files."""
    if preferred and preferred in students.columns and preferred in courses.columns:
        return preferred
    for column in _STUDENT_ID_COLUMNS:
        if column in students.columns and column in courses.columns:
            return column
    return None


def apply_student_criteria(
    students: pd.DataFrame,
    config: dict[str, Any],
    student_id_column: str,
) -> pd.DataFrame:
    """Apply upload-available training selection criteria."""
    criteria = config.get("student_criteria")
    if not isinstance(criteria, dict) or not criteria:
        return students
    applicable = {
        column: value for column, value in criteria.items() if column in students
    }
    skipped = sorted(str(column) for column in criteria if column not in students)
    if skipped:
        LOGGER.warning(
            "Skipping student_criteria columns missing from the student file: %s",
            ", ".join(skipped),
        )
    if not applicable:
        return students
    work = students.copy()
    prepared: dict[str, Any] = {}
    for column, value in applicable.items():
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            work[column] = (
                work[column].astype("string").str.strip().str.casefold().fillna("")
            )
            prepared[column] = [item.strip().casefold() for item in value]
        elif isinstance(value, str):
            work[column] = (
                work[column].astype("string").str.strip().str.casefold().fillna("")
            )
            prepared[column] = value.strip().casefold()
        else:
            prepared[column] = value
    selected = select_students_by_attributes(
        work, student_id_cols=student_id_column, **prepared
    )
    selected_ids = set(
        normalize_identifier_series(pd.Series(selected.index))
        .replace("", pd.NA)
        .dropna()
    )
    student_ids = normalize_identifier_series(students[student_id_column])
    return students[student_ids.isin(selected_ids)].copy()


def exclude_training_cohorts(
    students: pd.DataFrame,
    training_cohorts: list[str],
    *,
    required: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """Exclude students whose entry cohort was used for model training."""
    if not training_cohorts:
        return students, None
    cohort_pair = cohort_pair_columns(students)
    if cohort_pair is None:
        if not required:
            LOGGER.warning(
                "Skipping training-cohort exclusion; student file has no cohort columns."
            )
            return students, None
        return (
            students.iloc[0:0],
            "Student file is missing entry cohort columns needed to exclude "
            "training cohorts.",
        )
    year_column, term_column = cohort_pair
    normalized_students = students.copy()
    for column in (year_column, term_column):
        normalized_students[column] = normalized_students[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    try:
        filtered_students = exclude_training_cohort_students(
            normalized_students,
            training_cohorts=training_cohorts,
            cohort_term_column=term_column,
            cohort_column=year_column,
        )
        return students.loc[filtered_students.index].copy(), None
    except ValueError as exc:
        return students.iloc[0:0], str(exc)


def count_terms_from_labeled_courses(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    *,
    student_id_column: str,
    term_label_column: str,
    batch_name: str | None = None,
    empty_reason: str = "No academic term has eligible students with course data.",
) -> EligibleInferenceTermsResult:
    """Group eligible course rows by a prepared term-label column."""
    if term_label_column not in courses.columns:
        return invalid_eligible_terms(
            f"Course file is missing term label column: {term_label_column}.",
            batch_name,
        )
    eligible_student_ids = set(
        normalize_identifier_series(students[student_id_column])
        .replace("", pd.NA)
        .dropna()
        .unique()
    )
    labeled = courses.copy()
    labeled["_student_id"] = normalize_identifier_series(labeled[student_id_column])
    labeled["_term_label"] = (
        labeled[term_label_column].astype(str).str.strip().str.lower()
    )
    labeled = labeled[
        labeled["_student_id"].isin(eligible_student_ids)
        & labeled["_term_label"].ne("")
        & labeled[term_label_column].notna()
    ]
    terms: list[EligibleInferenceTermRow] = []
    for term_label, rows in labeled.groupby("_term_label"):
        valid_student_count = int(
            normalize_identifier_series(rows[student_id_column])
            .replace("", pd.NA)
            .nunique(dropna=True)
        )
        if valid_student_count:
            terms.append(
                EligibleInferenceTermRow(
                    term_label=str(term_label),
                    valid_student_count=valid_student_count,
                )
            )
    terms.sort(
        key=lambda term: (
            academic_year_sort_value(term.term_label),
            _ACADEMIC_TERM_ORDER.get(term.term_label.split(" ", 1)[0].upper(), -1),
        ),
        reverse=True,
    )
    if not terms:
        return invalid_eligible_terms(empty_reason, batch_name)
    return EligibleInferenceTermsResult(
        status="valid",
        batch_name=batch_name,
        terms=tuple(terms),
    )


def resolve_standardized_eligible_inference_terms(
    students: pd.DataFrame,
    courses: pd.DataFrame,
    config: dict[str, Any],
    batch_name: str | None = None,
) -> EligibleInferenceTermsResult:
    """Return newest-first terms from PDP/ES ``academic_term`` + ``academic_year``."""
    required_course_columns = {"academic_term", "academic_year"}
    missing_course_columns = sorted(required_course_columns - set(courses.columns))
    if missing_course_columns:
        return invalid_eligible_terms(
            "Course file is missing academic term columns: "
            + ", ".join(missing_course_columns)
            + ".",
            batch_name,
        )

    student_id_column = shared_student_id_column(
        students, courses, config.get("student_id_col")
    )
    if student_id_column is None:
        return invalid_eligible_terms(
            "Student and course files must share student_id, study_id, or learner_id.",
            batch_name,
        )

    selected_students = apply_student_criteria(students, config, student_id_column)
    selected_students, cohort_error = exclude_training_cohorts(
        selected_students,
        config.get("training_cohorts", []),
    )
    if cohort_error is not None:
        return invalid_eligible_terms(cohort_error, batch_name)

    work = courses.copy()
    work["_eligible_term_label"] = (
        work["academic_term"].astype(str).str.strip().str.lower()
        + " "
        + work["academic_year"].astype(str).str.strip().str.lower()
    )
    work = work[
        work["academic_term"].notna()
        & work["academic_year"].notna()
        & work["_eligible_term_label"].str.strip().ne("")
    ]
    return count_terms_from_labeled_courses(
        selected_students,
        work,
        student_id_column=student_id_column,
        term_label_column="_eligible_term_label",
        batch_name=batch_name,
    )
