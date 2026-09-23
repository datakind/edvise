"""Discover eligible inference terms from legacy (non-PDP/ES) uploads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from edvise.dataio.filename_matching import filename_match_tokens
from edvise.student_selection.eligible_inference_terms import (
    EligibleInferenceTermsResult,
    apply_student_criteria,
    count_terms_from_labeled_courses,
    exclude_training_cohorts,
    invalid_eligible_terms,
    resolve_standardized_eligible_inference_terms,
    shared_student_id_column,
)

_TERM_LABEL_COLUMNS = (
    "term_desc",
    "semester",
    "term",
    "academic_term",
    "ir_term_code",
    "strm",
    "strm_admitted",
    "term_order",
    "term_order_term",
)
_STUDENT_TOKENS = frozenset({"student", "cohort", "learner"})
_COURSE_TOKENS = frozenset({"course"})


def _tokens_for_name(name: str) -> frozenset[str]:
    return filename_match_tokens(name)


def _pick_frame(
    frames_by_schema: Mapping[str, pd.DataFrame],
    frames_by_filename: Mapping[str, pd.DataFrame],
    *,
    schema_name: str,
    filename_tokens: frozenset[str],
) -> pd.DataFrame | None:
    schema_frame = frames_by_schema.get(schema_name)
    if schema_frame is not None:
        return schema_frame
    matches = [
        frame
        for name, frame in frames_by_filename.items()
        if filename_tokens & _tokens_for_name(name)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return pd.concat(matches, ignore_index=True)
    return None


def _first_present_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    columns = {str(column).strip().lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate in columns:
            return str(columns[candidate])
    return None


def _shared_id_any(
    students: pd.DataFrame, courses: pd.DataFrame, preferred: str | None
) -> str | None:
    found = shared_student_id_column(students, courses, preferred)
    if found is not None:
        return found
    student_cols = {str(column).strip().lower() for column in students.columns}
    for column in courses.columns:
        lowered = str(column).strip().lower()
        if lowered in student_cols and (
            "id" in lowered or lowered in {"emplid", "sis_id"}
        ):
            return str(column)
    return None


def resolve_legacy_eligible_inference_terms(
    frames_by_schema: Mapping[str, pd.DataFrame],
    config: dict[str, Any],
    *,
    frames_by_filename: Mapping[str, pd.DataFrame] | None = None,
    batch_name: str | None = None,
) -> EligibleInferenceTermsResult:
    """
    Discover eligible terms for legacy schools.

    Uses PDP/ES columns when present. Otherwise locates student and course files
    by schema tag or filename tokens and enumerates values from a native term
    column (``term_desc``, ``semester``, ``term_order``, CUNY codes, …).
    """
    named = frames_by_filename or {}
    students = _pick_frame(
        frames_by_schema, named, schema_name="STUDENT", filename_tokens=_STUDENT_TOKENS
    )
    courses = _pick_frame(
        frames_by_schema, named, schema_name="COURSE", filename_tokens=_COURSE_TOKENS
    )
    if students is None or courses is None:
        if len(named) >= 2:
            leftover = list(named.values())
            if students is None:
                students = leftover[0]
            if courses is None:
                courses = leftover[1] if leftover[1] is not students else leftover[0]
        elif len(frames_by_schema) >= 2:
            leftover = list(frames_by_schema.values())
            students = students if students is not None else leftover[0]
            courses = courses if courses is not None else leftover[-1]
        elif len(named) == 1:
            only = next(iter(named.values()))
            students = students if students is not None else only
            courses = courses if courses is not None else only
        elif len(frames_by_schema) == 1:
            only = next(iter(frames_by_schema.values()))
            students = students if students is not None else only
            courses = courses if courses is not None else only

    if students is None or courses is None:
        return invalid_eligible_terms(
            "Batch is missing readable student or course data.",
            batch_name,
        )

    if {"academic_term", "academic_year"} <= set(courses.columns):
        return resolve_standardized_eligible_inference_terms(
            students, courses, config, batch_name
        )

    student_id_column = _shared_id_any(students, courses, config.get("student_id_col"))
    if student_id_column is None:
        return invalid_eligible_terms(
            "Student and course files must share a student identifier column.",
            batch_name,
        )

    term_column = _first_present_column(courses, _TERM_LABEL_COLUMNS)
    if term_column is None:
        return invalid_eligible_terms(
            "Course file is missing a recognizable term column "
            f"({', '.join(_TERM_LABEL_COLUMNS)}).",
            batch_name,
        )

    selected_students = apply_student_criteria(students, config, student_id_column)
    selected_students, cohort_error = exclude_training_cohorts(
        selected_students,
        config.get("training_cohorts", []),
        required=False,
    )
    if cohort_error is not None:
        return invalid_eligible_terms(cohort_error, batch_name)

    return count_terms_from_labeled_courses(
        selected_students,
        courses,
        student_id_column=student_id_column,
        term_label_column=term_column,
        batch_name=batch_name,
        empty_reason="No term has eligible students with course data.",
    )
