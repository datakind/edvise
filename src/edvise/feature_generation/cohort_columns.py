"""
Cohort column names for :func:`~edvise.feature_generation.student.add_features`.

**Raw (bronze) schemas** — source of truth for uploads:

- **ES (Edvise):** :mod:`edvise.data_audit.schemas.raw_edvise_student`,
  :mod:`edvise.data_audit.schemas.raw_edvise_course`
- **PDP (Clearinghouse):** :mod:`edvise.data_audit.schemas.raw_cohort`,
  :mod:`edvise.data_audit.schemas.raw_course`

This module names columns on **standardized / silver** cohort frames. PDP uses
Clearinghouse-style names. ES keeps native Edvise field names on the cohort
frame (plus ``student_id`` from ``learner_id`` in
:func:`~edvise.dataio.read._prepare_edvise_cohort_after_validation`); use
:data:`ES_STUDENT_COHORT_COLUMNS` so ``student.add_features`` reads the right
physical columns.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StudentCohortColumns:
    """Source column names on the standardized cohort frame for student features."""

    cohort_year_col: str
    cohort_term_col: str
    program_of_study_term_1_col: str
    program_of_study_year_1_col: str
    pell_col: str
    gpa_term_1_col: str
    gpa_year_1_col: str
    #: Prefix before year index, e.g. ``number_of_credits_earned_year_`` + ``1`` → ``..._1``
    credit_earned_prefix: str = "number_of_credits_earned_year_"
    credit_attempted_prefix: str = "number_of_credits_attempted_year_"


PDP_STUDENT_COHORT_COLUMNS = StudentCohortColumns(
    cohort_year_col="cohort",
    cohort_term_col="cohort_term",
    program_of_study_term_1_col="program_of_study_term_1",
    program_of_study_year_1_col="program_of_study_year_1",
    pell_col="pell_status_first_year",
    gpa_term_1_col="gpa_group_term_1",
    gpa_year_1_col="gpa_group_year_1",
)

# Native Edvise cohort column names on silver (see ``RawEdviseStudentDataSchema``).
# ``program_of_study_year_1`` has no separate Edvise field; ``intended_program_type``
# is the closest year-one program intent signal.
ES_STUDENT_COHORT_COLUMNS = StudentCohortColumns(
    cohort_year_col="entry_year",
    cohort_term_col="entry_term",
    program_of_study_term_1_col="declared_major_at_entry",
    program_of_study_year_1_col="intended_program_type",
    pell_col="pell_recipient_year1",
    gpa_term_1_col="gpa_group_term_1",
    gpa_year_1_col="gpa_group_year_1",
)
