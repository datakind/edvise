"""
Cohort column names for :func:`~edvise.feature_generation.student.add_features`.

PDP uses Clearinghouse-style names; Edvise Schema (ES) may differ once the
standardized cohort contract is fixed—update :data:`ES_STUDENT_COHORT_COLUMNS`
to match your ES ``df_cohort_standardized`` outputs.
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

# TODO: align with standardized ES cohort columns when schema is finalized.
# Currently mirrors PDP so ES jobs using the same column layout keep working.
ES_STUDENT_COHORT_COLUMNS = StudentCohortColumns(
    cohort_year_col="cohort",
    cohort_term_col="cohort_term",
    program_of_study_term_1_col="program_of_study_term_1",
    program_of_study_year_1_col="program_of_study_year_1",
    pell_col="pell_status_first_year",
    gpa_term_1_col="gpa_group_term_1",
    gpa_year_1_col="gpa_group_year_1",
)
