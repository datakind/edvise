"""Quota buckets for academic coverage during feature selection.

Prefix twins (``frac_`` / ``num_`` / ``cum*`` of the same stem) still collapse
through :func:`edvise.model_prep.target_associations.feature_family`. This map
answers a different question: which coverage bucket a column belongs to.

After redundancy drops, selection restores the strongest dropped member of a
bucket that would otherwise be empty. It does not add a second pass that keeps
only one prefix twin.
"""

from __future__ import annotations

_GRADE_TOKENS = ("a", "b", "c", "d", "f", "p", "w", "i", "m", "other", "audit")
_LEVEL_TOKENS = ("0", "0_1", "1", "2", "3", "4", "5", "6", "7", "8", "9")
_YEARS = (1, 2, 3, 4)


def _prefixed(prefix: str, tokens: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}{token}" for token in tokens)


def _term_diffs(stem: str) -> tuple[str, ...]:
    return (
        f"{stem}_diff_prev_term",
        f"{stem}_diff_term_1_to_term_2",
        f"{stem}_diff_term_2_to_term_3",
        f"{stem}_diff_term_3_to_term_4",
    )


def _year_cols(template: str) -> tuple[str, ...]:
    return tuple(template.format(year=year) for year in _YEARS)


_GRADES = (
    "course_grade_numeric_mean",
    "cummean_course_grade_numeric_mean",
    "cummin_course_grade_numeric_mean",
    *_prefixed("frac_courses_course_grade_", _GRADE_TOKENS),
    *_prefixed("num_courses_course_grade_", _GRADE_TOKENS),
    *_prefixed("cumfrac_num_courses_course_grade_", _GRADE_TOKENS),
    # PDP snapshots. Absent on ES, so the quota does not try to restore them.
    "gpa_group_term_1",
    "gpa_group_year_1",
)

_CREDITS = (
    "num_credits_earned",
    "num_credits_attempted",
    "frac_credits_earned",
    "cumsum_num_credits_earned",
    "cummean_num_credits_earned",
    "cummin_num_credits_earned",
    "cumsum_num_credits_attempted",
    "cummean_num_credits_attempted",
    "cummin_num_credits_attempted",
    *_year_cols("number_of_credits_earned_year_{year}"),
    *_year_cols("number_of_credits_attempted_year_{year}"),
    *_year_cols("frac_credits_earned_year_{year}"),
    "credits_earned_ap",
    "credits_earned_dual_enrollment",
)

_COURSE_SUCCESS = (
    "frac_courses_passed",
    "num_courses_passed",
    "cumfrac_num_courses_passed",
    "frac_courses_completed",
    "num_courses_completed",
    "cumfrac_num_courses_completed",
    "frac_courses_grade_above_section_avg",
    "num_courses_grade_above_section_avg",
    "cumfrac_num_courses_grade_above_section_avg",
    "frac_courses_grade_is_failing_or_withdrawal",
    "num_courses_grade_is_failing_or_withdrawal",
    "cumfrac_num_courses_grade_is_failing_or_withdrawal",
    "student_pass_rate_above_sections_avg",
    "cumsum_student_pass_rate_above_sections_avg",
    "student_completion_rate_above_sections_avg",
    "cumsum_student_completion_rate_above_sections_avg",
)

_GRADE_VARIABILITY = (
    "course_grade_numeric_std",
    "cumstd_course_grade_numeric_mean",
)

_COURSE_LEVEL = (
    "course_level_mean",
    "course_level_std",
    "cummean_course_level_mean",
    "cummin_course_level_mean",
    "cumstd_course_level_mean",
    *_prefixed("frac_courses_course_level_", _LEVEL_TOKENS),
    *_prefixed("num_courses_course_level_", _LEVEL_TOKENS),
    *_prefixed("cumfrac_num_courses_course_level_", _LEVEL_TOKENS),
)

_MOMENTUM = (
    *_term_diffs("course_grade_numeric_mean"),
    *_term_diffs("course_level_mean"),
    *_term_diffs("num_credits_earned"),
    *_term_diffs("num_courses"),
    "diff_gpa_term_1_to_year_1",
    "term_program_of_study_changed_prev_term",
    "term_program_of_study_area_changed_prev_term",
    "term_degree_changed_prev_term",
)

_PROGRAM_FIT = (
    "term_program_of_study",
    "term_program_of_study_area",
    "program_of_study_term_1",
    "program_of_study_year_1",
    "frac_courses_in_term_program_of_study_area",
    "num_courses_in_term_program_of_study_area",
    "cumfrac_num_courses_in_term_program_of_study_area",
    "term_degree",
    "credential_type_sought_year_1",
    "declared_major_at_entry",
    "intended_program_type",
)

_SECTION_CONTEXT = (
    "section_num_students_enrolled_mean",
    "section_num_students_enrolled_std",
    "frac_sections_students_passed",
    "frac_sections_students_completed",
)

# Enrollment intensity and terms enrolled. Course-load columns such as
# ``num_courses`` stay outside this bucket; their term-to-term diffs are momentum.
_ENROLLMENT = (
    "enrollment_intensity_first_term",
    "enrollment_type",
    "student_term_enrollment_intensity",
    "attendance_status_term_1",
    "dual_and_summer_enrollment",
    "year_of_enrollment_at_cohort_inst",
    "cumfrac_terms_enrolled",
    "cumnum_terms_enrolled",
    "cumfrac_core_terms_enrolled",
    "cumnum_core_terms_enrolled",
    "cumfrac_noncore_terms_enrolled",
    "cumnum_noncore_terms_enrolled",
    "term_is_pre_cohort",
    "cumsum_term_is_pre_cohort",
    "term_is_while_student_enrolled_at_other_inst",
    "cumsum_term_is_while_student_enrolled_at_other_inst",
    "frac_courses_enrolled_at_other_institution_s_y",
    "num_courses_enrolled_at_other_institution_s_y",
    "cumfrac_num_courses_enrolled_at_other_institution_s_y",
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "grades": _GRADES,
    "credits": _CREDITS,
    "course_success": _COURSE_SUCCESS,
    "grade_variability": _GRADE_VARIABILITY,
    "course_level": _COURSE_LEVEL,
    "momentum": _MOMENTUM,
    "program_fit": _PROGRAM_FIT,
    "section_context": _SECTION_CONTEXT,
    "enrollment": _ENROLLMENT,
}

_COLUMN_TO_FAMILY: dict[str, str] = {
    column: family for family, columns in FEATURE_FAMILIES.items() for column in columns
}

if len(_COLUMN_TO_FAMILY) != sum(len(columns) for columns in FEATURE_FAMILIES.values()):
    raise RuntimeError("feature quota families overlap")


def quota_family(name: str) -> str | None:
    """Return the coverage bucket for ``name``, or None when it has no quota."""
    return _COLUMN_TO_FAMILY.get(name)
