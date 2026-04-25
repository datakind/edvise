"""
Build feature-generation specs from columns present on Edvise standardized
cohort/course frames (using ``ES_*_INPUT_COLUMNS`` names directly).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from .column_names import (
    CohortInputColumns,
    CourseInputColumns,
    CumulativeExpandingColumnSpec,
    CumulativeFeatureSpec,
    CourseFeatureSpec,
    SectionFeatureSpec,
    StudentFeatureSpec,
    StudentTermAddFeatureSpec,
    StudentTermAggregateSpec,
    TermFeatureSpec,
    ES_STUDENT_FEATURE_SPEC_DEFAULT,
)


@dataclass(frozen=True, slots=True)
class EdviseFeatureSpecBundle:
    """Specs selected from Edvise ``df_cohort`` / ``df_course`` column presence."""

    student: StudentFeatureSpec
    course: CourseFeatureSpec
    term: TermFeatureSpec
    section: SectionFeatureSpec
    student_term_aggregate: StudentTermAggregateSpec
    student_term_add: StudentTermAddFeatureSpec
    cumulative: CumulativeFeatureSpec


def _has(df: pd.DataFrame, col: str | None) -> bool:
    return bool(col) and col in df.columns


def build_edvise_feature_specs(
    df_cohort: pd.DataFrame,
    df_course: pd.DataFrame,
    *,
    cohort_cols: CohortInputColumns,
    course_cols: CourseInputColumns,
) -> EdviseFeatureSpecBundle:
    """
    Disable a feature when its **inputs** are absent (or a column config field is ``None``).

    Cumulative expanding columns are turned off when the upstream **student-term** inputs
    they aggregate are not expected on the final frame; that is approximated from the
    course/term/section inputs available on ``df_course`` plus common derived names.
    """
    has_cohort_keys = _has(
        df_cohort, cohort_cols.cohort_year_col
    ) and _has(df_cohort, cohort_cols.cohort_term_col)
    has_pell = _has(df_cohort, cohort_cols.pell_status_col)
    has_gpa = (
        cohort_cols.gpa_group_term_1_col is not None
        and cohort_cols.gpa_group_year_1_col is not None
        and _has(df_cohort, cohort_cols.gpa_group_term_1_col)
        and _has(df_cohort, cohort_cols.gpa_group_year_1_col)
    )
    has_frac_credits = False
    if (
        cohort_cols.credits_earned_year_template
        and cohort_cols.credits_attempted_year_template
    ):
        for yr in (1, 2, 3, 4):
            e = cohort_cols.earned_col(yr)
            a = cohort_cols.attempted_col(yr)
            if e in df_cohort.columns and a in df_cohort.columns:
                has_frac_credits = True
                break
    student = dataclasses.replace(
        ES_STUDENT_FEATURE_SPEC_DEFAULT,
        cohort_id=has_cohort_keys,
        cohort_start_dt=has_cohort_keys,
        pell=has_pell,
        diff_gpa=has_gpa,
        frac_credits_by_year=has_frac_credits,
    )

    g = _has(df_course, course_cols.grade)
    cip = _has(df_course, course_cols.course_cip)
    pfx = _has(df_course, course_cols.course_prefix)
    num = _has(df_course, course_cols.course_number)
    course = CourseFeatureSpec(
        course_id=pfx and num,
        course_subject_area=cip,
        course_passed=g,
        course_completed=g,
        course_level=bool(num and g),
        course_grade_numeric=g,
        course_grade=g,
    )

    has_ay = _has(df_course, course_cols.academic_year)
    has_at = _has(df_course, course_cols.academic_term)
    term_on = has_ay and has_at
    term = TermFeatureSpec(
        term_id=term_on,
        term_start_dt=term_on,
        term_rank=term_on,
        term_rank_core=term_on,
        term_rank_noncore=term_on,
        term_in_peak_covid=term_on,
        term_is_core=term_on,
        term_is_noncore=term_on,
    )

    sec_id = _has(df_course, course_cols.section_id) or _has(
        df_course, "course_section_id"
    )
    section = SectionFeatureSpec(
        section_num_students_enrolled=sec_id and pfx and num and g,
        section_num_students_passed=sec_id and pfx and num and g,
        section_num_students_completed=sec_id and pfx and num and g,
        section_course_grade_numeric_mean=sec_id and pfx and num and g,
    )

    # Dummy and value_equality: need at least one source column; aggregate filters missing.
    dummy_source_cols = (
        "course_type",
        "delivery_method",
        "math_or_english_gateway",
        "co_requisite_course",
        "course_instructor_employment_status",
        "course_instructor_rank",
    )
    dummies = any(c in df_course.columns for c in dummy_source_cols) or (g and num)
    dummies = dummies or any(
        c in df_course.columns
        for c in (
            "instructional_modality",
            "gateway_or_developmental_flag",
            "instructor_appointment_status",
            "gen_ed_flag",
        )
    )
    # g and num → course.add_features will add course_level, course_grade for dummies
    has_core = _has(df_course, course_cols.core_course) or _has(
        df_course, "gen_ed_flag"
    )
    has_ct = _has(df_course, course_cols.course_type)
    has_oth = (
        _has(df_course, course_cols.enrolled_at_other_institution_s)
        if course_cols.enrolled_at_other_institution_s
        else False
    )
    value_equality = has_core or has_ct or has_oth
    st_agg = StudentTermAggregateSpec(
        summary_aggregations=True,
        dummies=bool(dummies),
        value_equality=bool(value_equality),
        multicol_grade=bool(g),
    )

    c_cert = bool(
        cohort_cols.first_year_to_certificate_at_cohort_inst
        and cohort_cols.years_to_latest_certificate_at_cohort_inst
        and _has(
            df_cohort,
            cohort_cols.first_year_to_certificate_at_cohort_inst,
        )
        and _has(
            df_cohort,
            cohort_cols.years_to_latest_certificate_at_cohort_inst,
        )
    )
    o_cert = bool(
        cohort_cols.first_year_to_certificate_at_other_inst
        and cohort_cols.years_to_latest_certificate_at_other_inst
        and _has(
            df_cohort,
            cohort_cols.first_year_to_certificate_at_other_inst,
        )
        and _has(
            df_cohort,
            cohort_cols.years_to_latest_certificate_at_other_inst,
        )
    )
    # ``cohort_start_dt`` is produced by :func:`student.add_features` when ``cohort_id`` runs.
    will_have_cohort_start = has_cohort_keys
    has_term_prog = _has(df_course, course_cols.term_program_of_study)
    has_cr = _has(
        df_course, course_cols.number_of_credits_earned
    ) and _has(df_course, course_cols.number_of_credits_attempted)
    st_add = StudentTermAddFeatureSpec(
        year_of_enrollment_at_cohort_inst=will_have_cohort_start,
        student_certificates=will_have_cohort_start and (c_cert or o_cert),
        term_cohort_and_transfer_flags=will_have_cohort_start,
        program_of_study_area=has_term_prog,
        credit_fraction_and_intensity=has_cr,
        num_courses_in_program_area=has_term_prog,
        num_course_by_category_fracs=True,
        section_student_fractions=bool(section.section_num_students_enrolled and g),
        student_rate_vs_section_fractions=bool(
            section.section_num_students_enrolled and g
        ),
        program_change_from_prior_term=bool(has_term_prog and term_on),
    )

    # Cumulative: disable expanding sub-columns that cannot exist without Term/course/section outputs.
    exp = CumulativeExpandingColumnSpec(
        term_id=term_on,
        term_in_peak_covid=term_on,
        term_is_core=term_on,
        term_is_noncore=term_on,
        term_is_while_student_enrolled_at_other_inst=bool(has_oth and term_on),
        term_is_pre_cohort=will_have_cohort_start and term_on,
        course_level_mean=bool(num and g),
        course_grade_numeric_mean=bool(g),
        num_courses=True,
        num_credits_attempted=has_cr,
        num_credits_earned=has_cr,
        student_pass_rate_above_sections_avg=bool(
            section.section_num_students_enrolled and g
        ),
        student_completion_rate_above_sections_avg=bool(
            section.section_num_students_enrolled and g
        ),
    )
    cum = CumulativeFeatureSpec(
        expanding_aggregate=True,
        expanding_columns=exp,
        cumnum_unique_repeated=True,
        cumfrac_terms_enrolled=term_on,
        term_differences=term_on,
    )

    return EdviseFeatureSpecBundle(
        student=student,
        course=course,
        term=term,
        section=section,
        student_term_aggregate=st_agg,
        student_term_add=st_add,
        cumulative=cum,
    )
