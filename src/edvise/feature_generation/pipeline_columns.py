"""
Column / key configuration for the shared feature-generation pipeline (term → course →
section → student-term → cumulative).

PDP and ES differ mainly on **standardized** course/cohort physical names; derived
feature names (``course_id``, ``term_id``, …) stay the same unless you change the pipes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TermStandardizedColumns:
    """Year/term columns on the standardized course frame (before/after term features)."""

    academic_year_col: str
    academic_term_col: str


@dataclass(frozen=True, slots=True)
class SectionPipelineColumns:
    """Section groupby keys and source columns for section-level aggregates."""

    section_id_cols: tuple[str, ...]
    student_id_col: str = "student_id"
    course_passed_col: str = "course_passed"
    course_completed_col: str = "course_completed"
    course_grade_numeric_col: str = "course_grade_numeric"


@dataclass(frozen=True, slots=True)
class StudentTermAggregationColumns:
    """Passthrough and aggregation sources in ``aggregate_from_course_level_features``."""

    student_term_id_cols: tuple[str, ...] = ("student_id", "term_id")
    merge_student_on: tuple[str, ...] = ("institution_id", "student_id")
    institution_id_col: str = "institution_id"
    academic_year_col: str = "academic_year"
    academic_term_col: str = "academic_term"
    num_credits_attempted_col: str = "number_of_credits_attempted"
    num_credits_earned_col: str = "number_of_credits_earned"
    #: Column whose values are rolled into list agg output ``course_subjects``
    course_list_for_subjects_col: str = "course_cip"
    grade_col: str = "grade"
    grade_numeric_col: str = "course_grade_numeric"
    section_grade_numeric_mean_col: str = "section_course_grade_numeric_mean"


@dataclass(frozen=True, slots=True)
class StudentTermFeatureColumns:
    """
    Column names on the **post-merge** student-term frame read by
    :func:`~edvise.feature_generation.student_term.add_features`.

    These are mostly aggregated / derived feature names (e.g. ``num_credits_attempted``),
    not raw standardized course field names—see :class:`StudentTermAggregationColumns`
    for the latter.
    """

    cohort_start_dt_col: str = "cohort_start_dt"
    term_start_dt_col: str = "term_start_dt"
    term_program_of_study_col: str = "term_program_of_study"
    #: List-of-subject-areas column produced by course→student-term aggregation
    course_subject_areas_col: str = "course_subject_areas"
    #: Course count from aggregation (denominator for ``frac_courses_*`` features)
    num_courses_col: str = "num_courses"
    num_credits_earned_col: str = "num_credits_earned"
    num_credits_attempted_col: str = "num_credits_attempted"
    #: Dummy sum column for other-institution enrollment (from val-equals aggregation)
    other_institution_enrollment_num_course_col: str = (
        "num_courses_enrolled_at_other_institution_s_Y"
    )
    sections_num_students_enrolled_col: str = "sections_num_students_enrolled"
    sections_num_students_passed_col: str = "sections_num_students_passed"
    sections_num_students_completed_col: str = "sections_num_students_completed"
    #: Output of ``year_of_enrollment_at_cohort_inst`` in the same ``assign`` batch
    enrollment_year_at_cohort_col: str = "year_of_enrollment_at_cohort_inst"
    first_year_to_certificate_at_cohort_inst_col: str = (
        "first_year_to_certificate_at_cohort_inst"
    )
    years_to_latest_certificate_at_cohort_inst_col: str = (
        "years_to_latest_certificate_at_cohort_inst"
    )
    first_year_to_certificate_at_other_inst_col: str = (
        "first_year_to_certificate_at_other_inst"
    )
    years_to_latest_certificate_at_other_inst_col: str = (
        "years_to_latest_certificate_at_other_inst"
    )
    student_program_of_study_area_term_1_col: str = "student_program_of_study_area_term_1"
    student_program_of_study_area_year_1_col: str = "student_program_of_study_area_year_1"
    #: Target column for ``num_courses_in_term_program_of_study_area`` (same-batch assign)
    term_program_of_study_area_col: str = "term_program_of_study_area"
    frac_courses_passed_col: str = "frac_courses_passed"
    frac_courses_completed_col: str = "frac_courses_completed"
    frac_sections_students_passed_col: str = "frac_sections_students_passed"
    frac_sections_students_completed_col: str = "frac_sections_students_completed"


@dataclass(frozen=True, slots=True)
class CumulativePipelineColumns:
    """Grouping and chronological sort for ``cumulative.add_features``."""

    student_id_cols: tuple[str, ...] = ("institution_id", "student_id")
    sort_cols: tuple[str, ...] = ("academic_year", "academic_term")


@dataclass(frozen=True, slots=True)
class FeaturePipelineColumns:
    """Bundle of pipeline column maps (one object on ``FeatureGenerationBackend``)."""

    term: TermStandardizedColumns
    section: SectionPipelineColumns
    student_term_agg: StudentTermAggregationColumns
    student_term_features: StudentTermFeatureColumns
    cumulative: CumulativePipelineColumns


def _pdp_term() -> TermStandardizedColumns:
    return TermStandardizedColumns(
        academic_year_col="academic_year",
        academic_term_col="academic_term",
    )


def _es_term() -> TermStandardizedColumns:
    # Adjust if ES standardized course uses different year/term column names.
    return TermStandardizedColumns(
        academic_year_col="academic_year",
        academic_term_col="academic_term",
    )


def ensure_student_term_academic_cols_match_term(
    student_term_agg: StudentTermAggregationColumns,
    term: TermStandardizedColumns,
) -> StudentTermAggregationColumns:
    """Align passthrough academic year/term names with ``TermStandardizedColumns``."""
    if (
        student_term_agg.academic_year_col == term.academic_year_col
        and student_term_agg.academic_term_col == term.academic_term_col
    ):
        return student_term_agg
    return StudentTermAggregationColumns(
        student_term_id_cols=student_term_agg.student_term_id_cols,
        merge_student_on=student_term_agg.merge_student_on,
        institution_id_col=student_term_agg.institution_id_col,
        academic_year_col=term.academic_year_col,
        academic_term_col=term.academic_term_col,
        num_credits_attempted_col=student_term_agg.num_credits_attempted_col,
        num_credits_earned_col=student_term_agg.num_credits_earned_col,
        course_list_for_subjects_col=student_term_agg.course_list_for_subjects_col,
        grade_col=student_term_agg.grade_col,
        grade_numeric_col=student_term_agg.grade_numeric_col,
        section_grade_numeric_mean_col=student_term_agg.section_grade_numeric_mean_col,
    )


def ensure_cumulative_sort_matches_term(
    cumulative: CumulativePipelineColumns,
    term: TermStandardizedColumns,
) -> CumulativePipelineColumns:
    """Use the same year/term names for cumulative sort as for term features."""
    if cumulative.sort_cols == (term.academic_year_col, term.academic_term_col):
        return cumulative
    return CumulativePipelineColumns(
        student_id_cols=cumulative.student_id_cols,
        sort_cols=(term.academic_year_col, term.academic_term_col),
    )


def build_backend_pipeline_columns(
    *,
    term: TermStandardizedColumns,
    section: SectionPipelineColumns,
    student_term_agg: StudentTermAggregationColumns | None = None,
    student_term_features: StudentTermFeatureColumns | None = None,
    cumulative: CumulativePipelineColumns | None = None,
) -> FeaturePipelineColumns:
    """Merge term column names into student-term passthrough and cumulative sort."""
    sta = student_term_agg or StudentTermAggregationColumns()
    sta = ensure_student_term_academic_cols_match_term(sta, term)
    cum = cumulative or CumulativePipelineColumns()
    cum = ensure_cumulative_sort_matches_term(cum, term)
    return FeaturePipelineColumns(
        term=term,
        section=section,
        student_term_agg=sta,
        student_term_features=student_term_features or StudentTermFeatureColumns(),
        cumulative=cum,
    )


PDP_FEATURE_PIPELINE_COLUMNS = build_backend_pipeline_columns(
    term=_pdp_term(),
    section=SectionPipelineColumns(
        section_id_cols=("term_id", "course_id", "section_id"),
    ),
)

ES_FEATURE_PIPELINE_COLUMNS = build_backend_pipeline_columns(
    term=_es_term(),
    section=SectionPipelineColumns(
        section_id_cols=("term_id", "course_id", "section_id"),
    ),
)
