from __future__ import annotations

"""Input column contracts and per-step feature specs for :mod:`edvise.feature_generation`.

:mod:`edvise.feature_generation.shared` only holds shared helpers; it does not
define an ``add_features`` entry point, so there is no ``SharedFeatureSpec`` here.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CohortInputColumns:
    """
    Cohort / learner row **inputs** to feature generation (and credential-year names
    for :mod:`student_term` when those columns exist).

    A field is ``str | None`` when a product has no raw comparable (e.g. Edvise has
    no PDP-style per-year credit templates, GPA buckets, or credential *year* columns
    in the long PDP roster).
    """

    student_id: str
    cohort_year_col: str
    cohort_term_col: str
    pell_status_col: str
    # PDP standard cohort; Edvise raw: often added in silver. ``None`` = not on frame.
    institution_id: str | None = None
    gpa_group_term_1_col: str | None = None
    gpa_group_year_1_col: str | None = None
    credits_earned_year_template: str | None = None
    credits_attempted_year_template: str | None = None
    first_year_to_associates_at_cohort_inst: str | None = None
    years_to_latest_associates_at_cohort_inst: str | None = None
    first_year_to_certificate_at_cohort_inst: str | None = None
    years_to_latest_certificate_at_cohort_inst: str | None = None
    first_year_to_associates_at_other_inst: str | None = None
    years_to_latest_associates_at_other_inst: str | None = None
    first_year_to_certificate_at_other_inst: str | None = None
    years_to_latest_certificate_at_other_inst: str | None = None

    def earned_col(self, year: int) -> str:
        if self.credits_earned_year_template is None:
            raise ValueError("credits_earned_year_template is not set for this product")
        return self.credits_earned_year_template.format(yr=year)

    def attempted_col(self, year: int) -> str:
        if self.credits_attempted_year_template is None:
            raise ValueError("credits_attempted_year_template is not set for this product")
        return self.credits_attempted_year_template.format(yr=year)


@dataclass(frozen=True, slots=True)
class CourseInputColumns:
    """
    Physical column names on a course-level row used (or to be used) by feature
    generation. One logical field per attribute; values differ by product
    (e.g. :attr:`course_cip` is ``\"course_cip\"`` for PDP, ``\"department\"`` for Edvise
    in the current map).

    ``| None`` = that product has no raw column for this role (see :data:`ES_COURSE_INPUT_COLUMNS`).
    """

    student_id: str
    academic_year: str
    academic_term: str
    course_prefix: str
    course_number: str
    grade: str
    course_cip: str
    number_of_credits_attempted: str
    number_of_credits_earned: str
    term_program_of_study: str
    delivery_method: str
    math_or_english_gateway: str
    course_instructor_employment_status: str
    core_course: str
    section_id: str
    institution_id: str | None = None
    course_type: str | None = None
    co_requisite_course: str | None = None
    course_instructor_rank: str | None = None
    enrolled_at_other_institution_s: str | None = None


@dataclass(frozen=True, slots=True)
class StudentFeatureSpec:
    """Toggles for :func:`edvise.feature_generation.student.add_features`."""

    cohort_id: bool = True
    cohort_start_dt: bool = True
    pell: bool = True
    diff_gpa: bool = True
    frac_credits_by_year: bool = True

    @classmethod
    def all(cls) -> StudentFeatureSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class CourseFeatureSpec:
    """Toggles for :func:`edvise.feature_generation.course.add_features`."""

    course_id: bool = True
    course_subject_area: bool = True
    course_passed: bool = True
    course_completed: bool = True
    course_level: bool = True
    course_grade_numeric: bool = True
    course_grade: bool = True

    @classmethod
    def all(cls) -> CourseFeatureSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class TermFeatureSpec:
    """Toggles for :func:`edvise.feature_generation.term.add_features`."""

    term_id: bool = True
    term_start_dt: bool = True
    term_rank: bool = True
    term_rank_core: bool = True
    term_rank_noncore: bool = True
    term_in_peak_covid: bool = True
    term_is_core: bool = True
    term_is_noncore: bool = True

    @classmethod
    def all(cls) -> TermFeatureSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class SectionFeatureSpec:
    """Toggles for :func:`edvise.feature_generation.section.add_features`."""

    section_num_students_enrolled: bool = True
    section_num_students_passed: bool = True
    section_num_students_completed: bool = True
    section_course_grade_numeric_mean: bool = True

    @classmethod
    def all(cls) -> SectionFeatureSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class CumulativeExpandingColumnSpec:
    """
    Toggles for which (column, agg) families feed the expanding block in
    :func:`edvise.feature_generation.cumulative.add_features`.
    """

    term_id: bool = True
    term_in_peak_covid: bool = True
    term_is_core: bool = True
    term_is_noncore: bool = True
    term_is_while_student_enrolled_at_other_inst: bool = True
    term_is_pre_cohort: bool = True
    course_level_mean: bool = True
    course_grade_numeric_mean: bool = True
    num_courses: bool = True
    num_credits_attempted: bool = True
    num_credits_earned: bool = True
    student_pass_rate_above_sections_avg: bool = True
    student_completion_rate_above_sections_avg: bool = True

    @classmethod
    def all(cls) -> CumulativeExpandingColumnSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class CumulativeFeatureSpec:
    """Toggles for :func:`edvise.feature_generation.cumulative.add_features`."""

    expanding_aggregate: bool = True
    expanding_columns: CumulativeExpandingColumnSpec = field(
        default_factory=CumulativeExpandingColumnSpec
    )
    cumnum_unique_repeated: bool = True
    cumfrac_terms_enrolled: bool = True
    term_differences: bool = True

    @classmethod
    def all(cls) -> CumulativeFeatureSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class StudentTermAggregateSpec:
    """Toggles for :func:`edvise.feature_generation.student_term.aggregate_from_course_level_features`."""

    summary_aggregations: bool = True
    dummies: bool = True
    value_equality: bool = True
    multicol_grade: bool = True

    @classmethod
    def all(cls) -> StudentTermAggregateSpec:
        return cls()


@dataclass(frozen=True, slots=True)
class StudentTermAddFeatureSpec:
    """
    Toggles for :func:`edvise.feature_generation.student_term.add_features`
    (post-aggregate, joined to students).
    """

    year_of_enrollment_at_cohort_inst: bool = True
    student_certificates: bool = True
    term_cohort_and_transfer_flags: bool = True
    program_of_study_area: bool = True
    credit_fraction_and_intensity: bool = True
    num_courses_in_program_area: bool = True
    num_course_by_category_fracs: bool = True
    section_student_fractions: bool = True
    student_rate_vs_section_fractions: bool = True
    program_change_from_prior_term: bool = True

    @classmethod
    def all(cls) -> StudentTermAddFeatureSpec:
        return cls()


# No GPA or per-year credit templates; credential-year features come from :mod:`student_term` with guards.
ES_STUDENT_FEATURE_SPEC_DEFAULT = StudentFeatureSpec(
    diff_gpa=False,
    frac_credits_by_year=False,
)


# --- PDP / Edvise instances (raw naming at standardized boundary) ---


## PDP: after cohort validation, ``student_id`` is the standard join key
## (``study_id`` / ``student_guid`` are renamed to ``student_id`` in raw_cohort).
PDP_COHORT_INPUT_COLUMNS = CohortInputColumns(
    institution_id="institution_id",
    student_id="student_id",
    cohort_year_col="cohort",
    cohort_term_col="cohort_term",
    pell_status_col="pell_status_first_year",
    gpa_group_term_1_col="gpa_group_term_1",
    gpa_group_year_1_col="gpa_group_year_1",
    credits_earned_year_template="number_of_credits_earned_year_{yr}",
    credits_attempted_year_template="number_of_credits_attempted_year_{yr}",
    first_year_to_associates_at_cohort_inst="first_year_to_associates_at_cohort_inst",
    years_to_latest_associates_at_cohort_inst="years_to_latest_associates_at_cohort_inst",
    first_year_to_certificate_at_cohort_inst="first_year_to_certificate_at_cohort_inst",
    years_to_latest_certificate_at_cohort_inst="years_to_latest_certificate_at_cohort_inst",
    first_year_to_associates_at_other_inst="first_year_to_associates_at_other_inst",
    years_to_latest_associates_at_other_inst="years_to_latest_associates_at_other_inst",
    first_year_to_certificate_at_other_inst="first_year_to_certificate_at_other_inst",
    years_to_latest_certificate_at_other_inst="years_to_latest_certificate_at_other_inst",
)

## Edvise: mapping per ``RawEdviseStudentDataSchema``; no per-year credit templates, GPA, or
## credential-year fields on raw. Add ``institution_id`` in silver if joins require it. Use
## :data:`ES_STUDENT_FEATURE_SPEC_DEFAULT` with :func:`student.add_features` unless you add
## data and enable flags.
ES_COHORT_INPUT_COLUMNS = CohortInputColumns(
    institution_id=None,
    student_id="learner_id",
    cohort_year_col="entry_year",
    cohort_term_col="entry_term",
    pell_status_col="pell_recipient_year1",
    # gpa, credits_*, credential years: all default None
)

PDP_COURSE_INPUT_COLUMNS = CourseInputColumns(
    institution_id="institution_id",
    student_id="student_id",
    academic_year="academic_year",
    academic_term="academic_term",
    course_prefix="course_prefix",
    course_number="course_number",
    grade="grade",
    course_cip="course_cip",
    number_of_credits_attempted="number_of_credits_attempted",
    number_of_credits_earned="number_of_credits_earned",
    term_program_of_study="term_program_of_study",
    delivery_method="delivery_method",
    math_or_english_gateway="math_or_english_gateway",
    course_instructor_employment_status="course_instructor_employment_status",
    core_course="core_course",
    section_id="section_id",
    course_type="course_type",
    co_requisite_course="co_requisite_course",
    course_instructor_rank="course_instructor_rank",
    enrolled_at_other_institution_s="enrolled_at_other_institution_s",
)

## Edvise: see module doc and :class:`RawEdviseCourseDataSchema`. Unmapped PDP-only
## columns (e.g. student_age on course) are omitted; optional fields are ``None``.
ES_COURSE_INPUT_COLUMNS = CourseInputColumns(
    institution_id=None,
    student_id="learner_id",
    academic_year="academic_year",
    academic_term="academic_term",
    course_prefix="course_prefix",
    course_number="course_number",
    grade="grade",
    course_cip="department",
    number_of_credits_attempted="course_credits_attempted",
    number_of_credits_earned="course_credits_earned",
    term_program_of_study="term_degree",
    delivery_method="instructional_modality",
    math_or_english_gateway="gateway_or_developmental_flag",
    course_instructor_employment_status="instructor_appointment_status",
    core_course="gen_ed_flag",
    section_id="course_section_id",
    course_type=None,
    co_requisite_course=None,
    course_instructor_rank=None,
    enrolled_at_other_institution_s=None,
)


# Backward-compatible aliases (older names referred only to the student sub-profile)
StudentCohortColumns = CohortInputColumns
PDP_STUDENT_COHORT_COLUMNS = PDP_COHORT_INPUT_COLUMNS
ES_STUDENT_COHORT_COLUMNS = ES_COHORT_INPUT_COLUMNS

__all__ = [
    "CumulativeExpandingColumnSpec",
    "CumulativeFeatureSpec",
    "CourseFeatureSpec",
    "CohortInputColumns",
    "CourseInputColumns",
    "ES_STUDENT_FEATURE_SPEC_DEFAULT",
    "ES_COHORT_INPUT_COLUMNS",
    "ES_COURSE_INPUT_COLUMNS",
    "ES_STUDENT_COHORT_COLUMNS",
    "PDP_COHORT_INPUT_COLUMNS",
    "PDP_COURSE_INPUT_COLUMNS",
    "PDP_STUDENT_COHORT_COLUMNS",
    "SectionFeatureSpec",
    "StudentCohortColumns",
    "StudentFeatureSpec",
    "StudentTermAddFeatureSpec",
    "StudentTermAggregateSpec",
    "TermFeatureSpec",
]
