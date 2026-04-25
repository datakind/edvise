from __future__ import annotations

from dataclasses import dataclass


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
    "CohortInputColumns",
    "CourseInputColumns",
    "ES_STUDENT_FEATURE_SPEC_DEFAULT",
    "ES_COHORT_INPUT_COLUMNS",
    "ES_COURSE_INPUT_COLUMNS",
    "ES_STUDENT_COHORT_COLUMNS",
    "PDP_COHORT_INPUT_COLUMNS",
    "PDP_COURSE_INPUT_COLUMNS",
    "PDP_STUDENT_COHORT_COLUMNS",
    "StudentCohortColumns",
    "StudentFeatureSpec",
]
