import pandas as pd

from edvise.data_audit.standardizer import (
    ESCourseStandardizer,
    PDPCohortStandardizer,
)


def test_pdp_cohort_standardizer_keeps_first_term_snapshots() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "enrollment_intensity_first_term": ["FULL-TIME"],
            "attendance_status_term_1": ["First-Time Full-Time"],
            "program_of_study_term_1": ["24.0101"],
            "program_of_study_year_1": ["24.0101"],
        }
    )
    standardized = PDPCohortStandardizer().standardize(df)
    assert "enrollment_intensity_first_term" in standardized.columns
    assert "attendance_status_term_1" in standardized.columns
    assert "program_of_study_term_1" in standardized.columns
    assert "program_of_study_year_1" in standardized.columns


def _es_course_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "learner_id": "s1",
        "course_prefix": "MATH",
        "course_number": "101",
        "grade": "A",
    }
    row.update(overrides)
    return row


def test_es_course_standardizer_keeps_identified_rows() -> None:
    df = pd.DataFrame(
        [
            _es_course_row(learner_id="s1", grade="A"),
            _es_course_row(learner_id="s2", course_number="201", grade="NA"),
        ]
    )
    out = ESCourseStandardizer().standardize(df)
    assert out["learner_id"].tolist() == ["s1", "s2"]
    assert out["grade"].tolist() == ["A", "NA"]


def test_es_course_standardizer_nulls_missing_sentinel_on_other_columns() -> None:
    df = pd.DataFrame(
        [
            _es_course_row(learner_id="s1", course_number="101", grade="A"),
            _es_course_row(learner_id="s2", course_number="102", grade="MISSING"),
            _es_course_row(learner_id="s3", course_number="103", grade=" missing "),
            _es_course_row(learner_id="s4", course_number="104", grade="NA"),
        ]
    )
    out = ESCourseStandardizer().standardize(df)
    assert out["learner_id"].tolist() == ["s1", "s2", "s3", "s4"]
    assert out["grade"].isna().tolist() == [False, True, True, False]
    assert out["grade"].iloc[0] == "A"
    assert out["grade"].iloc[3] == "NA"


def test_es_course_standardizer_drops_rows_missing_course_identifiers() -> None:
    df = pd.DataFrame(
        [
            _es_course_row(learner_id="s1"),
            _es_course_row(learner_id="s2", course_prefix="MISSING"),
            _es_course_row(learner_id="s3", course_number=" missing "),
            _es_course_row(learner_id="s4", course_prefix=None),
        ]
    )
    out = ESCourseStandardizer().standardize(df)
    assert out["learner_id"].tolist() == ["s1"]
