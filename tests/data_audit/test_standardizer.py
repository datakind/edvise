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


def test_es_course_standardizer_nulls_missing_sentinel() -> None:
    df = pd.DataFrame({"grade": ["A", "MISSING", " missing ", "NA"]})
    out = ESCourseStandardizer().standardize(df)
    assert out["grade"].isna().tolist() == [False, True, True, False]
    assert out["grade"].iloc[0] == "A"
    assert out["grade"].iloc[3] == "NA"
