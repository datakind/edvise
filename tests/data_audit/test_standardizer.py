import pandas as pd

from edvise.data_audit.standardizer import PDPCohortStandardizer


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
