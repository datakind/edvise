import pandas as pd

from edvise.data_audit.standardizer import PDPCohortStandardizer


def test_pdp_cohort_standardizer_keeps_first_term_enrollment_intensity() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "enrollment_intensity_first_term": ["FULL-TIME"],
            "program_of_study_term_1": ["CS"],
        }
    )
    standardized = PDPCohortStandardizer().standardize(df)
    assert "enrollment_intensity_first_term" in standardized.columns
    assert "program_of_study_term_1" not in standardized.columns
