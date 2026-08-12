import pandas as pd

from edvise.model_prep.cleanup_features import PDPCleanup


def test_cleanup_drops_redundant_first_term_enrollment_intensity() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "enrollment_intensity_first_term": ["FULL-TIME"],
            "student_term_enrollment_intensity": ["FULL-TIME"],
        }
    )
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(df)
    assert "enrollment_intensity_first_term" not in cleaned.columns
    assert "student_term_enrollment_intensity" in cleaned.columns
