import pandas as pd

from edvise.model_prep.cleanup_features import ESCleanup


def test_es_cleanup_canonicalizes_and_drops_raw_pell() -> None:
    df = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "target": [1],
            "pell_recipient_year_1": ["Y"],
            "student_is_pell_recipient_first_year": [True],
            "intended_program_type": ["Associate's Degree"],
        }
    )
    cleaned = ESCleanup().clean_up_labeled_dataset_cols_and_vals(df)
    assert "pell_recipient_year_1" not in cleaned.columns
    assert "pell_recipient_year1" not in cleaned.columns
    assert "student_is_pell_recipient_first_year" in cleaned.columns
