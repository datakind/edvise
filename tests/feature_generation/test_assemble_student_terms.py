import pandas as pd

from edvise.data_audit.schemas._edvise_shared import canonicalize_edvise_cohort_column_names
from edvise.utils.data_cleaning import convert_to_snake_case


def test_canonicalize_after_snake_case_restores_pell_recipient_year1() -> None:
    """Feature assembly re-snake-cases columns; canonicalize must undo year1 mangling."""
    df = pd.DataFrame(
        {
            "pell_recipient_year1": ["Y"],
            "student_is_pell_recipient_first_year": [True],
        }
    )
    df = df.rename(columns=convert_to_snake_case)
    assert "pell_recipient_year_1" in df.columns

    df = canonicalize_edvise_cohort_column_names(df)
    assert "pell_recipient_year1" in df.columns
    assert "pell_recipient_year_1" not in df.columns
    assert "student_is_pell_recipient_first_year" in df.columns
