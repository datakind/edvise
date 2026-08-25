import pandas as pd
import pytest

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


@pytest.mark.parametrize(
    ("term_cols", "drop_entry", "keep_entry"),
    [
        (
            {"term_degree": ["Associate"]},
            ["intended_program_type"],
            ["declared_major_at_entry", "term_degree"],
        ),
        (
            {"term_declared_major": ["Biology"]},
            ["declared_major_at_entry"],
            ["intended_program_type", "term_declared_major"],
        ),
        (
            {"term_program_of_study": ["24.0101"]},
            ["declared_major_at_entry"],
            ["intended_program_type", "term_program_of_study"],
        ),
        (
            {},
            [],
            ["intended_program_type", "declared_major_at_entry"],
        ),
    ],
)
def test_es_cleanup_drops_entry_cols_only_when_term_counterparts_present(
    term_cols: dict[str, list[str]],
    drop_entry: list[str],
    keep_entry: list[str],
) -> None:
    df = pd.DataFrame(
        {
            "learner_id": ["s1"],
            "intended_program_type": ["Associate's Degree"],
            "declared_major_at_entry": ["Biology"],
            **term_cols,
        }
    )
    cleaned = ESCleanup().clean_up_labeled_dataset_cols_and_vals(df)
    for col in drop_entry:
        assert col not in cleaned.columns
    for col in keep_entry:
        assert col in cleaned.columns
