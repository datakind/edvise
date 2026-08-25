import pandas as pd
import pytest

from edvise.model_prep.cleanup_features import PDPCleanup


@pytest.mark.parametrize(
    ("term_cols", "drop_entry", "keep_entry"),
    [
        (
            {"course_grade_numeric_mean": [3.2]},
            ["gpa_group_term_1", "gpa_group_year_1"],
            ["course_grade_numeric_mean"],
        ),
        (
            {"cummean_course_grade_numeric_mean": [3.1]},
            ["gpa_group_year_1"],
            ["gpa_group_term_1", "cummean_course_grade_numeric_mean"],
        ),
        (
            {"cumsum_num_credits_earned": [12.0], "num_credits_attempted": [15.0]},
            ["number_of_credits_earned_year_1", "number_of_credits_attempted_year_1"],
            ["cumsum_num_credits_earned", "frac_credits_earned_year_1"],
        ),
        (
            {"frac_credits_earned": [0.8]},
            ["frac_credits_earned_year_1"],
            ["gpa_group_term_1", "frac_credits_earned"],
        ),
        (
            {"num_courses_diff_term_1_to_term_2": [1]},
            [],
            [
                "gpa_group_term_1",
                "gpa_group_year_1",
                "num_courses_diff_term_1_to_term_2",
            ],
        ),
    ],
)
def test_pdp_cleanup_drops_term1_year1_cols_only_when_term_counterparts_present(
    term_cols: dict[str, list[object]],
    drop_entry: list[str],
    keep_entry: list[str],
) -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "year_of_enrollment_at_cohort_inst": [2],
            "cumnum_terms_enrolled": [2],
            "gpa_group_term_1": [3.0],
            "gpa_group_year_1": [3.2],
            "number_of_credits_earned_year_1": [12.0],
            "number_of_credits_attempted_year_1": [15.0],
            "frac_credits_earned_year_1": [0.8],
            **term_cols,
        }
    )
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(df)
    for col in drop_entry:
        assert col not in cleaned.columns
    for col in keep_entry:
        assert col in cleaned.columns
