import pandas as pd

from edvise.model_prep.cleanup_features import PDPCleanup


def test_pdp_cleanup_drops_term1_gpa_keeps_year_change_features() -> None:
    df = pd.DataFrame(
        {
            "student_id": ["s1"],
            "year_of_enrollment_at_cohort_inst": [2],
            "cumnum_terms_enrolled": [2],
            "gpa_group_term_1": [3.0],
            "gpa_group_year_1": [3.2],
            "diff_gpa_term_1_to_year_1": [0.2],
            "frac_credits_earned_year_1": [0.8],
            "num_courses_diff_term_1_to_term_2": [1],
            "course_grade_numeric_mean": [3.1],
        }
    )
    cleaned = PDPCleanup().clean_up_labeled_dataset_cols_and_vals(df)
    assert "gpa_group_term_1" not in cleaned.columns
    assert "gpa_group_year_1" in cleaned.columns
    assert "diff_gpa_term_1_to_year_1" in cleaned.columns
    assert "frac_credits_earned_year_1" in cleaned.columns
    assert "num_courses_diff_term_1_to_term_2" in cleaned.columns
    assert "course_grade_numeric_mean" in cleaned.columns
