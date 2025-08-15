import pandas as pd
import logging
import functools as ft
import re

LOGGER = logging.getLogger(__name__)

from src.utils._drop_columns_safely import drop_columns_safely

class PDPCleanup:
    def clean_up_labeled_dataset_cols_and_vals(
        self,
        df: pd.DataFrame,
        num_credits_col: str = "num_credits_earned_cumsum",
        num_credit_check: int = 12,
    ) -> pd.DataFrame:
        """
        Drop a bunch of columns not needed or wanted for modeling, and set to null
        any values corresponding to time after a student's current year of enrollment.

        Args:
            df: DataFrame as created with features and targets and limited to the checkpoint term.
            num_credits_col: Name of the column containing cumulative earned credits.
        """
        credit_pattern = re.compile(rf"in_{num_credit_check}_creds")

        for col in df.columns:
            if credit_pattern.search(col):
                df[col] = df[col].mask(df[num_credits_col] < num_credit_check)

        # Drop columns not needed for modeling
        df = drop_columns_safely(df, 
            cols_to_drop=[
                # metadata
                "institution_id",
                "term_id",
                "academic_year",
                # "academic_term",  # keeping this to see if useful
                "cohort",
                # "cohort_term",  # keeping this to see if useful
                "cohort_id",
                "term_rank",
                "min_student_term_rank",
                "term_rank_core",
                "min_student_term_rank_core",
                "term_is_core",
                "term_rank_noncore",
                "min_student_term_rank_noncore",
                "term_is_noncore",
                # columns used to derive other features, but not features themselves
                # "grade",  # TODO: should this be course_grade?
                "course_ids",
                "course_subjects",
                "course_subject_areas",
                "min_student_term_rank",
                "min_student_term_rank_core",
                "min_student_term_rank_noncore",
                "sections_num_students_enrolled",
                "sections_num_students_passed",
                "sections_num_students_completed",
                "term_start_dt",
                "cohort_start_dt",
                "pell_status_first_year",
                # "outcome" variables / likely sources of data leakage
                "retention",
                "persistence",
                # years to bachelors
                "years_to_bachelors_at_cohort_inst",
                "years_to_bachelor_at_other_inst",
                "first_year_to_bachelors_at_cohort_inst",
                "first_year_to_bachelor_at_other_inst",
                # years to associates
                "years_to_latest_associates_at_cohort_inst",
                "years_to_latest_associates_at_other_inst",
                "first_year_to_associates_at_cohort_inst",
                "first_year_to_associates_at_other_inst",
                # years to associates / certificate
                "years_to_associates_or_certificate_at_cohort_inst",
                "years_to_associates_or_certificate_at_other_inst",
                "first_year_to_associates_or_certificate_at_cohort_inst",
                "first_year_to_associates_or_certificate_at_other_inst",
                # years of last enrollment
                "years_of_last_enrollment_at_cohort_institution",
                "years_of_last_enrollment_at_other_institution",
            ]
        )

        # Mask columns
        df = df.assign(
            **{
                col: ft.partial(self.mask_year_values_based_on_enrollment_year, col=col)
                for col in df.columns[df.columns.str.contains(r"^(?:first_year_to_certificate|years_to_latest_certificate)")]
            },
            **{
                col: ft.partial(self.mask_year_column_based_on_enrollment_year, col=col)
                for col in df.columns[df.columns.str.contains(r"_year_\d$")]
            },
            **{
                col: ft.partial(self.mask_term_column_based_on_enrollment_term, col=col)
                for col in df.columns[df.columns.str.contains(r"_term_\d$")]
            },
        )

        return df
    
    def mask_year_values_based_on_enrollment_year(
        self, df: pd.DataFrame, *, col: str, enrollment_year_col: str = "year_of_enrollment_at_cohort_inst"
    ) -> pd.Series:
        return df[col].mask(df[col].ge(df[enrollment_year_col]), other=pd.NA)

    def mask_year_column_based_on_enrollment_year(
        self, df: pd.DataFrame, *, col: str, enrollment_year_col: str = "year_of_enrollment_at_cohort_inst"
    ) -> pd.Series:
        if match := re.search(r"_year_(?P<yr>\d)$", col):
            col_year = int(match.group("yr"))
        else:
            raise ValueError(f"Column '{col}' does not end with '_year_NUM'")
        return df[col].mask(df[enrollment_year_col].le(col_year), other=pd.NA)

    def mask_term_column_based_on_enrollment_term(
        self, df: pd.DataFrame, *, col: str, enrollment_term_col: str = "cumnum_terms_enrolled"
    ) -> pd.Series:
        if match := re.search(r"_term_(?P<num>\d)$", col):
            col_term = int(match.group("num"))
        else:
            raise ValueError(f"Column '{col}' does not end with '_term_NUM'")
        return df[col].mask(df[enrollment_term_col].lt(col_term), other=pd.NA)
