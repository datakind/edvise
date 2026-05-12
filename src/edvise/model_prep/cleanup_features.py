"""Drop non-feature/leakage columns and mask future-relative values for modeling.

Defines a :class:`BaseCleanup` with the shared masking logic, plus product-specific
subclasses (:class:`PDPCleanup`, :class:`ESCleanup`) whose only meaningful
difference is the list of columns to drop.
"""

from __future__ import annotations

import functools as ft
import logging
import re
import typing as t

import pandas as pd

from edvise.utils.drop_columns_safely import drop_columns_safely

LOGGER = logging.getLogger(__name__)


class BaseCleanup:
    """Shared cleanup logic for labeled / unlabeled student-term datasets.

    Subclasses set :attr:`cols_to_drop` with metadata, derivation-source, and
    leakage columns specific to a product (PDP vs Edvise ES). Masking is
    pattern-based and shared.
    """

    cols_to_drop: t.ClassVar[list[str]] = []

    def clean_up_labeled_dataset_cols_and_vals(
        self,
        df: pd.DataFrame,
        num_credits_col: str = "cumsum_num_credits_earned",
        num_credit_check: int = 12,
    ) -> pd.DataFrame:
        """
        Drop columns in :attr:`cols_to_drop` and null out values corresponding
        to time after a student's current year (or term) of enrollment.

        Args:
            df: DataFrame with features and (optionally) targets, limited to the
                checkpoint term.
            num_credits_col: Column with cumulative earned credits, used to mask
                ``in_{num_credit_check}_creds`` features.
            num_credit_check: Credit threshold for masking ``in_N_creds`` features.
        """
        if num_credits_col in df.columns:
            credit_pattern = re.compile(rf"in_{num_credit_check}_creds")
            for col in df.columns:
                if credit_pattern.search(col):
                    df[col] = df[col].mask(df[num_credits_col] < num_credit_check)

        df = drop_columns_safely(df, cols_to_drop=self.cols_to_drop)

        df = df.assign(
            **{
                col: ft.partial(self.mask_year_values_based_on_enrollment_year, col=col)
                for col in df.columns[
                    df.columns.str.contains(
                        r"^(?:first_year_to_certificate|years_to_latest_certificate)"
                    )
                ]
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
        self,
        df: pd.DataFrame,
        *,
        col: str,
        enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
    ) -> pd.Series:
        return df[col].mask(df[col].ge(df[enrollment_year_col]), other=pd.NA)

    def mask_year_column_based_on_enrollment_year(
        self,
        df: pd.DataFrame,
        *,
        col: str,
        enrollment_year_col: str = "year_of_enrollment_at_cohort_inst",
    ) -> pd.Series:
        if match := re.search(r"_year_(?P<yr>\d)$", col):
            col_year = int(match.group("yr"))
        else:
            raise ValueError(f"Column '{col}' does not end with '_year_NUM'")
        return df[col].mask(df[enrollment_year_col].le(col_year), other=pd.NA)

    def mask_term_column_based_on_enrollment_term(
        self,
        df: pd.DataFrame,
        *,
        col: str,
        enrollment_term_col: str = "cumnum_terms_enrolled",
    ) -> pd.Series:
        if match := re.search(r"_term_(?P<num>\d)$", col):
            col_term = int(match.group("num"))
        else:
            raise ValueError(f"Column '{col}' does not end with '_term_NUM'")
        return df[col].mask(df[enrollment_term_col].lt(col_term), other=pd.NA)


class PDPCleanup(BaseCleanup):
    """Cleanup for PDP labeled / unlabeled datasets."""

    cols_to_drop: t.ClassVar[list[str]] = [
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


class ESCleanup(BaseCleanup):
    """Cleanup for Edvise ES labeled / unlabeled datasets.

    Drops Edvise raw cohort metadata (entry/conferral/certificate dates) and the
    PDP-aligned credential-year columns derived from those dates, since they
    feed :func:`edvise.targets.retention_edvise.assign_retention_column` and
    would leak the retention target. Mirrors :class:`PDPCleanup` for metadata
    (term ranks, course id/subject helpers, section counts, etc.).
    """

    cols_to_drop: t.ClassVar[list[str]] = [
        # metadata
        "institution_id",
        "term_id",
        "academic_year",
        # "academic_term",  # keeping this for parity with PDPCleanup
        # Edvise cohort identifiers (analog of PDP "cohort" / "cohort_term")
        "entry_year",
        # "entry_term",  # keeping this for parity with PDPCleanup
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
        "course_ids",
        "course_subjects",
        "course_subject_areas",
        "sections_num_students_enrolled",
        "sections_num_students_passed",
        "sections_num_students_completed",
        "term_start_dt",
        "cohort_start_dt",
        # Edvise raw pell column (replaced by derived "pell" feature)
        "pell_recipient_year1",
        # Edvise raw cohort dates feeding credential-year derivation
        "matriculation_date",
        "bachelors_degree_conferral_date",
        "associates_degree_conferral_date",
        "certificate1_date",
        "certificate2_date",
        "certificate3_date",
        "conferred_credential_type",
        "major_at_completion",
        # "outcome" variables / likely sources of data leakage
        "retention",
        "persistence",
        # Credential-year columns derived in :mod:`edvise.data_audit.es_cohort_credential_years`.
        # Used in :func:`assign_retention_column` to compute the retention target,
        # so they leak the label.
        "first_year_to_bachelors_at_cohort_inst",
        "first_year_to_associates_at_cohort_inst",
        "years_to_latest_associates_at_cohort_inst",
    ]
