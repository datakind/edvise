import functools as ft
import logging

import pandas as pd

from edvise.utils import types
from . import constants, shared
from .column_names import (
    CohortInputColumns,
    PDP_COHORT_INPUT_COLUMNS,
    StudentFeatureSpec,
)

LOGGER = logging.getLogger(__name__)


def add_features(
    df: pd.DataFrame,
    *,
    first_term_of_year: types.TermType = constants.DEFAULT_FIRST_TERM_OF_YEAR,  # type: ignore
    cols: CohortInputColumns = PDP_COHORT_INPUT_COLUMNS,
    spec: StudentFeatureSpec | None = None,
) -> pd.DataFrame:
    """
    Compute student-level features from a standardized cohort dataset,
    and add as columns to ``df`` .

    Args:
        df: Cohort frame with columns named by ``cols`` (default: PDP).
        first_term_of_year: First term in the academic year (for date bounds).
        cols: Raw cohort column names (:class:`CohortInputColumns`); default PDP.
        spec: Subset of features to compute; default is all. ``cohort_start_dt``
            requires ``cohort_id`` in the same run (it uses the ``cohort_id`` column).
    """
    LOGGER.info("adding student features ...")
    s = spec or StudentFeatureSpec.all()
    if s.cohort_start_dt and not s.cohort_id:
        raise ValueError(
            "cohort_start_dt requires cohort_id=True so column cohort_id exists."
        )
    if s.diff_gpa and (
        cols.gpa_group_term_1_col is None or cols.gpa_group_year_1_col is None
    ):
        raise ValueError(
            "diff_gpa requires CohortInputColumns gpa_group_term_1_col and "
            "gpa_group_year_1_col to be set, or set spec diff_gpa=False."
        )

    assign_kw: dict = {}
    if s.cohort_id:
        assign_kw["cohort_id"] = ft.partial(
            shared.year_term,
            year_col=cols.cohort_year_col,
            term_col=cols.cohort_term_col,
        )
    if s.cohort_start_dt:
        assign_kw["cohort_start_dt"] = ft.partial(
            shared.year_term_dt,
            col="cohort_id",
            bound="start",
            first_term_of_year=first_term_of_year,
        )
    if s.pell:
        assign_kw["student_is_pell_recipient_first_year"] = ft.partial(
            student_is_pell_recipient_first_year,
            pell_col=cols.pell_status_col,
        )
    if s.diff_gpa:
        assert cols.gpa_group_term_1_col is not None and cols.gpa_group_year_1_col is not None
        assign_kw["diff_gpa_term_1_to_year_1"] = ft.partial(
            diff_gpa_term_1_to_year_1,
            term_col=cols.gpa_group_term_1_col,
            year_col=cols.gpa_group_year_1_col,
        )
    if s.frac_credits_by_year:
        if (
            cols.credits_earned_year_template is None
            or cols.credits_attempted_year_template is None
        ):
            raise ValueError(
                "frac_credits_by_year requires credits_earned_year_template and "
                "credits_attempted_year_template, or set spec frac_credits_by_year=False."
            )
        credits_years = [
            yr
            for yr in (1, 2, 3, 4)
            if cols.earned_col(yr) in df.columns
        ]
        assign_kw.update(
            {
                f"frac_credits_earned_year_{yr}": ft.partial(
                    shared.frac_credits_earned,
                    earned_col=cols.earned_col(yr),
                    attempted_col=cols.attempted_col(yr),
                )
                for yr in credits_years
            }
        )

    return df.assign(**assign_kw)


def student_is_pell_recipient_first_year(
    df: pd.DataFrame,
    *,
    pell_col: str = "pell_status_first_year",
) -> pd.Series:
    return df[pell_col].map({"Y": True, "N": False}).astype("boolean")


def diff_gpa_term_1_to_year_1(
    df: pd.DataFrame,
    *,
    term_col: str = "gpa_group_term_1",
    year_col: str = "gpa_group_year_1",
) -> pd.Series:
    return df[year_col].sub(df[term_col])
