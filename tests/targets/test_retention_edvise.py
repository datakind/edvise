"""Tests for :mod:`edvise.targets.retention_edvise`."""

import pandas as pd
import pytest

from edvise.targets import retention_edvise as r

LEARNER = "learner_id"
Y = r.YEAR_OF_ENROLLMENT_COL
B, A, C = r._FIRST_YEAR_BACHELORS, r._FIRST_YEAR_ASSOCIATES, r._FIRST_YEAR_CERT


def _base_st(id_val: str, n_terms: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            LEARNER: [id_val] * n_terms,
            "term_rank": list(range(1, n_terms + 1)),
        }
    )


def test_assign_requires_year_of_enrollment() -> None:
    df = pd.DataFrame({LEARNER: ["a"]})
    with pytest.raises(ValueError, match="year_of_enrollment_at_cohort_inst"):
        r.assign_retention_column(df, student_id_col=LEARNER)


def test_assign_requires_non_empty() -> None:
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty"):
        r.assign_retention_column(df, student_id_col=LEARNER)


def test_enrollment_only_year_two_true() -> None:
    df = _base_st("s1", 2)
    df[Y] = [1, 2]
    for c in (B, A, C):
        df[c] = pd.NA
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert (out["retention"] == 1).all()


def test_enrollment_year_one_only_false() -> None:
    df = _base_st("s1", 2)
    df[Y] = [1, 1]
    for c in (B, A, C):
        df[c] = pd.NA
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert (out["retention"] == 0).all()


def test_credential_year_one_without_enrollment_year_two() -> None:
    df = _base_st("s1", 2)
    df[Y] = [1, 1]
    df[B] = [1, 1]
    df[A] = [pd.NA, pd.NA]
    df[C] = [pd.NA, pd.NA]
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert (out["retention"] == 1).all()


def test_credential_year_three_only_false() -> None:
    df = _base_st("s1", 2)
    df[Y] = [1, 1]
    df[B] = [3, 3]
    df[A] = [pd.NA, pd.NA]
    df[C] = [pd.NA, pd.NA]
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert (out["retention"] == 0).all()


def test_two_students_mixed() -> None:
    df = pd.DataFrame(
        {
            LEARNER: ["a", "a", "b", "b"],
            "term_rank": [1, 2, 1, 2],
            Y: [1, 2, 1, 1],
            B: [pd.NA, pd.NA, pd.NA, pd.NA],
            A: [pd.NA, pd.NA, pd.NA, pd.NA],
            C: [pd.NA, pd.NA, pd.NA, pd.NA],
        }
    )
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert out["retention"].tolist() == [1, 1, 0, 0]


def test_no_credential_columns_enrollment_only() -> None:
    df = _base_st("s1", 2)
    df[Y] = [1, 2]
    out = r.assign_retention_column(df, student_id_col=LEARNER)
    assert out["retention"].tolist() == [1, 1]


def test_retention_compute_target_with_int_retained_column() -> None:
    """``1`` = retained, ``0`` = not; unchanged :func:`retention.compute_target`."""
    from edvise.targets import retention as rt

    base = {
        "learner_id": ["x"],
        "cohort_id": ["2020 FALL"],
        "term_id": ["2021-22 FALL"],
    }
    s_ret = rt.compute_target(
        pd.DataFrame({**base, "retention": [1]}).astype({"retention": "Int8"}),
        max_academic_year="2025",
        student_id_cols="learner_id",
    )
    s_not = rt.compute_target(
        pd.DataFrame({**base, "retention": [0]}).astype({"retention": "Int8"}),
        max_academic_year="2025",
        student_id_cols="learner_id",
    )
    assert not bool(s_ret.iloc[0])
    assert bool(s_not.iloc[0])
