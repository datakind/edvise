"""
Edvise retention input: derive a ``retention`` column on student-term data.

Raw Edvise feeds do not include retention; this module builds it from
``year_of_enrollment_at_cohort_inst`` and ``first_year_to_*_at_cohort_inst`` (see
:func:`assign_retention_column`).

**Encoding**

- ``1`` = **retained**
- ``0`` = **not retained**

:func:`edvise.targets.retention.compute_target` negates after coercion:
``~retention.astype("boolean")``. Pandas maps ``1→True`` and ``0→False``, so retained
rows (``1``) become modeling target ``False`` (not the “non-retention” positive class).
``retention.py`` is unchanged.
"""

from __future__ import annotations

import logging
import typing as t

import pandas as pd

from edvise.utils import types as type_utils

LOGGER = logging.getLogger(__name__)

YEAR_OF_ENROLLMENT_COL = "year_of_enrollment_at_cohort_inst"
_FIRST_YEAR_BACHELORS = "first_year_to_bachelors_at_cohort_inst"
_FIRST_YEAR_ASSOCIATES = "first_year_to_associates_at_cohort_inst"
_FIRST_YEAR_CERT = "first_year_to_certificate_at_cohort_inst"
_CREDENTIAL_COLS: tuple[str, ...] = (
    _FIRST_YEAR_BACHELORS,
    _FIRST_YEAR_ASSOCIATES,
    _FIRST_YEAR_CERT,
)

# Credential completed in academic year 1 or 2 (year index on cohort, values 1–7)
_Y1_Y2: tuple[int, int] = (1, 2)


def assign_retention_column(
    df: pd.DataFrame,
    *,
    student_id_col: str | t.Sequence[str] = "learner_id",
    retention_col: str = "retention",
) -> pd.DataFrame:
    """
    Add an integer ``retention`` column (default name ``retention``), **nullable Int8**:
    ``1`` = retained, ``0`` = not retained (see module doc).

    **Retained** if **either**

    1. max ``year_of_enrollment_at_cohort_inst`` ≥ 2 (enrollment into second academic
       year at cohort institution), **or**
    2. any ``first_year_to_*_at_cohort_inst`` is ``1`` or ``2`` (credential completed
       in first or second academic year at cohort institution).

    ``student_id_col`` should match the project
    (e.g. :attr:`edvise.configs.es.ESProjectConfig.student_id_col`).

    Raises:
        ValueError: if the enrollment column or id columns are missing, or the frame
            is empty.
    """
    if df.empty:
        raise ValueError("assign_retention_column: dataframe is empty.")
    if YEAR_OF_ENROLLMENT_COL not in df.columns:
        raise ValueError(
            f"Missing required column {YEAR_OF_ENROLLMENT_COL!r} "
            "(from student-term features; needs cohort year/term on the cohort data)."
        )

    id_cols = type_utils.to_list(student_id_col)
    for c in id_cols:
        if c not in df.columns:
            raise ValueError(
                f"Missing student id column {c!r}; set project `student_id_col`."
            )

    present_creds = [c for c in _CREDENTIAL_COLS if c in df.columns]
    if not present_creds:
        LOGGER.warning(
            "No first_year_to_*_at_cohort_inst columns; "
            "retention uses enrollment leg only (credential leg always false)."
        )

    # --- Leg 1: continued enrollment into the second academic year (at cohort inst) ---
    enroll_max = (
        df.groupby(id_cols, sort=False)[YEAR_OF_ENROLLMENT_COL].max()
    )
    leg_enroll = enroll_max.ge(2).fillna(False).astype(bool)

    # --- Leg 2: credential completed in academic year 1 or 2 (cohort-inst buckets) ---
    if present_creds:
        parts = [df[c].isin(_Y1_Y2) for c in present_creds]
        row_has = parts[0]
        for p in parts[1:]:
            row_has = row_has | p
        leg_cred = (
            df.assign(_cred=row_has.fillna(False).astype(bool))
            .groupby(id_cols, sort=False)["_cred"]
            .any()
        )
    else:
        leg_cred = pd.Series(False, index=leg_enroll.index, dtype=bool)

    leg_cred = leg_cred.reindex(leg_enroll.index, fill_value=False).astype(
        bool
    )
    # True iff retained by either leg. Unmapped students after merge → not retained (0).
    per_student_retained = leg_enroll | leg_cred
    rdf = per_student_retained.reset_index(name="_retained_bool")
    out = df.merge(
        rdf, on=id_cols, how="left", sort=False, validate="many_to_one"
    )
    # 1 = retained, 0 = not; compute_target applies ~ after astype("boolean").
    out[retention_col] = (
        out["_retained_bool"].fillna(False).astype(int).astype("Int8")
    )
    return out.drop(columns=["_retained_bool"], errors="ignore")
