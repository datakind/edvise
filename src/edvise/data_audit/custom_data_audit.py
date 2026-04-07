"""Structured custom-school data audit: validation and column inference.

These helpers return reports, anomaly frames, or inferred column names without applying
pipeline cleaning transforms. Exploratory logging-only utilities live in
:mod:`edvise.data_audit.eda`; DataFrame transforms live in
:mod:`edvise.data_audit.custom_cleaning`.
"""

import logging
import re
import typing as t

import numpy as np
import pandas as pd

from edvise.shared.utils import percent_of_rows

LOGGER = logging.getLogger(__name__)


class EarnedAttemptedCheckResult(t.TypedDict):
    """Structured return from :func:`check_earned_vs_attempted`."""

    anomalies: pd.DataFrame
    summary: pd.DataFrame


def validate_ids_terms_consistency(
    student_df: t.Optional[pd.DataFrame],
    semester_df: pd.DataFrame,
    course_df: pd.DataFrame,
    *,
    id_col: str = "student_id",
    sem_col: str = "semester_code",
    student_id_col: t.Optional[str] = None,
) -> t.Dict[str, t.Any]:
    """Check (student id, term) keys and bare student ids across course, semester, and student tables.

    Args:
        student_df: Optional student-level table (one row per student). When omitted,
            student-id coverage checks return empty frames.
        semester_df: Semester-level grain; must contain ``id_col`` and ``sem_col``.
        course_df: Course-level grain; must contain ``id_col`` and ``sem_col``.
        id_col: Student identifier column name on course and semester frames.
        sem_col: Term / semester code column name on course and semester frames.
        student_id_col: Column name on ``student_df`` if it differs from ``id_col``;
            defaults to ``id_col``.

    Returns:
        Dict with:

        - ``summary``: int counts (unmatched keys, ids missing from peer tables, null keys, etc.).
        - ``unmatched_course_side`` / ``unmatched_semester_side``: (id, term) keys present on
          one file but not the other.
        - ``course_ids_not_in_semester``, ``course_ids_not_in_student``,
          ``semester_ids_not_in_student``: id-only anti-joins.
        - ``course_terms_not_in_semester_terms``: term values appearing in courses but not semesters.
        - ``null_course_keys`` / ``null_semester_keys``: rows with null id or term.

    Note:
        Does not mutate input frames; copies key columns where needed.
    """
    student_id_col = student_id_col or id_col
    c_keys = course_df[[id_col, sem_col]].copy()
    s_keys = semester_df[[id_col, sem_col]].copy()

    st = None
    if student_df is not None:
        st = student_df[[student_id_col]].drop_duplicates().copy()
        if student_id_col != id_col:
            st = st.rename(columns={student_id_col: id_col})

    null_course_keys = course_df.loc[
        course_df[id_col].isna() | course_df[sem_col].isna(),
        [id_col, sem_col],
    ].drop_duplicates()

    null_semester_keys = semester_df.loc[
        semester_df[id_col].isna() | semester_df[sem_col].isna(),
        [id_col, sem_col],
    ].drop_duplicates()

    course_keys = c_keys.drop_duplicates()
    sem_keys = s_keys.drop_duplicates()

    unmatched_course_side = (
        course_keys.merge(sem_keys, on=[id_col, sem_col], how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col, sem_col]]
        .sort_values([id_col, sem_col])
        .reset_index(drop=True)
    )

    unmatched_semester_side = (
        sem_keys.merge(course_keys, on=[id_col, sem_col], how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col, sem_col]]
        .sort_values([id_col, sem_col])
        .reset_index(drop=True)
    )

    course_ids = course_df[[id_col]].dropna().drop_duplicates()
    semester_ids = semester_df[[id_col]].dropna().drop_duplicates()

    course_ids_not_in_semester = (
        course_ids.merge(semester_ids, on=id_col, how="left", indicator=True)
        .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
        .sort_values(id_col)
        .reset_index(drop=True)
    )

    if st is not None:
        course_ids_not_in_student = (
            course_ids.merge(st, on=id_col, how="left", indicator=True)
            .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
            .sort_values(id_col)
            .reset_index(drop=True)
        )
        semester_ids_not_in_student = (
            semester_ids.merge(st, on=id_col, how="left", indicator=True)
            .loc[lambda df: df["_merge"].eq("left_only"), [id_col]]
            .sort_values(id_col)
            .reset_index(drop=True)
        )
    else:
        course_ids_not_in_student = pd.DataFrame(columns=[id_col])
        semester_ids_not_in_student = pd.DataFrame(columns=[id_col])

    course_terms = set(course_df[sem_col].dropna().unique())
    semester_terms = set(semester_df[sem_col].dropna().unique())
    course_terms_missing = sorted(course_terms - semester_terms)
    course_terms_not_in_semester_terms = pd.DataFrame({sem_col: course_terms_missing})

    summary = {
        "total_semesters_in_semester_file": int(len(semester_df)),
        "unique_student_semesters_in_courses": int(len(course_keys)),
        "unmatched_course_keys": int(len(unmatched_course_side)),
        "unmatched_semester_keys": int(len(unmatched_semester_side)),
        "course_ids_not_in_semester": int(len(course_ids_not_in_semester)),
        "course_ids_not_in_student": int(len(course_ids_not_in_student)),
        "semester_ids_not_in_student": int(len(semester_ids_not_in_student)),
        "course_terms_not_in_semester_terms": int(
            len(course_terms_not_in_semester_terms)
        ),
        "course_rows_with_null_keys": int(len(null_course_keys)),
        "semester_rows_with_null_keys": int(len(null_semester_keys)),
    }

    return {
        "summary": summary,
        "unmatched_course_side": unmatched_course_side,
        "unmatched_semester_side": unmatched_semester_side,
        "course_ids_not_in_semester": course_ids_not_in_semester,
        "course_ids_not_in_student": course_ids_not_in_student,
        "semester_ids_not_in_student": semester_ids_not_in_student,
        "course_terms_not_in_semester_terms": course_terms_not_in_semester_terms,
        "null_course_keys": null_course_keys,
        "null_semester_keys": null_semester_keys,
    }


def _duplicate_key_conflict_metrics(
    df: pd.DataFrame, primary_keys: list[str]
) -> pd.DataFrame:
    """Share of duplicate-key groups (percent) where each column takes multiple values."""
    empty = pd.DataFrame(columns=["column", "pct_conflicting_groups"])
    dup = df[df.duplicated(subset=primary_keys, keep=False)]
    if dup.empty:
        return empty
    grp = dup.groupby(primary_keys, dropna=False)
    conflict = grp.nunique(dropna=False) > 1
    conflict = conflict[conflict.any(axis=1)]
    if conflict.empty:
        return empty
    return (
        conflict.mean()
        .mul(100)
        .rename("pct_conflicting_groups")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("pct_conflicting_groups", ascending=False)
        .reset_index(drop=True)
    )


def find_dupes(df: pd.DataFrame, primary_keys: list[str]) -> pd.DataFrame:
    """Print duplicate-row stats and column-level conflict rates; return all duplicate-key rows.

    Args:
        df: Table to scan (not mutated).
        primary_keys: Columns defining a row identity; duplicates are rows sharing the same
            key values (``keep=False`` semantics — all rows in a duplicate group are kept).

    Returns:
        Copy of rows that participate in any duplicate key group. If ``student_id`` exists,
        sorted by that column for stable notebook output.

    Note:
        Side effect: prints human-readable summary to stdout (intentional for notebooks).
    """
    dupes = df[df.duplicated(subset=primary_keys, keep=False)].copy()

    if "student_id" in dupes.columns:
        dupes = dupes.sort_values("student_id", ignore_index=True)

    total_rows = len(df)
    dupe_rows = len(dupes)
    pct_dupes = percent_of_rows(dupe_rows, total_rows)

    print(
        f"{dupe_rows} duplicate rows based on {primary_keys} "
        f"({pct_dupes:.2f}% of {total_rows} total rows)"
    )

    print(_duplicate_key_conflict_metrics(df, primary_keys))
    return dupes


def check_earned_vs_attempted(
    df: pd.DataFrame,
    *,
    earned_col: str,
    attempted_col: str,
) -> EarnedAttemptedCheckResult:
    """Flag course (or similar) rows where earned vs attempted credits are inconsistent.

    Row-wise rules (after numeric coercion):

    - Earned credits must not exceed attempted credits.
    - If attempted credits are zero, earned credits must be zero.

    Args:
        df: Input table; must contain ``earned_col`` and ``attempted_col``.
        earned_col: Column for credits earned (coerced with ``errors='coerce'``).
        attempted_col: Column for credits attempted (coerced with ``errors='coerce'``).

    Returns:
        ``anomalies``: rows failing either rule, with boolean flags
        ``earned_gt_attempted`` and ``earned_when_no_attempt``;
        ``summary``: one-row frame with counts and ``percent_of_rows`` for each violation type.
    """
    earned = pd.to_numeric(df[earned_col], errors="coerce")
    attempted = pd.to_numeric(df[attempted_col], errors="coerce")

    earned_gt_attempted = earned > attempted
    earned_when_no_attempt = (attempted == 0) & (earned > 0)
    mask = earned_gt_attempted | earned_when_no_attempt

    anomalies = df[mask].copy()
    anomalies["earned_gt_attempted"] = earned_gt_attempted[mask]
    anomalies["earned_when_no_attempt"] = earned_when_no_attempt[mask]

    total_rows = len(df)
    n_earned_gt = int(earned_gt_attempted.sum())
    n_earned_no_attempt = int(earned_when_no_attempt.sum())
    n_total = int(mask.sum())
    summary = pd.DataFrame(
        {
            "earned_gt_attempted": [n_earned_gt],
            "earned_gt_attempted_pct": [percent_of_rows(n_earned_gt, total_rows)],
            "earned_when_no_attempt": [n_earned_no_attempt],
            "earned_when_no_attempt_pct": [
                percent_of_rows(n_earned_no_attempt, total_rows)
            ],
            "total_anomalous_rows": [n_total],
            "total_anomalous_rows_pct": [percent_of_rows(n_total, total_rows)],
        }
    )

    return {"anomalies": anomalies, "summary": summary}


def _credit_reconciliation_mismatch_mask(
    merged: pd.DataFrame,
    *,
    sem_has_attempted: bool,
    sem_has_earned: bool,
    match_attempted_col: str = "match_attempted",
    match_earned_col: str = "match_earned",
) -> pd.Series:
    """True where semester vs course credit reconciliation disagrees (missing match counts as mismatch)."""
    mismatch_mask = pd.Series(False, index=merged.index)
    if sem_has_attempted and match_attempted_col in merged.columns:
        mismatch_mask |= ~merged[match_attempted_col].fillna(True)
    if sem_has_earned and match_earned_col in merged.columns:
        mismatch_mask |= ~merged[match_earned_col].fillna(True)
    return mismatch_mask


def log_semester_reconciliation_summary(
    *,
    logger: logging.Logger,
    merged: pd.DataFrame,
    agg: pd.DataFrame,
    s: pd.DataFrame,
    id_col: str,
    sem_col: str,
    sem_has_attempted: bool,
    sem_has_earned: bool,
    diff_attempted_col: str = "diff_attempted",
    match_attempted_col: str = "match_attempted",
    diff_earned_col: str = "diff_earned",
    match_earned_col: str = "match_earned",
) -> None:
    """Emit WARNING logs summarizing semester vs aggregated course credit reconciliation.

    Args:
        logger: Target logger for detailed reconciliation lines (school or pipeline logger).
        merged: Semester rows left-joined to course aggregates; expected columns include
            ``has_course_rows``, optional ``diff_*`` / ``match_*`` pairs for attempted and earned.
        agg: Course-side aggregate frame (unused here but kept for call-site symmetry).
        s: Semester slice that was merged (defines row count for percentages).
        id_col: Student id column name.
        sem_col: Term column name.
        sem_has_attempted: When True, log attempted-credit diff statistics.
        sem_has_earned: When True, log earned-credit diff statistics.
        diff_attempted_col / match_attempted_col: Column names on ``merged`` for attempted
            credit reconciliation (difference and boolean match flag).
        diff_earned_col / match_earned_col: Same for earned credits.

    Note:
        Also logs one line at module ``LOGGER`` scope for overall reconciliation row count.
    """
    total_sem_rows = int(len(s))
    mismatch_mask = _credit_reconciliation_mismatch_mask(
        merged,
        sem_has_attempted=sem_has_attempted,
        sem_has_earned=sem_has_earned,
        match_attempted_col=match_attempted_col,
        match_earned_col=match_earned_col,
    )

    mismatch_rows = int(mismatch_mask.sum())
    mismatch_pct = percent_of_rows(mismatch_rows, total_sem_rows)

    no_course_rows = (
        int((~merged["has_course_rows"]).sum())
        if "has_course_rows" in merged.columns
        else 0
    )

    logger.warning(
        "Semester reconciliation: rows=%d, mismatches=%d (%.1f%%), semester_rows_without_course_rows=%d",
        total_sem_rows,
        mismatch_rows,
        mismatch_pct,
        no_course_rows,
    )

    LOGGER.warning(
        "Semester reconciliation scope: %d student-semester rows compared; raw key coverage verified prior to aggregation",
        total_sem_rows,
    )

    def _log_credit_diff(label: str, diff_col: str, match_col: str) -> None:
        if diff_col not in merged.columns or match_col not in merged.columns:
            return

        mism = int((~merged[match_col].fillna(True)).sum())
        neg = int((merged[diff_col] < 0).sum())
        pos = int((merged[diff_col] > 0).sum())

        logger.warning(
            " - %s: mismatches=%d (%.1f%%); direction sem>courses=%d, courses>sem=%d; abs_diff median=%.1f, p90=%.1f, max=%.1f",
            label,
            mism,
            percent_of_rows(mism, total_sem_rows),
            neg,
            pos,
            float(merged[diff_col].abs().median()),
            float(merged[diff_col].abs().quantile(0.90)),
            float(merged[diff_col].abs().max()),
        )

    if sem_has_attempted:
        _log_credit_diff("Attempted credits", diff_attempted_col, match_attempted_col)

    if sem_has_earned:
        _log_credit_diff("Earned credits", diff_earned_col, match_earned_col)


def validate_credit_consistency(
    course_df: pd.DataFrame,
    semester_df: t.Optional[pd.DataFrame] = None,
    cohort_df: t.Optional[pd.DataFrame] = None,
    *,
    id_col: str = "student_id",
    sem_col: str = "semester",
    course_credits_attempted_col: t.Optional[str] = "credits_attempted",
    course_credits_earned_col: t.Optional[str] = "credits_earned",
    semester_credits_attempted_col: t.Optional[
        str
    ] = "number_of_semester_credits_attempted",
    semester_credits_earned_col: t.Optional[str] = "number_of_semester_credits_earned",
    semester_courses_count_col: t.Optional[str] = "number_of_semester_courses_enrolled",
    cohort_credits_attempted_col: t.Optional[str] = "inst_tot_credits_attempted",
    cohort_credits_earned_col: t.Optional[str] = "inst_tot_credits_earned",
    credit_tol: float = 0.0,
    strict_columns: bool = False,
) -> t.Dict[str, t.Any]:
    """Cross-check course-, semester-, and optional cohort-level credit totals and row rules.

    Aggregates course credits by ``(id_col, sem_col)``, compares to semester file when
    provided, and optionally compares cohort institutional totals to sums of semester credits.
    Also surfaces row-level anomalies (earned vs attempted) when the relevant columns exist.

    Args:
        course_df: Course-grain data (required).
        semester_df: Optional semester-grain file for reconciliation.
        cohort_df: Optional cohort-grain file for institutional total checks.
        id_col: Student identifier column (default ``student_id``).
        sem_col: Term column on course and semester frames (default ``semester``).
        course_credits_attempted_col / course_credits_earned_col: Course-row credit columns;
            when missing, non-strict mode may fall back to common alternate names on the frame.
        semester_credits_attempted_col / semester_credits_earned_col / semester_courses_count_col:
            Semester file columns for attempted, earned, and course count (defaults match
            typical SST naming).
        cohort_credits_attempted_col / cohort_credits_earned_col: Cohort columns for
            institution-level attempted/earned totals.
        credit_tol: Absolute tolerance when comparing summed vs reported credits.
        strict_columns: If True, only use a credit column name when it is non-empty **and**
            present on the frame — no alternate-name fallbacks (for notebooks that pass
            inferred names only).

    Returns:
        Dict of audit artifacts (summary frames, mismatch masks, reconciliation tables, etc.).
        Keys are documented in the institutional report helper and notebook templates.

    Note:
        Logging is verbose (INFO/WARNING) for notebook visibility; does not raise on mismatch.
    """
    LOGGER.info(
        "Starting credit consistency validation "
        "(course_df=%d rows, semester_df=%s, cohort_df=%s)",
        len(course_df),
        "provided" if semester_df is not None else "None",
        "provided" if cohort_df is not None else "None",
    )

    # -------------------------------------------------------
    # Resolve course credit column names
    # -------------------------------------------------------
    if strict_columns:
        resolved_attempted = (
            course_credits_attempted_col
            if course_credits_attempted_col
            and course_credits_attempted_col in course_df.columns
            else None
        )
        resolved_earned = (
            course_credits_earned_col
            if course_credits_earned_col
            and course_credits_earned_col in course_df.columns
            else None
        )
    else:
        resolved_attempted = (
            course_credits_attempted_col
            if course_credits_attempted_col
            and course_credits_attempted_col in course_df.columns
            else "course_credits_attempted"
            if "course_credits_attempted" in course_df.columns
            else None
        )

        resolved_earned = (
            course_credits_earned_col
            if course_credits_earned_col
            and course_credits_earned_col in course_df.columns
            else "course_credits_earned"
            if "course_credits_earned" in course_df.columns
            else None
        )

    has_course_credit_cols = (
        resolved_attempted is not None and resolved_earned is not None
    )

    # =======================================================
    # A) COURSE-LEVEL CHECKS
    # =======================================================
    course_anomalies = None
    course_anomalies_summary = None

    if has_course_credit_cols:
        LOGGER.info("Running course-level earned <= attempted checks")

        cchk = course_df[
            [c for c in [id_col, sem_col] if c in course_df.columns]
            + [resolved_attempted, resolved_earned]
        ].copy()

        cchk[resolved_attempted] = pd.to_numeric(
            cchk[resolved_attempted], errors="coerce"
        )
        cchk[resolved_earned] = pd.to_numeric(cchk[resolved_earned], errors="coerce")

        cchk["diff"] = cchk[resolved_earned] - cchk[resolved_attempted]
        cchk["earned_exceeds_attempted"] = cchk["diff"] > credit_tol
        cchk["attempted_negative"] = cchk[resolved_attempted] < 0
        cchk["earned_negative"] = cchk[resolved_earned] < 0

        course_anomalies = cchk.loc[
            cchk["earned_exceeds_attempted"]
            | cchk["attempted_negative"]
            | cchk["earned_negative"]
        ]

        total_course_rows = len(cchk)
        n_anomalies = int(len(course_anomalies))
        course_anomalies_summary = {
            "rows_checked": total_course_rows,
            "rows_with_anomalies": n_anomalies,
            "pct_of_data": percent_of_rows(n_anomalies, total_course_rows),
        }

        if len(course_anomalies) > 0:
            LOGGER.warning(
                "Detected %d course-level anomalies (%.2f%% of course data)",
                len(course_anomalies),
                course_anomalies_summary["pct_of_data"],
            )
        else:
            LOGGER.info("No course-level credit anomalies detected")

    # =======================================================
    # B) SEMESTER RECONCILIATION
    # =======================================================
    mismatches = None
    merged = None
    reconciliation_summary = None

    sem_has_attempted = False
    sem_has_earned = False
    sem_has_count = False
    if semester_df is not None:
        sem_has_attempted = (
            bool(semester_credits_attempted_col)
            and semester_credits_attempted_col in semester_df.columns
        )
        sem_has_earned = (
            bool(semester_credits_earned_col)
            and semester_credits_earned_col in semester_df.columns
        )
        sem_has_count = (
            bool(semester_courses_count_col)
            and semester_courses_count_col in semester_df.columns
        )

    if (
        semester_df is not None
        and has_course_credit_cols
        and id_col in course_df.columns
        and sem_col in course_df.columns
        and id_col in semester_df.columns
        and sem_col in semester_df.columns
        and (sem_has_attempted or sem_has_earned)
    ):
        LOGGER.info("Reconciling semester aggregates with course data")

        c = course_df[[id_col, sem_col, resolved_attempted, resolved_earned]].copy()
        c[resolved_attempted] = pd.to_numeric(c[resolved_attempted], errors="coerce")
        c[resolved_earned] = pd.to_numeric(c[resolved_earned], errors="coerce")

        s_cols = [id_col, sem_col]
        if sem_has_attempted:
            assert semester_credits_attempted_col is not None
            s_cols.append(semester_credits_attempted_col)
        if sem_has_earned:
            assert semester_credits_earned_col is not None
            s_cols.append(semester_credits_earned_col)
        if sem_has_count:
            assert semester_courses_count_col is not None
            s_cols.append(semester_courses_count_col)

        s = semester_df[s_cols].copy()

        agg = (
            c.groupby([id_col, sem_col], dropna=False)
            .agg(
                course_sum_attempted=(resolved_attempted, "sum"),
                course_sum_earned=(resolved_earned, "sum"),
                course_count=(resolved_attempted, "size"),
            )
            .reset_index()
        )

        merged = s.merge(agg, on=[id_col, sem_col], how="left", indicator="_merge")
        merged["has_course_rows"] = merged["_merge"] == "both"

        if sem_has_attempted:
            merged["diff_attempted"] = merged["course_sum_attempted"] - pd.to_numeric(
                merged[semester_credits_attempted_col], errors="coerce"
            )
            merged["match_attempted"] = merged["diff_attempted"].abs() <= credit_tol

        if sem_has_earned:
            merged["diff_earned"] = merged["course_sum_earned"] - pd.to_numeric(
                merged[semester_credits_earned_col], errors="coerce"
            )
            merged["match_earned"] = merged["diff_earned"].abs() <= credit_tol

        mismatch_mask = _credit_reconciliation_mismatch_mask(
            merged,
            sem_has_attempted=sem_has_attempted,
            sem_has_earned=sem_has_earned,
        )

        mismatches = merged.loc[mismatch_mask]

        total_sem = int(len(s))
        n_mismatched = int(len(mismatches))
        reconciliation_summary = {
            "total_semester_rows": total_sem,
            "mismatched_rows": n_mismatched,
            "pct_of_data": percent_of_rows(n_mismatched, total_sem),
        }

        # 🔹 Clean summary logging
        log_semester_reconciliation_summary(
            logger=LOGGER,
            merged=merged,
            agg=agg,
            s=s,
            id_col=id_col,
            sem_col=sem_col,
            sem_has_attempted=sem_has_attempted,
            sem_has_earned=sem_has_earned,
        )

    # =======================================================
    # C) COHORT CHECKS
    # =======================================================
    cohort_anomalies = None
    cohort_anomalies_summary = None

    cohort_attempted_ok = (
        bool(cohort_credits_attempted_col)
        and cohort_credits_attempted_col in cohort_df.columns
        if cohort_df is not None
        else False
    )
    cohort_earned_ok = (
        bool(cohort_credits_earned_col)
        and cohort_credits_earned_col in cohort_df.columns
        if cohort_df is not None
        else False
    )

    if cohort_df is not None and cohort_attempted_ok and cohort_earned_ok:
        LOGGER.info("Running cohort-level earned <= attempted checks")

        cohort_checks = check_earned_vs_attempted(
            cohort_df,
            earned_col=t.cast(str, cohort_credits_earned_col),
            attempted_col=t.cast(str, cohort_credits_attempted_col),
        )

        cohort_anomalies = cohort_checks.get("anomalies")
        cohort_anomalies_summary = cohort_checks.get("summary")

        if isinstance(cohort_anomalies, pd.DataFrame) and len(cohort_anomalies) > 0:
            pct = None
            if (
                cohort_anomalies_summary is not None
                and isinstance(cohort_anomalies_summary, pd.DataFrame)
                and "total_anomalous_rows_pct" in cohort_anomalies_summary.columns
            ):
                pct = cohort_anomalies_summary["total_anomalous_rows_pct"].iloc[0]
            if pct is not None:
                LOGGER.warning(
                    "Detected %d cohort-level anomalies (%.2f%% of cohort data)",
                    len(cohort_anomalies),
                    pct,
                )
            else:
                LOGGER.warning(
                    "Detected %d cohort-level anomalies", len(cohort_anomalies)
                )
        else:
            LOGGER.info("No cohort-level credit anomalies detected")

    # =======================================================
    # Final Summary
    # =======================================================
    LOGGER.info(
        "Credit validation summary: course_anomalies=%d, semester_mismatches=%s, cohort_anomalies=%s",
        0 if course_anomalies is None else int(len(course_anomalies)),
        "skipped" if mismatches is None else int(len(mismatches)),
        "skipped" if cohort_anomalies is None else int(len(cohort_anomalies)),
    )

    out: dict[str, t.Any] = {
        "course_anomalies": course_anomalies,
        "course_anomalies_summary": course_anomalies_summary,
        "reconciliation_summary": reconciliation_summary,
        "reconciliation_mismatches": mismatches,
        "reconciliation_merged_detail": merged,
        "cohort_anomalies": cohort_anomalies,
        "cohort_anomalies_summary": cohort_anomalies_summary,
    }
    out["institution_report"] = format_credit_consistency_institution_report(out)
    return out


def _credit_report_df_scalar(df: t.Any, col: str, row: int = 0) -> t.Any:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    if col not in df.columns:
        return None
    return df[col].iloc[row]


def format_credit_consistency_institution_report(
    result: t.Mapping[str, t.Any],
) -> str:
    """
    Turn the dict returned by :func:`validate_credit_consistency` into a short narrative
    for institutional readers, including suggested next steps.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CREDIT CONSISTENCY — INSTITUTION SUMMARY")
    lines.append("=" * 72)

    course_sum = result.get("course_anomalies_summary")
    recon_sum = result.get("reconciliation_summary")
    cohort_sum = result.get("cohort_anomalies_summary")
    course_bad = result.get("course_anomalies")
    recon_bad = result.get("reconciliation_mismatches")
    cohort_bad = result.get("cohort_anomalies")

    n_course = int(len(course_bad)) if isinstance(course_bad, pd.DataFrame) else 0
    n_recon = int(len(recon_bad)) if isinstance(recon_bad, pd.DataFrame) else 0
    n_cohort = int(len(cohort_bad)) if isinstance(cohort_bad, pd.DataFrame) else 0

    # --- Course-level ---
    lines.append("")
    lines.append("1) Course file (per-row earned vs attempted credits)")
    if isinstance(course_sum, dict):
        checked = course_sum.get("rows_checked", 0)
        bad = course_sum.get("rows_with_anomalies", 0)
        pct = course_sum.get("pct_of_data", 0.0)
        lines.append(
            f"   Checked {checked:,} enrollment rows. "
            f"{bad:,} rows ({pct}%) show earned > attempted, negative credits, or similar issues."
        )
        if bad == 0:
            lines.append("   Status: No issues flagged at course row level.")
        else:
            lines.append(
                "   Status: Review recommended — course-level credits contradict basic rules."
            )
    else:
        lines.append(
            "   Not run — the course file is missing usable attempted/earned credit columns "
            "(or names did not resolve). Confirm column names in config / bronze extract."
        )

    # --- Semester reconciliation ---
    lines.append("")
    lines.append("2) Semester file vs summed course credits (same student + term)")
    if isinstance(recon_sum, dict):
        total = recon_sum.get("total_semester_rows", 0)
        mm = recon_sum.get("mismatched_rows", 0)
        pct = recon_sum.get("pct_of_data", 0.0)
        lines.append(
            f"   Compared {total:,} student-term rows on the semester file to aggregates from courses."
        )
        lines.append(
            f"   {mm:,} rows ({pct}%) do not match within tolerance (attempted and/or earned totals)."
        )
        if mm == 0:
            lines.append("   Status: Semester totals align with summed course credits.")
        else:
            lines.append(
                "   Status: Investigate term keys, withdrawal rules, and how semester aggregates are built."
            )
    else:
        lines.append(
            "   Not run — needs semester extract plus matching student_id and term columns on both "
            "course and semester files, and compatible credit columns."
        )

    # --- Cohort ---
    lines.append("")
    lines.append("3) Cohort / student file (institutional attempted vs earned totals)")
    if isinstance(cohort_sum, pd.DataFrame) and not cohort_sum.empty:
        total_anom = int(
            _credit_report_df_scalar(cohort_sum, "total_anomalous_rows") or 0
        )
        total_pct = _credit_report_df_scalar(cohort_sum, "total_anomalous_rows_pct")
        eg = int(_credit_report_df_scalar(cohort_sum, "earned_gt_attempted") or 0)
        ena = int(_credit_report_df_scalar(cohort_sum, "earned_when_no_attempt") or 0)
        lines.append(
            f"   {total_anom:,} student rows ({total_pct}%) break earned <= attempted or "
            f"earned credit with zero attempted (earned>attempted: {eg:,}; "
            f"credit with no attempt: {ena:,})."
        )
        if total_anom == 0:
            lines.append(
                "   Status: Institutional totals look consistent at row level."
            )
        else:
            lines.append(
                "   Status: Fix upstream SIS totals or clarify transfer / test credit treatment."
            )
    else:
        lines.append(
            "   Not run — cohort file missing institutional attempted/earned total columns, "
            "or cohort extract not provided."
        )

    # --- Overall ---
    lines.append("")
    lines.append("-" * 72)
    any_issue = (n_course + n_recon + n_cohort) > 0
    skipped_all = (
        not isinstance(course_sum, dict)
        and recon_sum is None
        and (not isinstance(cohort_sum, pd.DataFrame) or cohort_sum.empty)
    )
    if skipped_all:
        lines.append(
            'Overall: Checks could not run end-to-end — see sections marked "Not run".'
        )
    elif any_issue:
        lines.append(
            "Overall: At least one layer failed checks. Use the detailed tables in the audit "
            "notebook to sample offending rows and trace back to source systems."
        )
    else:
        lines.append(
            "Overall: No credit consistency issues were flagged in the checks that ran. "
            "Keep monitoring after SIS or ETL changes."
        )

    # --- Next steps ---
    lines.append("")
    lines.append("SUGGESTED NEXT STEPS FOR THE INSTITUTION")
    lines.append("-" * 72)
    steps: list[str] = []
    if not isinstance(course_sum, dict):
        steps.append(
            "Map the correct course-level attempted and earned credit fields in `config.toml` "
            "and re-run the audit (or rename columns in the bronze extract)."
        )
    elif n_course > 0:
        steps.append(
            "Course file: Spot-check programs with the highest anomaly rates; verify credit hours "
            "vs enrollment status and repeat/audit courses."
        )
    if recon_sum is None:
        steps.append(
            "Semester reconciliation: Ensure course and semester files share the same student ID "
            "and term identifier; align column names with `validate_credit_consistency` arguments."
        )
    elif n_recon > 0:
        steps.append(
            "Semester file: Reconcile aggregation logic (sum of course credits vs official term "
            "totals); confirm part-term drops and cross-listed sections are handled consistently."
        )
    if not isinstance(cohort_sum, pd.DataFrame) or cohort_sum.empty:
        steps.append(
            "Cohort file: Expose institutional cumulative attempted and earned credits in the "
            "extract if you want this check; confirm field definitions with the registrar."
        )
    elif n_cohort > 0:
        steps.append(
            "Cohort totals: Work with the registrar or data owner to correct lifetime attempted/"
            "earned totals or document known exceptions (e.g. transfer credit timing)."
        )
    if not any_issue and not skipped_all:
        steps.append(
            "Documentation: Archive this run (counts and date) as evidence for internal QA or "
            "accreditation folders."
        )
    if skipped_all:
        steps.append(
            "Prioritize fixing bronze schema and paths so all three layers (course, semester, cohort) "
            "can be validated automatically."
        )
    if not steps:
        steps.append("No specific follow-up beyond routine monitoring.")
    for i, s in enumerate(steps, start=1):
        lines.append(f"  {i}. {s}")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


CHECK_PF_DEFAULT_PASSING_GRADES: tuple[str, ...] = (
    "P",
    "P*",
    "A",
    "A-",
    "B+",
    "B",
    "B-",
    "C+",
    "C",
    "C-",
    "D+",
    "D",
    "D-",
)
CHECK_PF_DEFAULT_FAILING_GRADES: tuple[str, ...] = (
    "F",
    "E",
    "^E",
    "F*",
    "REF",
    "NR",
    "W",
    "W*",
    "I",
)
CHECK_PF_DEFAULT_PASS_FLAGS: tuple[str, ...] = ("P",)
CHECK_PF_DEFAULT_FAIL_FLAGS: tuple[str, ...] = ("F",)

_PASS_FAIL_FLAG_VALUE_PAIRS: dict[
    frozenset[str], tuple[tuple[str, ...], tuple[str, ...]]
] = {
    frozenset({"Y", "N"}): (("Y",), ("N",)),
    frozenset({"P", "F"}): (("P",), ("F",)),
    frozenset({"PASS", "FAIL"}): (("PASS",), ("FAIL",)),
    frozenset({"COMPLETE", "INCOMPLETE"}): (("COMPLETE",), ("INCOMPLETE",)),
    frozenset({"1", "0"}): (("1",), ("0",)),
    frozenset({"T", "F"}): (("T",), ("F",)),
}


def _observed_upper_tokens(series: pd.Series) -> set[str]:
    return {
        x
        for x in series.dropna().astype(str).str.strip().str.upper().unique()
        if x and str(x).upper() != "NAN"
    }


def infer_pass_fail_flag_tuples(
    pf_series: pd.Series,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    """
    Infer ``pass_flags`` and ``fail_flags`` for :func:`check_pf_grade_consistency` when the
    column has exactly two distinct non-null tokens (e.g. Y/N, P/F).
    """
    u = frozenset(_observed_upper_tokens(pf_series))
    if len(u) != 2:
        return None
    return _PASS_FAIL_FLAG_VALUE_PAIRS.get(u)


_LIKELY_PASSING_GRADE_RE = re.compile(
    r"^([ABCD][+-]?|P\*?|PASS)$",
    re.IGNORECASE,
)
_LIKELY_FAILING_GRADE_RE = re.compile(
    r"^(F[\*]?|E|W[\*]?|I|NR|REF|\^E|INCOMP)$",
    re.IGNORECASE,
)


def infer_check_pf_grade_list_kwargs(
    df: pd.DataFrame,
    grade_col: str,
    pf_col: str,
) -> dict[str, tuple[str, ...]]:
    """
    Build ``passing_grades``, ``failing_grades``, ``pass_flags``, and ``fail_flags`` for
    :func:`check_pf_grade_consistency`: defaults plus pass/fail flags inferred from two-value
    columns, and any observed grade tokens that look like pass/fail but are not in the defaults.
    """
    passing: set[str] = set(CHECK_PF_DEFAULT_PASSING_GRADES)
    failing: set[str] = set(CHECK_PF_DEFAULT_FAILING_GRADES)
    if grade_col in df.columns:
        for g in _observed_upper_tokens(df[grade_col]):
            if g in passing or g in failing:
                continue
            if _LIKELY_PASSING_GRADE_RE.match(g):
                passing.add(g)
            elif (
                _LIKELY_FAILING_GRADE_RE.match(g)
                or len(g) == 1
                and g in {"F", "E", "I", "W"}
            ):
                failing.add(g)

    pf_inf = None
    if pf_col in df.columns:
        pf_inf = infer_pass_fail_flag_tuples(df[pf_col])
    if pf_inf is not None:
        pass_flags, fail_flags = pf_inf
    else:
        pass_flags, fail_flags = (
            CHECK_PF_DEFAULT_PASS_FLAGS,
            CHECK_PF_DEFAULT_FAIL_FLAGS,
        )

    return {
        "passing_grades": tuple(sorted(passing)),
        "failing_grades": tuple(sorted(failing)),
        "pass_flags": pass_flags,
        "fail_flags": fail_flags,
    }


PF_GRADE_ANOMALY_FLAG_COLUMNS: tuple[str, ...] = (
    "earned_with_failing_grade",
    "no_credits_with_passing_grade",
    "grade_pf_disagree",
)


def _pass_fail_label_series(
    normalized_tokens: pd.Series,
    pass_values: tuple[str, ...],
    fail_values: tuple[str, ...],
) -> pd.Series:
    """Map normalized tokens to pass (``True``), fail (``False``), or unknown (missing)."""
    return pd.Series(
        np.where(
            normalized_tokens.isin(pass_values),
            True,
            np.where(normalized_tokens.isin(fail_values), False, np.nan),
        ),
        index=normalized_tokens.index,
        dtype="object",
    )


def check_pf_grade_consistency(
    df: pd.DataFrame,
    grade_col: str = "grade",
    pf_col: str = "pass_fail_flag",
    credits_col: str = "credits_earned",
    *,
    passing_grades: tuple[str, ...] = CHECK_PF_DEFAULT_PASSING_GRADES,
    failing_grades: tuple[str, ...] = CHECK_PF_DEFAULT_FAILING_GRADES,
    pass_flags: tuple[str, ...] = CHECK_PF_DEFAULT_PASS_FLAGS,
    fail_flags: tuple[str, ...] = CHECK_PF_DEFAULT_FAIL_FLAGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    CUSTOM SCHOOL FUNCTION

    Checks that:
      1. Students NEVER earn credits for failing grades.
      2. Students DO always earn credits for passing grades.
      3. Grade and pass_fail_flag are consistent.

    Returns (anomalies_df, summary_df)
    """
    LOGGER.info(
        "Running PF/grade consistency checks "
        "(rows=%d, grade_col=%s, pf_col=%s, credits_col=%s)",
        len(df),
        grade_col,
        pf_col,
        credits_col,
    )

    out = df.copy()

    # Normalize
    g = out[grade_col].astype(str).str.strip().str.upper()
    pf = out[pf_col].astype(str).str.strip().str.upper()
    credits = pd.to_numeric(out[credits_col], errors="coerce")  # keep NaNs as NaN
    LOGGER.debug(
        "Normalized grade/PF/credits (non-null counts: grade=%d, pf=%d, credits=%d)",
        g.notna().sum(),
        pf.notna().sum(),
        credits.notna().sum(),
    )

    pfg = _pass_fail_label_series(g, passing_grades, failing_grades)
    pff = _pass_fail_label_series(pf, pass_flags, fail_flags)

    LOGGER.debug(
        "Derived PF indicators (from grade: pass=%d fail=%d unknown=%d; "
        "from flag: pass=%d fail=%d unknown=%d)",
        int((pfg == True).sum()),
        int((pfg == False).sum()),
        int(pfg.isna().sum()),
        int((pff == True).sum()),
        int((pff == False).sum()),
        int(pff.isna().sum()),
    )

    rules = dict(
        zip(
            PF_GRADE_ANOMALY_FLAG_COLUMNS,
            (
                (pff == False) & credits.notna() & (credits > 0),
                (pff == True) & credits.notna() & (credits == 0),
                pfg.notna() & pff.notna() & (pfg != pff),
            ),
            strict=True,
        )
    )

    LOGGER.debug(
        "Rule violations: %s",
        ", ".join(f"{k}={int(v.sum())}" for k, v in rules.items()),
    )

    mask = pd.Series(False, index=out.index)
    for series in rules.values():
        mask |= series

    anomalies = out.loc[mask].copy()
    for name, series in rules.items():
        anomalies[name] = series.loc[anomalies.index]

    total_rows = len(df)
    summary_payload: dict[str, list[t.Any]] = {}
    for name, series in rules.items():
        n = int(series.sum())
        summary_payload[name] = [n]
        summary_payload[f"{name}_pct"] = [percent_of_rows(n, total_rows)]
    n_total = int(mask.sum())
    summary_payload["total_anomalous_rows"] = [n_total]
    summary_payload["total_anomalous_rows_pct"] = [percent_of_rows(n_total, total_rows)]
    summary = pd.DataFrame(summary_payload)

    if summary["total_anomalous_rows"].iloc[0] > 0:
        LOGGER.warning(
            "Detected %d PF/grade consistency anomalies (%.2f%% of data)",
            summary["total_anomalous_rows"].iloc[0],
            summary["total_anomalous_rows_pct"].iloc[0],
        )
    else:
        LOGGER.info("No PF/grade consistency anomalies detected")

    LOGGER.debug("PF/grade anomaly summary:\n%s", summary)
    return anomalies, summary


_ACADEMIC_TERM_SEASON_TOKENS = frozenset(
    {"Spring", "Summer", "Fall", "Winter", "Autumn"}
)


def value_looks_like_term(val: t.Any) -> bool:
    """
    True if *val* looks like a term string such as ``Spring 2024`` or ``2024 Spring``.
    """
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return False
    parts = s.split()
    if len(parts) != 2:
        return False
    if parts[0].isdigit() and len(parts[0]) == 4:
        token = parts[1].strip().title()
    elif parts[1].isdigit() and len(parts[1]) == 4:
        token = parts[0].strip().title()
    else:
        return False
    return token in _ACADEMIC_TERM_SEASON_TOKENS


def term_column_name_hint_score(col: str, name_hints: tuple[str, ...]) -> float:
    """Small bonus when *col* matches typical term column name substrings."""
    c = col.lower()
    if c in {h.lower() for h in name_hints}:
        return 0.15
    for h in name_hints:
        if h.lower() in c:
            return 0.08
    return 0.0


def infer_term_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...],
    min_match_rate: float = 0.35,
    max_sample: int = 8000,
) -> str | None:
    """Pick the column whose values best resemble academic term codes (e.g. ``Spring 2024``).

    Args:
        df: Table to scan (all columns considered except boolean dtypes).
        name_hints: Substrings / exact names that boost a column (via
            :func:`term_column_name_hint_score`).
        min_match_rate: Minimum share of sampled non-null values that must match
            :func:`value_looks_like_term` for a strong pick (unless name-hint fallback applies).
        max_sample: Cap on non-null rows scored per column (performance for wide/long files).

    Returns:
        Best-scoring column name, or ``None`` if no column clears thresholds.

    Note:
        Score combines empirical term-like rate and name hints; a second pass relaxes
        thresholds slightly if nothing matched the first pass.
    """
    best_col: str | None = None
    best_score = -1.0

    def consider_col(col: str) -> None:
        nonlocal best_col, best_score
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            return
        non_null = s.dropna()
        if len(non_null) == 0:
            return
        sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
        str_sample = sample.astype("string")
        rates = str_sample.map(value_looks_like_term)
        rate = float(rates.mean()) if len(rates) else 0.0
        hint = term_column_name_hint_score(col, name_hints)
        score = rate + hint
        if rate >= min_match_rate or (hint >= 0.08 and rate >= 0.15):
            if score > best_score:
                best_score = score
                best_col = col

    for col in df.columns:
        consider_col(col)

    if best_col is None:
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_bool_dtype(s):
                continue
            non_null = s.dropna()
            if len(non_null) == 0:
                continue
            sample = (
                non_null.head(max_sample) if len(non_null) > max_sample else non_null
            )
            str_sample = sample.astype("string")
            rate = float(str_sample.map(value_looks_like_term).mean())
            hint = term_column_name_hint_score(col, name_hints)
            score = rate + hint
            if score > best_score and rate >= 0.2:
                best_score = score
                best_col = col

    return best_col


# --- Column inference for student-level audits (IDs, credits, demographics) ---

_AUDIT_DEMOGRAPHIC_NAME_BLOCKLIST = (
    "ssn",
    "email",
    "phone",
    "address",
    "uuid",
    "hash",
    "password",
    "name",
    "first_name",
    "last_name",
    "middle_name",
    "street",
    "zip",
    "dob",
    "date_of_birth",
    "birth",
)


def audit_demographic_column_name_blocked(col: str) -> bool:
    """
    True if *col* should not be used for demographic / student-type inference
    (PII-ish or free-text name fields). Does not block legitimate ``student_id``.
    """
    c = col.lower()
    return any(f in c for f in _AUDIT_DEMOGRAPHIC_NAME_BLOCKLIST)


def audit_value_substring_match_rate(
    series: pd.Series,
    substrings: tuple[str, ...],
    *,
    max_sample: int = 8000,
) -> float:
    """Fraction of non-null *series* values whose string form contains a substring."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null

    def matches(v: t.Any) -> bool:
        t_ = str(v).strip().lower()
        if not t_ or t_ == "nan":
            return False
        return any(sub in t_ for sub in substrings)

    return float(sample.map(matches).mean())


DEFAULT_STUDENT_TYPE_NAME_HINTS: tuple[str, ...] = (
    "entry_type",
    "student_type",
    "admit_type",
    "admission_type",
    "enrollment_type",
    "stu_type",
    "student_class",
    "class_level",
    "cohort_type",
)
DEFAULT_STUDENT_TYPE_VALUE_SUBSTRINGS: tuple[str, ...] = (
    "transfer",
    "freshman",
    "fresh",
    "ftic",
    "ftf",
    "first time",
    "first-time",
    "first year",
    "readmit",
    "re-admit",
    "readm",
    "re_admit",
    "continuing",
    "returning",
    "non-degree",
    "nondegree",
    "transient",
    "dual",
    "new",
)
DEFAULT_FIRST_GEN_NAME_HINTS: tuple[str, ...] = (
    "first_gen",
    "first_generation",
    "firstgen",
    "fg_status",
    "firstgeneration",
    "fgen",
    "first_time_college",
    "gen1",
    "parent_education",
)
DEFAULT_RACE_NAME_HINTS: tuple[str, ...] = (
    "race",
    "ipeds_race",
    "racial",
    "race_code",
    "race_ethnicity",
    "ethrace",
)
DEFAULT_ETHNICITY_NAME_HINTS: tuple[str, ...] = (
    "ethnicity",
    "ethnic",
    "hispanic",
    "latinx",
    "latino",
    "latina",
    "hl_indicator",
    "hispanic_latino",
    "is_hispanic",
)
DEFAULT_GENDER_NAME_HINTS: tuple[str, ...] = (
    "gender",
    "legal_sex",
    "biological_sex",
    "sex",
    "gender_identity",
)
DEFAULT_AGE_NAME_HINTS: tuple[str, ...] = (
    "age",
    "student_age",
    "age_at_entry",
    "age_as_of",
    "stu_age",
    "current_age",
    "age_years",
)
DEFAULT_PELL_NAME_HINTS: tuple[str, ...] = (
    "pell",
    "awarded_pell",
    "pell_elig",
    "pell_eligible",
    "pell_recipient",
    "pell_flag",
    "pell_status",
)
DEFAULT_INCARCERATION_NAME_HINTS: tuple[str, ...] = (
    "incarceration",
    "incarcerat",
    "correctional",
    "justice_involved",
    "corrections",
)
DEFAULT_MILITARY_NAME_HINTS: tuple[str, ...] = (
    "military",
    "military_status",
    "veteran",
    "vet_status",
    "armed_forces",
    "service_status",
    "ad_t",
    "national_guard",
    "reserve",
)
DEFAULT_EMPLOYMENT_STATUS_NAME_HINTS: tuple[str, ...] = (
    "employment_status",
    "emp_status",
    "work_status",
    "student_employment",
    "employment",
    "job_status",
    "labor_status",
)
DEFAULT_DISABILITY_NAME_HINTS: tuple[str, ...] = (
    "disability",
    "disab_status",
    "ada",
    "accessibility",
    "disabled",
)

DEFAULT_STUDENT_ID_NAME_HINTS: tuple[str, ...] = (
    "student_id",
    "student id",
    "studentid",
    "stu_id",
    "stuid",
    "emplid",
    "empl_id",
    "pid",
    "person_id",
    "banner_id",
    "bannerid",
    "sis_id",
    "school_id",
    "student_number",
    "student_num",
    "id_number",
)


def infer_student_file_categorical(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...],
    value_substrings: tuple[str, ...] | None,
    exclude_cols: set[str],
    max_sample: int = 8000,
    min_nunique: int = 2,
    max_nunique: int = 80,
    min_value_rate: float = 0.12,
    min_name_hint: float = 0.08,
) -> str | None:
    """
    Pick a column using name hints plus optional substring matches in values.
    Skips PII-ish names and near-unique columns (likely IDs).
    """
    n_rows = len(df)
    best_col: str | None = None
    best_score = -1.0

    for col in df.columns:
        if col in exclude_cols or audit_demographic_column_name_blocked(col):
            continue
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.astype(str).nunique())
        if nunique < min_nunique or nunique > max_nunique:
            continue
        if n_rows and nunique > max(0.92 * n_rows, 500):
            continue

        hint = term_column_name_hint_score(col, name_hints)
        if value_substrings is not None:
            rate = audit_value_substring_match_rate(
                s, value_substrings, max_sample=max_sample
            )
            if hint < min_name_hint and rate < min_value_rate:
                continue
            score = 2.5 * hint + rate
        else:
            if hint < min_name_hint:
                continue
            score = 3.0 * hint + min(1.0, nunique / 50.0)

        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def string_looks_like_age_bucket(s: str) -> bool:
    """
    True if *s* looks like an age band or inequality label (e.g. ``<24``, ``20-24``,
    ``older than 24``), not arbitrary free text.
    """
    t = s.strip().lower()
    if not t or t == "nan":
        return False
    compact = re.sub(r"\s+", " ", t)

    if re.search(r"^\d{1,2}\s*[-–—]\s*\d{1,2}$", compact):
        return True
    if re.search(r"^\d{1,2}\s+to\s+\d{1,2}$", compact):
        return True
    if re.search(r"^\d{1,2}\s+through\s+\d{1,2}$", compact):
        return True
    if re.search(r"^<\s*\d{1,2}$", compact):
        return True
    if re.search(r"^<=\s*\d{1,2}$", compact):
        return True
    if re.search(r"^≤\s*\d{1,2}$", compact):
        return True
    if re.search(r"^>\s*\d{1,2}$", compact):
        return True
    if re.search(r"^>=\s*\d{1,2}$", compact):
        return True
    if re.search(r"^≥\s*\d{1,2}$", compact):
        return True
    if re.search(r"older\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"over\s+\d{1,2}", compact):
        return True
    if re.search(r"under\s+\d{1,2}", compact):
        return True
    if re.search(r"less\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"below\s+\d{1,2}", compact):
        return True
    if re.search(r"at\s+least\s+\d{1,2}", compact):
        return True
    if re.search(r"more\s+than\s+\d{1,2}", compact):
        return True
    if re.search(r"^\d{1,2}\s*\+\s*$", compact):
        return True
    if re.search(r"^\d{1,3}\s*\+\s*$", compact):
        return True
    return False


def age_single_value_plausible(val: t.Any) -> bool:
    """True for plausible numeric age (10–100) or bucket string (e.g. ``20-24``, ``<24``)."""
    if pd.isna(val):
        return False
    n = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
    if pd.notna(n) and np.isfinite(float(n)) and 10 <= float(n) <= 100:
        return True
    s = str(val).strip().lower()
    if not s or s == "nan":
        return False
    return string_looks_like_age_bucket(s)


def _age_plausibility_rate(series: pd.Series, max_sample: int = 8000) -> float:
    """Fraction of non-null values that are plausible numeric ages or age-band strings."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    ok = sample.map(age_single_value_plausible)
    return float(ok.mean()) if len(ok) else 0.0


def infer_age_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...] = DEFAULT_AGE_NAME_HINTS,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_hint: float = 0.08,
    min_plausible_rate: float = 0.65,
    max_nunique: int = 120,
) -> str | None:
    """
    Infer a student age column: integer ages, or categorical bands such as ``<24``,
    ``20-24``, ``older than 24`` (see :func:`string_looks_like_age_bucket`).
    """
    used = set(exclude_cols or ())
    best_col: str | None = None
    best_score = -1.0
    n_rows = len(df)

    for col in df.columns:
        if col in used or audit_demographic_column_name_blocked(col):
            continue
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.nunique(dropna=True))
        if nunique < 2 or nunique > max_nunique:
            continue
        rate = _age_plausibility_rate(s)
        # Near-unique columns are usually IDs; allow when values look like ages.
        if n_rows and nunique > 0.95 * n_rows and rate < 0.85:
            continue
        hint = term_column_name_hint_score(col, name_hints)
        if hint < min_name_hint and rate < min_plausible_rate:
            continue
        score = 2.5 * hint + rate
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def infer_student_audit_columns(
    df: pd.DataFrame,
    *,
    term_col: str | None = None,
    student_type_name_hints: tuple[str, ...] = DEFAULT_STUDENT_TYPE_NAME_HINTS,
    student_type_value_substrings: tuple[
        str, ...
    ] = DEFAULT_STUDENT_TYPE_VALUE_SUBSTRINGS,
    first_gen_name_hints: tuple[str, ...] = DEFAULT_FIRST_GEN_NAME_HINTS,
    race_name_hints: tuple[str, ...] = DEFAULT_RACE_NAME_HINTS,
    ethnicity_name_hints: tuple[str, ...] = DEFAULT_ETHNICITY_NAME_HINTS,
    gender_name_hints: tuple[str, ...] = DEFAULT_GENDER_NAME_HINTS,
    age_name_hints: tuple[str, ...] = DEFAULT_AGE_NAME_HINTS,
    pell_name_hints: tuple[str, ...] = DEFAULT_PELL_NAME_HINTS,
    incarceration_name_hints: tuple[str, ...] = DEFAULT_INCARCERATION_NAME_HINTS,
    military_name_hints: tuple[str, ...] = DEFAULT_MILITARY_NAME_HINTS,
    employment_name_hints: tuple[str, ...] = DEFAULT_EMPLOYMENT_STATUS_NAME_HINTS,
    disability_name_hints: tuple[str, ...] = DEFAULT_DISABILITY_NAME_HINTS,
) -> dict[str, str | None]:
    """
    Infer student-type and equity-related columns; each role maps to at most one column.

    Roles: ``student_type``, ``first_gen``, ``race``, ``ethnicity``, ``gender``, ``age``, ``pell``,
    plus ``incarceration``, ``military``, ``employment``, ``disability`` for extended bias audits.
    """
    used: set[str] = set()
    if term_col:
        used.add(term_col)

    out: dict[str, str | None] = {}

    out["student_type"] = infer_student_file_categorical(
        df,
        name_hints=student_type_name_hints,
        value_substrings=student_type_value_substrings,
        exclude_cols=used,
    )
    if out["student_type"]:
        used.add(out["student_type"])

    for key, hints in (
        ("first_gen", first_gen_name_hints),
        ("race", race_name_hints),
        ("ethnicity", ethnicity_name_hints),
        ("gender", gender_name_hints),
    ):
        out[key] = infer_student_file_categorical(
            df,
            name_hints=hints,
            value_substrings=None,
            exclude_cols=used,
            max_nunique=120,
        )
        inferred_col = out[key]
        if inferred_col:
            used.add(inferred_col)

    out["age"] = infer_age_column(df, name_hints=age_name_hints, exclude_cols=used)
    if out["age"]:
        used.add(out["age"])

    out["pell"] = infer_student_file_categorical(
        df,
        name_hints=pell_name_hints,
        value_substrings=None,
        exclude_cols=used,
        max_nunique=120,
    )
    if out["pell"]:
        used.add(out["pell"])

    for key, hints in (
        ("incarceration", incarceration_name_hints),
        ("military", military_name_hints),
        ("employment", employment_name_hints),
        ("disability", disability_name_hints),
    ):
        out[key] = infer_student_file_categorical(
            df,
            name_hints=hints,
            value_substrings=None,
            exclude_cols=used,
            max_nunique=120,
        )
        inferred_ext = out[key]
        if inferred_ext:
            used.add(inferred_ext)

    return out


def bias_variable_codebook_line(role: str) -> str | None:
    """
    Short decoding hint for institutional audit printouts (codes vary by SIS).

    Typical encodings align with common IPEDS-style and registrar exports.
    """
    hints: dict[str, str] = {
        "first_gen": "Typical codes: Y=Yes, N=No (optional at some institutions).",
        "pell": "Typical codes: Y=Yes, N=No (recipient or eligibility; optional).",
        "incarceration": "Typical codes: Y=Yes, N=No (optional field).",
        "military": (
            "Typical codes: 1=Veteran; 2=Active Duty/Reserves/National Guard; "
            "3=Never served (optional field)."
        ),
        "employment": (
            "Typical codes: 1=full-time; 2=less than full-time but at least half-time; "
            "3=less than half-time; 4=not employed (optional field)."
        ),
        "disability": "Typical codes: Y=has a disability; N=does not.",
    }
    return hints.get(role)


def infer_student_id_column(
    df: pd.DataFrame,
    *,
    name_hints: tuple[str, ...] = DEFAULT_STUDENT_ID_NAME_HINTS,
    min_name_hint: float = 0.08,
    max_sample: int = 8000,
) -> str | None:
    """Infer the primary student identifier column using name hints and cardinality heuristics.

    Args:
        df: Student- or roster-grain table (one row per student expected for best results).
        name_hints: Tokens that indicate an ID column (passed to :func:`term_column_name_hint_score`).
        min_name_hint: Minimum name-hint score for a column to be considered at all.
        max_sample: Max non-null values read per column when checking token length / shape.

    Returns:
        Selected column name, or ``None`` if the frame is empty or no column passes filters.

    Note:
        Strong name matches (e.g. normalized header equals ``studentid``) relax distinct-count
        requirements; weakly named columns need high nunique and mostly short string tokens.
    """
    n_rows = len(df)
    if n_rows == 0:
        return None

    best_col: str | None = None
    best_score = -1.0

    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        nunique = int(non_null.nunique())
        hint = term_column_name_hint_score(col, name_hints)
        if hint < min_name_hint:
            continue
        if hint >= 0.15:
            if nunique < 2:
                continue
        else:
            min_distinct = max(10, min(500, n_rows // 500 or 1))
            if nunique < min_distinct:
                continue
            id_ratio = nunique / n_rows if n_rows else 0.0
            if id_ratio < 0.02:
                continue
        sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
        str_sample = sample.astype("string").str.strip()
        frac_short = float((str_sample.str.len() <= 32).mean())
        if frac_short < 0.95:
            continue
        id_ratio = nunique / n_rows if n_rows else 0.0
        score = 3.0 * hint + id_ratio + 0.1 * frac_short
        cnorm = col.replace("_", "").replace(" ", "").lower()
        if cnorm == "studentid":
            score += 2.0
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def _numeric_coercion_rate(series: pd.Series, max_sample: int = 8000) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    cleaned = sample.astype("string").str.strip().str.replace(",", "", regex=False)
    n = pd.to_numeric(cleaned, errors="coerce")
    return float(n.notna().mean())


DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS: tuple[str, ...] = (
    "inst_tot_credits_attempted",
    "institution_credits_attempted",
    "inst_credits_attempted",
    "total_credits_attempted",
    "ug_credits_attempted",
    "undergrad_credits_attempted",
    "cumulative_credits_attempted",
    "cum_credits_attempted",
    "career_credits_attempted",
    "credits_attempted_inst",
    "number_of_credits_attempted",
    "num_credits_attempted",
    "credit_hours",
    "credit_hours_attempted",
)
DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS: tuple[str, ...] = (
    "inst_tot_credits_earned",
    "institution_credits_earned",
    "inst_credits_earned",
    "total_credits_earned",
    "ug_credits_earned",
    "undergrad_credits_earned",
    "cumulative_credits_earned",
    "cum_credits_earned",
    "career_credits_earned",
    "credits_earned_inst",
    "number_of_credits_earned",
    "num_credits_earned",
    "no_of_credits_earned",
    "credit_hours_earned",
)


_CREDIT_NAME_TYPO_NORMALIZATIONS: tuple[tuple[str, str], ...] = (
    ("cumlative", "cumulative"),
    ("cumulitive", "cumulative"),
    ("comulative", "cumulative"),
    ("comulitive", "cumulative"),
)


def _normalize_credit_column_name(col: str) -> str:
    c = col.lower().replace(" ", "_").replace("-", "_")
    for wrong, right in _CREDIT_NAME_TYPO_NORMALIZATIONS:
        c = c.replace(wrong, right)
    return c


def credit_column_name_has_attempt_marker(col: str) -> bool:
    """
    True if *col* name signals **attempted** credits (not cumulative totals).

    Matches ``attempt``, ``attmpt``, or a standalone ``att`` token (e.g. ``sem_att_credits``).
    """
    c = _normalize_credit_column_name(col)
    if "attempt" in c or "attmpt" in c:
        return True
    return re.search(r"(^|_)att(_|$)", c) is not None


def credits_attempted_column_name_score(col: str) -> float:
    """
    **Attempted** institutional / aggregate credits: requires **attempt** markers (or
    enrollment **hours**). Cumulative/total **without** attempt → not scored here (those
    default to **earned** in :func:`credits_earned_column_name_score`).
    """
    c = _normalize_credit_column_name(col)
    if "credit" not in c:
        return 0.0
    if ("earned" in c) and not credit_column_name_has_attempt_marker(col):
        return 0.0
    has_att = credit_column_name_has_attempt_marker(col)
    s = 0.0
    if "hour" in c or "hrs" in c or c.endswith("_hr") or "_hr_" in c:
        if "earned" not in c:
            s += 1.15
    if has_att:
        s += 1.25
        if "cum" in c or "cumulative" in c:
            s += 0.5
        if "total" in c or "tot_" in c or c.startswith("inst") or "_inst_" in c:
            s += 0.3
        if "number_of" in c or "num_" in c or "nbr_" in c:
            s += 0.15
    return s


def credits_earned_column_name_score(col: str) -> float:
    """
    **Earned** credits: explicit ``earned``, or cumulative/total credit columns **without**
    attempt markers (e.g. ``total_cumlative_credits`` → earned total).
    """
    c = _normalize_credit_column_name(col)
    if "credit" not in c:
        return 0.0
    has_att = credit_column_name_has_attempt_marker(col)
    if has_att and "earned" not in c:
        return 0.0
    s = 0.0
    if "earned" in c:
        s += 1.25
    if not has_att:
        if "cum" in c or "cumulative" in c:
            s += 0.85
        if "total" in c or "tot_" in c or c.startswith("inst") or "_inst_" in c:
            s += 0.55
        if "number_of" in c or "num_" in c or "nbr_" in c:
            s += 0.15
        if s < 0.9 and ("cum" in c or "cumulative" in c):
            s = max(s, 0.95)
    return s


def _pick_distinct_credit_columns(
    att_ranked: list[tuple[float, str]],
    ern_ranked: list[tuple[float, str]],
) -> tuple[str | None, str | None]:
    attempted_col = att_ranked[0][1] if att_ranked else None
    earned_col = ern_ranked[0][1] if ern_ranked else None
    if attempted_col and earned_col and attempted_col == earned_col:
        alt_e = next((c for _, c in ern_ranked if c != attempted_col), None)
        alt_a = next((c for _, c in att_ranked if c != earned_col), None)
        if alt_e is not None:
            earned_col = alt_e
        elif alt_a is not None:
            attempted_col = alt_a
        else:
            earned_col = None
    return attempted_col, earned_col


def _rank_columns_by_name_score(
    df: pd.DataFrame,
    *,
    column_score_fn: t.Callable[[str], float],
    name_hints: tuple[str, ...],
    min_base_score: float,
    tiebreak_fn: t.Callable[[pd.Series], float],
    tiebreak_weight: float,
    exclude_cols: t.AbstractSet[str] | None = None,
) -> list[tuple[float, str]]:
    """
    Rank dataframe columns by ``column_score_fn`` + hint bonus, with an optional tie-break
    from *tiebreak_fn(series)* (e.g. numeric coercion rate or string populated rate).
    """
    ranked: list[tuple[float, str]] = []
    skip = set(exclude_cols or ())
    for col in df.columns:
        if col in skip or pd.api.types.is_bool_dtype(df[col]):
            continue
        base = float(column_score_fn(col))
        base += 2.0 * term_column_name_hint_score(col, name_hints)
        if base < min_base_score:
            continue
        tb = tiebreak_fn(df[col])
        ranked.append((base + tiebreak_weight * tb, col))
    ranked.sort(key=lambda x: -x[0])
    return ranked


def _infer_two_credit_columns_by_name(
    df: pd.DataFrame,
    *,
    attempted_score_fn: t.Callable[[str], float],
    earned_score_fn: t.Callable[[str], float],
    attempted_name_hints: tuple[str, ...],
    earned_name_hints: tuple[str, ...],
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """Shared ranking for attempted vs earned credit columns (numeric tie-break only)."""
    if len(df.columns) == 0:
        return None, None

    def _num_tb(ser: pd.Series) -> float:
        return _numeric_coercion_rate(ser, max_sample=max_sample)

    return _pick_distinct_credit_columns(
        _rank_columns_by_name_score(
            df,
            column_score_fn=attempted_score_fn,
            name_hints=attempted_name_hints,
            min_base_score=min_name_score,
            tiebreak_fn=_num_tb,
            tiebreak_weight=numeric_tiebreak_weight,
        ),
        _rank_columns_by_name_score(
            df,
            column_score_fn=earned_score_fn,
            name_hints=earned_name_hints,
            min_base_score=min_name_score,
            tiebreak_fn=_num_tb,
            tiebreak_weight=numeric_tiebreak_weight,
        ),
    )


def infer_inst_tot_credits_columns(
    df: pd.DataFrame,
    *,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """
    Infer credits **attempted** and **earned** by **column name** (cohort or semester).

    Cumulative/total credit fields **without** ``attempt`` / ``att`` tokens are treated as
    **earned** (e.g. misspelled ``total_cumlative_credits``). Attempted requires attempt
    markers or **credit hours**-style enrollment columns.
    """
    if df is None or len(df.columns) == 0:
        return None, None
    return _infer_two_credit_columns_by_name(
        df,
        attempted_score_fn=credits_attempted_column_name_score,
        earned_score_fn=credits_earned_column_name_score,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )


# --- Course-row and semester aggregate columns (validate_credit_consistency / check_pf) ---

DEFAULT_COURSE_ROW_CREDITS_ATTEMPTED_NAME_HINTS: tuple[str, ...] = (
    "course_credits_attempted",
    "credits_attempted",
    "credit_hours",
    "credithours",
    "credit_hour",
    "class_credit_hours",
    "ug_credits_attempted",
    "attempted_credits",
    "enrollment_credits",
    "registered_credits",
)
DEFAULT_COURSE_ROW_CREDITS_EARNED_NAME_HINTS: tuple[str, ...] = (
    "course_credits_earned",
    "credits_earned",
    "number_of_credits_earned",
    "num_credits_earned",
    "no_of_credits_earned",
    "no._of_credits_earned",
    "credit_earned",
    "credits_earned_course",
)

DEFAULT_COURSE_GRADE_NAME_HINTS: tuple[str, ...] = (
    "grade",
    "class_grade",
    "course_grade",
    "letter_grade",
    "final_grade",
    "official_grade",
)
DEFAULT_COURSE_PF_NAME_HINTS: tuple[str, ...] = (
    "pass_fail",
    "passfail",
    "completion",
    "complete",
    "completion_status",
    "course_status",
    "enrollment_status",
)

DEFAULT_SEMESTER_COURSE_COUNT_NAME_HINTS: tuple[str, ...] = (
    "number_of_courses_enrolled",
    "number_of_courses",
    "courses_enrolled",
    "no_of_classes",
    "num_classes",
    "class_count",
    "n_courses",
    "course_count",
    "courses_taken",
)


def course_row_credits_attempted_name_score(col: str) -> float:
    """
    Course-enrollment **attempted** credits: e.g. *Credit Hours*, *credits attempted*, *units*.
    """
    c = _normalize_credit_column_name(col)
    if "earned" in c:
        if "hour" in c and "earned" in c:
            return 0.0
        if "attempt" not in c and "hour" not in c and "unit" not in c:
            return 0.0
    s = float(credits_attempted_column_name_score(col))
    if "hour" in c or "hrs" in c or c.endswith("_hr") or "_hr_" in c:
        s += 1.2
    if "unit" in c:
        s += 0.95
    if "credit" in c and "hour" in c and "earned" not in c:
        s = max(s, 1.4)
    if (
        "credit" in c
        and s < 0.55
        and not credit_column_name_has_attempt_marker(col)
        and "hour" not in c
        and "unit" not in c
    ):
        s = max(s, 0.52)
    return s


def course_row_credits_earned_name_score(col: str) -> float:
    """Course-row **earned** credits: e.g. *No. of Credits Earned*."""
    s = float(credits_earned_column_name_score(col))
    c = _normalize_credit_column_name(col)
    if "no_" in c or "number" in c or "nbr_" in c:
        s += 0.22
    return s


def infer_course_credit_columns(
    df: pd.DataFrame,
    *,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_COURSE_ROW_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_COURSE_ROW_CREDITS_EARNED_NAME_HINTS,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    max_sample: int = 8000,
) -> tuple[str | None, str | None]:
    """
    Infer per-enrollment credits **attempted** and **earned** on a **course** file
    (e.g. *Credit Hours* vs *No. of Credits Earned*).
    """
    if len(df.columns) == 0:
        return None, None
    return _infer_two_credit_columns_by_name(
        df,
        attempted_score_fn=course_row_credits_attempted_name_score,
        earned_score_fn=course_row_credits_earned_name_score,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )


def semester_course_count_column_name_score(col: str) -> float:
    """Semester-level count of classes/courses (not credit hours)."""
    c = _normalize_credit_column_name(col)
    if "credit" in c and "hour" in c:
        return 0.0
    if "credit" in c and "attempt" in c:
        return 0.0
    s = 0.0
    if "class" in c:
        s += 0.9
    if "classes" in c:
        s += 0.95
    if "course" in c and (
        "count" in c or "number" in c or "num" in c or "nbr" in c or "no_" in c
    ):
        s += 1.2
    if "enroll" in c and ("course" in c or "class" in c):
        s += 0.8
    if "sections" in c:
        s += 0.55
    return s


def _string_populated_rate(series: pd.Series, max_sample: int = 8000) -> float:
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    sample = non_null.head(max_sample) if len(non_null) > max_sample else non_null
    t = sample.astype("string").str.strip()
    return float((t.str.len() > 0).mean())


def infer_semester_credit_aggregate_columns(
    df: pd.DataFrame,
    *,
    min_name_score: float = 0.45,
    numeric_tiebreak_weight: float = 0.12,
    count_min_name_score: float = 0.42,
    count_tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    attempted_name_hints: tuple[
        str, ...
    ] = DEFAULT_INST_TOT_CREDITS_ATTEMPTED_NAME_HINTS,
    earned_name_hints: tuple[str, ...] = DEFAULT_INST_TOT_CREDITS_EARNED_NAME_HINTS,
    course_count_name_hints: tuple[str, ...] = DEFAULT_SEMESTER_COURSE_COUNT_NAME_HINTS,
) -> tuple[str | None, str | None, str | None]:
    """
    Infer semester file columns: credits **attempted**, **earned**, and **course count**
    (e.g. *credit_hours*, *no_of_credits_earned*, *no_of_classes*).
    """
    if len(df.columns) == 0:
        return None, None, None

    att, earn = infer_inst_tot_credits_columns(
        df,
        attempted_name_hints=attempted_name_hints,
        earned_name_hints=earned_name_hints,
        min_name_score=min_name_score,
        numeric_tiebreak_weight=numeric_tiebreak_weight,
        max_sample=max_sample,
    )
    used = {c for c in (att, earn) if c}
    ranked = _rank_columns_by_name_score(
        df,
        column_score_fn=semester_course_count_column_name_score,
        name_hints=course_count_name_hints,
        min_base_score=count_min_name_score,
        tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
        tiebreak_weight=count_tiebreak_weight,
        exclude_cols=used,
    )
    count_col = ranked[0][1] if ranked else None
    return att, earn, count_col


def semester_enrollment_intensity_column_name_score(col: str) -> float:
    """
    Semester-level **student** full-time vs part-time (enrollment intensity), not instructor FT/PT.
    """
    c = _normalize_credit_column_name(col)
    if "instructor" in c:
        return 0.0
    if "frac_" in c or "cumfrac" in c or "cum_frac" in c:
        return 0.0
    s = 0.0
    if c == "ftpt" or "_ftpt" in c or c.startswith("ftpt_"):
        s = max(s, 1.42)
    if "student_term_enrollment_intensity" in c:
        s = max(s, 1.38)
    if "enrollment_intensity" in c or "enroll_intensity" in c:
        s = max(s, 1.32)
    if "enroll" in c and "intensity" in c:
        s = max(s, 1.28)
    if "full" in c and "part" in c and "time" in c:
        s = max(s, 1.18)
    if ("full_time" in c or "part_time" in c) and "instructor" not in c:
        s = max(s, 1.08)
    if "time_status" in c or "timestatus" in c:
        s = max(s, 1.02)
    if "term" in c and "intensity" in c and "instructor" not in c:
        s = max(s, 1.0)
    if "load" in c and ("acad" in c or "enroll" in c):
        s = max(s, 0.55)
    return s


def infer_semester_enrollment_intensity_column(
    df: pd.DataFrame,
    *,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_score: float = 0.40,
    tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    name_hints: tuple[str, ...] | None = None,
) -> str | None:
    """
    Infer **full-time vs part-time** (enrollment intensity) on a **semester** / student-term file.

    Typical column names: *ftpt*, *student_term_enrollment_intensity*, *enrollment_intensity*.
    Excludes credit-hour and term-key columns when passed via ``exclude_cols``.

    When ``name_hints`` is None, a built-in default hint tuple is used. Pass an explicit
    tuple (including ``()``) to override.
    """
    if name_hints is None:
        name_hints = (
            "student_term_enrollment_intensity",
            "enrollment_intensity",
            "ftpt",
            "full_part_time",
            "full_part",
            "full_time",
            "part_time",
            "time_status",
            "academic_load",
            "credit_load",
        )
    if len(df.columns) == 0:
        return None
    ranked = _rank_columns_by_name_score(
        df,
        column_score_fn=semester_enrollment_intensity_column_name_score,
        name_hints=name_hints,
        min_base_score=min_name_score,
        tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
        tiebreak_weight=tiebreak_weight,
        exclude_cols=exclude_cols,
    )
    return ranked[0][1] if ranked else None


def course_grade_column_name_score(col: str) -> float:
    c = _normalize_credit_column_name(col)
    s = 0.0
    if "grade" in c:
        s += 1.2
    if "letter" in c:
        s += 0.45
    if "final" in c and "grade" in c:
        s += 0.35
    if "gpa" in c and "course" not in c and "class" not in c:
        s += 0.2
    if "mark" in c and "grade" not in c:
        s += 0.4
    if "midterm" in c or "mid_term" in c:
        s *= 0.35
    if "point" in c and "grade" not in c:
        return 0.0
    return s


def course_pass_fail_column_name_score(col: str) -> float:
    c = _normalize_credit_column_name(col)
    s = 0.0
    if "pass" in c and "fail" in c:
        s += 1.3
    if "pass_fail" in c or "passfail" in c:
        s += 1.25
    if "completion" in c or "complete" in c:
        s += 1.05
    if "status" in c:
        s += 0.5
    if "success" in c:
        s += 0.45
    return s


def infer_course_grade_pf_columns(
    df: pd.DataFrame,
    *,
    exclude_cols: t.AbstractSet[str] | None = None,
    min_name_score: float = 0.42,
    tiebreak_weight: float = 0.06,
    max_sample: int = 8000,
    grade_name_hints: tuple[str, ...] = DEFAULT_COURSE_GRADE_NAME_HINTS,
    pf_name_hints: tuple[str, ...] = DEFAULT_COURSE_PF_NAME_HINTS,
) -> tuple[str | None, str | None]:
    """
    Infer **grade** and **pass/fail or completion status** columns on a course file.

    Typical examples: *Class Class Grade*, *Class Completion Status*.
    """
    if len(df.columns) == 0:
        return None, None

    used: set[str] = set(exclude_cols or ())

    def pick(score_fn: t.Callable[[str], float], hints: tuple[str, ...]) -> str | None:
        ranked = _rank_columns_by_name_score(
            df,
            column_score_fn=score_fn,
            name_hints=hints,
            min_base_score=min_name_score,
            tiebreak_fn=lambda ser: _string_populated_rate(ser, max_sample=max_sample),
            tiebreak_weight=tiebreak_weight,
            exclude_cols=used,
        )
        return ranked[0][1] if ranked else None

    grade_col = pick(course_grade_column_name_score, grade_name_hints)
    if grade_col:
        used.add(grade_col)
    pf_col = pick(course_pass_fail_column_name_score, pf_name_hints)
    return grade_col, pf_col


def duplicate_conflict_columns(
    df: pd.DataFrame, primary_keys: list[str]
) -> pd.DataFrame:
    """Return a frame of columns and the percent of duplicate-key groups where values disagree.

    Args:
        df: Source table.
        primary_keys: Key columns defining duplicates (same semantics as :func:`find_dupes`).

    Returns:
        Two columns — ``column`` (feature name) and ``pct_conflicting_groups`` (0–100), sorted
        descending by conflict rate. Empty frame when there are no duplicate keys or no
        within-group conflicts.

    Note:
        Unlike :func:`find_dupes`, this does not print; use for programmatic reporting.
    """
    return _duplicate_key_conflict_metrics(df, primary_keys)
