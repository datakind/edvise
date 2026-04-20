"""Structured custom-school data audit: validation and reconciliation checks.

These helpers return reports and anomaly frames without applying pipeline cleaning
transforms. Exploratory logging-only utilities live in :mod:`edvise.data_audit.eda`;
DataFrame transforms live in :mod:`edvise.data_audit.custom_cleaning`.
"""

import logging
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
