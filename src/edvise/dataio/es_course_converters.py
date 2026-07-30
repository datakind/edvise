import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)

_MISSING = frozenset(
    {"", "NAN", "NONE", "NULL", "<NA>", "NAT", "N/A", "NA", "#N/A", ".", "-", "--"}
)
_HIGH_DROP = 100
_KEY = ("learner_id", "course_prefix", "course_number")


def handle_missing_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Null grade handling for Edvise Schema course data.

    Duplicate is defined as: same ``learner_id`` + ``course_id`` [where ``course_id`` is
    ``course_prefix`` + ``course_number``]
    Trailing suffix ``-{n}`` (ex. ENG101-1, ENG101-2) stripped for matching only
    (stored ``course_number`` is unchanged); this is because GenAI may suffix before
    we run this converter, so we ensure those rows aren't dropped / still count as
    repeats. Typically another term; term is not part of the key. But can also apply
    to a "retake" in the same term.
    Unique is defined as: no duplicate record found.

    Keep / drop rules for records with null grade values:

    - Duplicate found + credits_earned is null OR 0:
        Keep; grade='M'; normalize missing credits to 0 if not already 0; log count.
        If credits_attempted is ALSO null, set it to 0.

    - Duplicate found + credits_earned > 0:
        Keep; grade='M'; preserve credits_earned value; log count.
        If credits_attempted is null, set it to the credits_earned value.

    - If BOTH duplicates are missing grades, AND credits_earned is null OR 0 for both:
        Keep only the first row (input order); drop the rest. Log count.

    - Unique record, credits_earned null OR 0:
        Drop; log count; flag if count is high.

    - Unique record, credits_earned > 0:
        Drop; log count; flag if count is high.
    """
    if "grade" not in df.columns:
        return df

    # Stable: masks follow input row order; kept rows retain original index order.
    gnull = (
        df["grade"].astype("string").fillna("").str.strip().str.upper().isin(_MISSING)
    )
    if not gnull.any():
        return df

    earned = (
        pd.to_numeric(df["course_credits_earned"], errors="coerce")
        if "course_credits_earned" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )
    ez = earned.isna() | earned.eq(0)

    catalog = keys = None
    if all(c in df.columns for c in _KEY):
        catalog = (
            df["course_number"].astype("string").str.replace(r"-\d+$", "", regex=True)
        )
        keys = [df["learner_id"], df["course_prefix"], catalog]
        is_dup = pd.DataFrame(
            {"a": keys[0], "b": keys[1], "c": keys[2]}, index=df.index
        ).duplicated(keep=False)
        collapse = (
            is_dup
            & gnull.groupby(keys, dropna=False).transform("all")
            & ez.groupby(keys, dropna=False).transform("all")
        )
        drop_extra = collapse & pd.Series(0, index=df.index).groupby(
            keys, dropna=False
        ).cumcount().gt(0)
    else:
        is_dup = drop_extra = pd.Series(False, index=df.index)

    drop_unique = gnull & ~is_dup
    drop = drop_unique | drop_extra
    keep = gnull & is_dup & ~drop_extra

    n_null = int(gnull.sum())
    n_keep_earned_zero = int((keep & ez).sum())
    n_keep_earned_pos = int((keep & ~ez).sum())
    n_drop_unique_earned_zero = int((drop_unique & ez).sum())
    n_drop_unique_earned_pos = int((drop_unique & ~ez).sum())
    n_extra = int(drop_extra.sum())
    n_unique = int(drop_unique.sum())
    n_recode = n_keep_earned_zero + n_keep_earned_pos
    # Among kept rows, null credits_earned will be set to 0 (already-0 left unchanged).
    n_earned_modified = (
        int(earned.loc[df.index[keep]].isna().sum())
        if "course_credits_earned" in df.columns and keep.any()
        else 0
    )

    def _pct(n: int) -> float:
        return (100.0 * n / n_null) if n_null else 0.0

    LOGGER.warning(
        "handle_missing_grades: found %d null-grade row(s). Case breakdown:\n"
        "  Duplicate found + credits_earned is null OR 0 → "
        "Keep; grade='M': %d (%.1f%%)\n"
        "  Duplicate found + credits_earned > 0 → "
        "Keep; grade='M': %d (%.1f%%)\n"
        "  BOTH duplicates missing grades, AND credits_earned is null OR 0 "
        "for both → Keep first row; drop the rest: %d (%.1f%%)\n"
        "  Unique record, credits_earned null OR 0 → Drop: %d (%.1f%%)\n"
        "  Unique record, credits_earned > 0 → Drop: %d (%.1f%%)\n"
        "  Total kept and recoded to grade='M': %d (%.1f%%)\n"
        "  credits_earned field modified (null → 0) on kept rows: %d (%.1f%%)",
        n_null,
        n_keep_earned_zero,
        _pct(n_keep_earned_zero),
        n_keep_earned_pos,
        _pct(n_keep_earned_pos),
        n_extra,
        _pct(n_extra),
        n_drop_unique_earned_zero,
        _pct(n_drop_unique_earned_zero),
        n_drop_unique_earned_pos,
        _pct(n_drop_unique_earned_pos),
        n_recode,
        _pct(n_recode),
        n_earned_modified,
        _pct(n_earned_modified),
    )
    if n_unique >= _HIGH_DROP:
        LOGGER.error(
            "handle_missing_grades: dropped %d unique null-grade rows — "
            "count is high; contact the school.",
            n_unique,
        )
    if keep.any() and keys is not None and catalog is not None:
        lines = ["handle_missing_grades: duplicate-match examples"]
        for i, idx in enumerate(df.index[keep][:5], 1):
            a = df.loc[idx]
            same = (
                (keys[0] == a["learner_id"])
                & (keys[1] == a["course_prefix"])
                & (keys[2] == catalog.loc[idx])
                & (df.index != idx)
            )
            others = df.index[same]
            b = df.loc[others[0]] if len(others) else a
            lines += [
                f"  ex{i}  {a['learner_id']}  {a['course_prefix']} {a['course_number']}",
                f"    incomplete  {a.get('academic_year', '')} {a.get('academic_term', '')}"
                f"  grade={a['grade']!r}  attempted={a.get('course_credits_attempted', pd.NA)!r}"
                f"  earned={a.get('course_credits_earned', pd.NA)!r}",
                f"    duplicate-match  {b.get('academic_year', '')} {b.get('academic_term', '')}"
                f"  grade={b['grade']!r}  attempted={b.get('course_credits_attempted', pd.NA)!r}"
                f"  earned={b.get('course_credits_earned', pd.NA)!r}",
            ]
        LOGGER.warning("\n".join(lines))

    out = df.loc[~drop].copy()
    k = keep.loc[out.index]
    if not k.any():
        LOGGER.warning(
            "handle_missing_grades: no kept rows, so no credits fields were edited."
        )
        return out

    out.loc[k, "grade"] = "M"
    e = earned.loc[out.index].fillna(0.0)
    keep_ez = ez.loc[out.index] & k
    keep_pos = (~ez.loc[out.index]) & k
    n_attempted_to_zero = 0
    n_attempted_to_earned = 0
    if "course_credits_earned" in out.columns:
        # Null earned → 0; already-0 / positive values are left as-is (same numeric value).
        out.loc[k, "course_credits_earned"] = e.loc[k]
    if "course_credits_attempted" in out.columns:
        att = pd.to_numeric(out.loc[k, "course_credits_attempted"], errors="coerce")
        att_missing = att.isna()
        # Null attempted + earned null/0 → 0; null attempted + earned > 0 → earned.
        n_attempted_to_zero = int((att_missing & keep_ez.loc[out.index[k]]).sum())
        n_attempted_to_earned = int((att_missing & keep_pos.loc[out.index[k]]).sum())
        out.loc[k, "course_credits_attempted"] = att.fillna(e.loc[k])

    LOGGER.warning(
        "handle_missing_grades: credits field edits on kept rows:\n"
        "  credits_earned modified: %d "
        "(null → 0; already-0 and >0 values left unchanged)\n"
        "  credits_attempted modified: %d set to 0 "
        "(were null and credits_earned was null or 0); "
        "%d set to credits_earned "
        "(were null and credits_earned was > 0)",
        n_earned_modified,
        n_attempted_to_zero,
        n_attempted_to_earned,
    )
    return out
