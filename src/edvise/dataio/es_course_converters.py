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

    n_unique, n_extra = int(drop_unique.sum()), int(drop_extra.sum())
    LOGGER.warning(
        "missing_grade_func: null=%d keep_dup(e0=%d,e>0=%d) "
        "drop_unique(e0=%d,e>0=%d) drop_all_null_siblings=%d",
        int(gnull.sum()),
        int((keep & ez).sum()),
        int((keep & ~ez).sum()),
        int((drop_unique & ez).sum()),
        int((drop_unique & ~ez).sum()),
        n_extra,
    )
    if n_unique:
        log = LOGGER.error if n_unique >= _HIGH_DROP else LOGGER.warning
        log(
            "missing_grade_func: dropped %d unique null-grade rows%s",
            n_unique,
            " — HIGH; contact school" if n_unique >= _HIGH_DROP else "",
        )
    if n_extra:
        LOGGER.warning(
            "missing_grade_func: dropped %d extra all-null/zero-earned sibling row(s)",
            n_extra,
        )
    if keep.any() and keys is not None and catalog is not None:
        lines = ["missing_grade_func: duplicate-match examples"]
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
                f"  ex{i}  {a['learner_id']}  {a['course_prefix']} {a['course_number']}"
                f" (catalog {catalog.loc[idx]})",
                f"    incomplete  {a.get('academic_year', '')} {a.get('academic_term', '')}"
                f"  grade={a['grade']!r}  att={a.get('course_credits_attempted', pd.NA)!r}"
                f"  ern={a.get('course_credits_earned', pd.NA)!r}",
                f"    duplicate-match  {b.get('academic_year', '')} {b.get('academic_term', '')}"
                f"  grade={b['grade']!r}  att={b.get('course_credits_attempted', pd.NA)!r}"
                f"  ern={b.get('course_credits_earned', pd.NA)!r}"
                f"  course_number={b.get('course_number', '')!r}",
            ]
        LOGGER.warning("\n".join(lines))

    out = df.loc[~drop].copy()
    k = keep.loc[out.index]
    if not k.any():
        return out

    out.loc[k, "grade"] = "M"
    e = earned.loc[out.index].fillna(0.0)
    if "course_credits_earned" in out.columns:
        out.loc[k, "course_credits_earned"] = e.loc[k]
    if "course_credits_attempted" in out.columns:
        out.loc[k, "course_credits_attempted"] = pd.to_numeric(
            out.loc[k, "course_credits_attempted"], errors="coerce"
        ).fillna(e.loc[k])
    return out
