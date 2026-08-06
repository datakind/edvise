"""Target–feature association helpers for model_prep force_include suggestions."""

from __future__ import annotations

import re

import pandas as pd

from edvise.feature_generation.constants import (
    FRAC_COURSE_FEATURE_COL_PREFIX,
    NUM_COURSE_FEATURE_COL_PREFIX,
)

# Longest-first; strips frac/num/cum* prefixes so family variants share one key.
_FEATURE_FAMILY_RE = re.compile(
    rf"^(?:cumfrac_{NUM_COURSE_FEATURE_COL_PREFIX}_|cumfrac_num_|"
    rf"{FRAC_COURSE_FEATURE_COL_PREFIX}_|{NUM_COURSE_FEATURE_COL_PREFIX}_|"
    r"cumsum_|cummin_|cummean_|cummax_|cumstd_|cumnum_|cumcount_|cumfrac_|"
    r"frac_|num_)+"
)


def feature_family(name: str) -> str:
    """Map e.g. frac_/num_/cumfrac_num_courses_course_grade_f → course_grade_f."""
    return _FEATURE_FAMILY_RE.sub("", name)


def _name_pref(name: str) -> int:
    """Lower is better: frac_ < num_ < other < cum*."""
    if name.startswith("frac_"):
        return 0
    if name.startswith("num_"):
        return 1
    return 3 if name.startswith("cum") else 2


def _best_scored_name(
    scores: pd.Series,
    *,
    exclude: set[str],
    skip_cumulative: bool,
    abs_score: bool = False,
) -> dict[str, tuple[str, float]]:
    """Collapse scored columns to one (name, score) per feature family."""
    best: dict[str, tuple[str, float]] = {}
    for name, val in scores.items():
        ns = str(name)
        if ns in exclude or (skip_cumulative and ns.startswith("cum")):
            continue
        score = abs(float(val)) if abs_score else float(val)
        cmp = abs(score)
        fam = feature_family(ns)
        cur = best.get(fam)
        if cur is None or cmp > abs(cur[1]) or (
            cmp == abs(cur[1]) and _name_pref(ns) < _name_pref(cur[0])
        ):
            best[fam] = (ns, score)
    return best


def _top_names(ser: pd.Series, n: int, *, floor: float | None = None) -> list[str]:
    """Top ``n`` names by sort order, skipping tied values; optional |score| floor."""
    if floor is not None:
        ser = ser[ser.abs() >= floor]
    # After sorting, keep first of each distinct value.
    return ser[~ser.duplicated(keep="first")].head(n).index.astype(str).tolist()


def suggest_force_include_cols(
    corrs: pd.Series,
    assocs: pd.Series,
    *,
    exclude: set[str],
    n_each: int = 3,
    skip_cumulative: bool = False,
) -> list[str]:
    """Top-3 +pos and top-3 −neg corr families; stronger assoc can win the name."""
    best = _best_scored_name(
        corrs, exclude=exclude, skip_cumulative=skip_cumulative
    )
    assoc_best = _best_scored_name(
        assocs, exclude=exclude, skip_cumulative=skip_cumulative, abs_score=True
    )
    for fam, (aname, a) in assoc_best.items():
        cur = best.get(fam)
        if cur is None:
            continue
        cname, corr = cur
        if a > abs(corr) or (a == abs(corr) and _name_pref(aname) < _name_pref(cname)):
            best[fam] = (aname, corr)

    by_name = pd.Series({name: val for name, val in best.values()}, dtype="float64")
    pos = by_name[by_name > 0].sort_values(ascending=False)
    neg = by_name[by_name < 0].sort_values(ascending=True)
    floor = float(pos.iloc[0]) * 0.5 if not pos.empty else None
    return _top_names(pos, n_each, floor=floor) + _top_names(neg, n_each)
