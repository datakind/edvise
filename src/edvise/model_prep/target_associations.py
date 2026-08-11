"""Target–feature association helpers for model_prep force_include suggestions."""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from edvise.data_audit.eda import compute_pairwise_associations
from edvise.feature_generation.constants import (
    FRAC_COURSE_FEATURE_COL_PREFIX,
    NUM_COURSE_FEATURE_COL_PREFIX,
)
from edvise.shared.validation import is_boolean_like, warn_if
from edvise.utils.data_cleaning import unique_elements_in_order

LOGGER = logging.getLogger(__name__)

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
    if name.startswith("frac_"):
        return 0
    if name.startswith("num_"):
        return 1
    return 3 if name.startswith("cum") else 2


def _feature_cols(df: pd.DataFrame, target_col: str, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c != target_col and c not in exclude]


def compute_spearman_corrs(
    df: pd.DataFrame, *, target_col: str, exclude: set[str]
) -> pd.Series:
    cols = _feature_cols(df, target_col, exclude)
    if not cols:
        return pd.Series(dtype="float64")
    return (
        df[cols]
        .corrwith(df[target_col], method="spearman", numeric_only=True)
        .dropna()
        .sort_values(ascending=False)
    )


def compute_univariate_aucs(
    df: pd.DataFrame, *, target_col: str, exclude: set[str]
) -> pd.Series:
    """Orientation-invariant univariate ROC AUC vs binary target (in [0.5, 1])."""
    y = df[target_col]
    binary_ok = is_boolean_like(y)
    warn_if(
        not binary_ok,
        f"Skipping univariate AUC; target '{target_col}' is not boolean-like.",
        logger=LOGGER,
    )
    if not binary_ok:
        return pd.Series(dtype="float64")

    y_ok = y.notna().to_numpy()
    y_int = np.zeros(len(y), dtype=np.int8)
    y_int[y_ok] = y.to_numpy()[y_ok].astype(np.int8, copy=False)

    scores: dict[str, float] = {}
    for col in _feature_cols(df, target_col, exclude):
        x = df[col]
        x_arr = x.to_numpy()
        m = y_ok & pd.notna(x_arr)
        if int(m.sum()) < 10:
            continue
        yv = y_int[m]
        if np.unique(yv).size < 2:
            continue
        xv = x_arr[m]
        if pd.api.types.is_numeric_dtype(x):
            xv = xv.astype(float, copy=False)
        else:
            xv = pd.factorize(xv, sort=True)[0].astype(float)
        if xv.min() == xv.max():
            continue
        try:
            auc = float(roc_auc_score(yv, xv))
        except ValueError:
            continue
        scores[col] = max(auc, 1.0 - auc)

    return pd.Series(scores, dtype="float64").sort_values(ascending=False)


def compute_mutual_infos(
    df: pd.DataFrame,
    *,
    target_col: str,
    exclude: set[str],
    random_state: int | None = 0,
) -> pd.Series:
    """Mutual information vs binary target (higher = more dependence)."""
    binary_ok = is_boolean_like(df[target_col])
    warn_if(
        not binary_ok,
        f"Skipping mutual information; target '{target_col}' is not boolean-like.",
        logger=LOGGER,
    )
    if not binary_ok:
        return pd.Series(dtype="float64")

    cols = _feature_cols(df, target_col, exclude)
    if not cols:
        return pd.Series(dtype="float64")

    discrete = np.zeros(len(cols), dtype=bool)
    data: dict[str, np.ndarray | pd.Series] = {}
    for i, col in enumerate(cols):
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            data[col] = s
        else:
            data[col] = pd.Series(pd.factorize(s, sort=True)[0], index=s.index)
            discrete[i] = True
    data[target_col] = df[target_col]

    mat = pd.DataFrame(data).dropna()
    if len(mat) < 10 or int(mat[target_col].nunique()) < 2:
        return pd.Series(dtype="float64")

    try:
        mi = mutual_info_classif(
            mat[cols].to_numpy(dtype=float),
            mat[target_col].astype("int8").to_numpy(),
            discrete_features=discrete,
            random_state=random_state,
        )
    except ValueError as exc:
        LOGGER.warning("Mutual information failed: %s", exc)
        return pd.Series(dtype="float64")

    return (
        pd.Series(mi, index=cols, dtype="float64")
        .clip(lower=0)
        .sort_values(ascending=False)
    )


def compute_target_assocs(
    df: pd.DataFrame, *, target_col: str, exclude: set[str]
) -> pd.Series:
    """Pairwise associations vs target via :func:`compute_pairwise_associations`."""
    exclude_cols = [c for c in exclude if c != target_col and c in df.columns]
    assocs = compute_pairwise_associations(
        df, ref_col=target_col, exclude_cols=exclude_cols or None
    )[target_col]
    return assocs.dropna().sort_values(ascending=False)


def _top_names(ser: pd.Series, n: int, *, floor: float | None = None) -> list[str]:
    if floor is not None:
        ser = ser[ser >= floor]
    return [str(name) for name in ser[~ser.duplicated(keep="first")].head(n).index]


def suggest_force_include_cols(
    corrs: pd.Series,
    aucs: pd.Series,
    *,
    mis: pd.Series | None = None,
    assocs: pd.Series | None = None,
    exclude: set[str],
    n_each: int = 3,
    skip_cumulative: bool = False,
) -> list[str]:
    """Suggest force_include cols: AUC-first within +/- Spearman groups."""
    mi_s = mis if mis is not None else pd.Series(dtype="float64")
    assoc_s = assocs if assocs is not None else pd.Series(dtype="float64")

    names = {
        n
        for n in set(map(str, corrs.index))
        | set(map(str, aucs.index))
        | set(map(str, mi_s.index))
        if n not in exclude and not (skip_cumulative and n.startswith("cum"))
    }

    best: dict[str, str] = {}
    for name in names:
        fam = feature_family(name)
        cur = best.get(fam)
        if cur is None or (
            float(aucs.get(name, 0.5)),
            float(mi_s.get(name, 0.0)),
            abs(float(corrs.get(name, 0.0))),
            -_name_pref(name),
        ) > (
            float(aucs.get(cur, 0.5)),
            float(mi_s.get(cur, 0.0)),
            abs(float(corrs.get(cur, 0.0))),
            -_name_pref(cur),
        ):
            best[fam] = name

    def rank(name: str) -> float:
        return (
            float(aucs.get(name, 0.5))
            + 0.01 * float(mi_s.get(name, 0.0))
            + 0.001 * abs(float(corrs.get(name, 0.0)))
            + 0.001 * abs(float(assoc_s.get(name, 0.0)))
        )

    pos = pd.Series(
        {n: rank(n) for n in best.values() if float(corrs.get(n, 0.0)) > 0},
        dtype="float64",
    ).sort_values(ascending=False)
    neg = pd.Series(
        {n: rank(n) for n in best.values() if float(corrs.get(n, 0.0)) < 0},
        dtype="float64",
    ).sort_values(ascending=False)
    floor = (0.5 + 0.5 * (float(pos.iloc[0]) - 0.5)) if not pos.empty else None
    return list(
        unique_elements_in_order(
            _top_names(pos, n_each, floor=floor) + _top_names(neg, n_each)
        )
    )


def format_metric_table(
    *series: pd.Series,
    labels: list[str],
    top_n: int = 10,
) -> str:
    """Compact side-by-side top-N table for logging."""
    frames = [
        ser.head(top_n).rename(label)
        for ser, label in zip(series, labels, strict=True)
        if not ser.empty
    ]
    if not frames:
        return "(no metrics)"
    return str(pd.concat(frames, axis=1).to_string(float_format=lambda v: f"{v:.4f}"))
