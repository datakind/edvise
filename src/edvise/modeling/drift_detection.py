"""Minimal numeric feature drift detection for inference monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import typing as t

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    ks_stat: float
    p_value: float


def compute_numeric_ks_drift(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame,
    *,
    feature_columns: t.Sequence[str] | None = None,
) -> list[FeatureDriftResult]:
    """
    Compare numeric feature distributions (training reference vs inference) via KS test.

    Numeric columns are MinMax-scaled using the training split fit, matching legacy
    notebook behavior. Categorical features are not evaluated.
    """
    if feature_columns is not None:
        cols = [
            c
            for c in feature_columns
            if c in df_train.columns and c in df_infer.columns
        ]
        train = df_train.loc[:, cols]
        infer = df_infer.loc[:, cols]
    else:
        train = df_train
        infer = df_infer

    numeric_cols = train.select_dtypes(include="number").columns.intersection(
        infer.select_dtypes(include="number").columns
    )
    if len(numeric_cols) == 0:
        return []

    train_numeric = train.loc[:, numeric_cols]
    infer_numeric = infer.loc[:, numeric_cols]

    scaler = MinMaxScaler()
    scaler.fit(train_numeric)

    train_scaled = pd.DataFrame(
        scaler.transform(train_numeric),
        columns=train_numeric.columns,
        index=train_numeric.index,
    )
    infer_scaled = pd.DataFrame(
        scaler.transform(infer_numeric),
        columns=infer_numeric.columns,
        index=infer_numeric.index,
    )

    results: list[FeatureDriftResult] = []
    for col in infer_scaled.columns:
        d, p = ks_2samp(train_scaled[col], infer_scaled[col])
        results.append(
            FeatureDriftResult(feature=col, ks_stat=float(d), p_value=float(p))
        )
    return results


def log_numeric_ks_drift(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame,
    *,
    top_n: int = 20,
    feature_columns: t.Sequence[str] | None = None,
    context: str = "model drift detection",
    logger: logging.Logger | None = None,
) -> list[FeatureDriftResult]:
    """Run :func:`compute_numeric_ks_drift` and log the top features by KS statistic."""
    log = logger or LOGGER
    results = compute_numeric_ks_drift(
        df_train, df_infer, feature_columns=feature_columns
    )
    if not results:
        log.info("[%s] No numeric features to compare.", context)
        return results

    sorted_results = sorted(results, key=lambda r: r.ks_stat, reverse=True)
    n = min(top_n, len(sorted_results))
    log.info(
        "[%s] Top %d numeric features by KS statistic (categorical not evaluated):",
        context,
        n,
    )
    for result in sorted_results[:n]:
        log.info(
            "  %-30s  KS=%.4f  p=%.4e",
            result.feature,
            result.ks_stat,
            result.p_value,
        )
    return sorted_results
