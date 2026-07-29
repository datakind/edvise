"""Minimal numeric feature drift detection for inference monitoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass
import typing as t

import pandas as pd
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

LOGGER = logging.getLogger(__name__)

DRIFT_REPORT_COLUMNS: tuple[str, ...] = (
    "feature",
    "train_mean",
    "infer_mean",
    "mean_delta",
    "train_median",
    "infer_median",
    "median_delta",
    "train_std",
    "infer_std",
    "train_null_pct",
    "infer_null_pct",
    "ks_stat",
    "p_value",
)


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    ks_stat: float
    p_value: float


def _select_numeric_feature_frames(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame,
    *,
    feature_columns: t.Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
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
    return train.loc[:, numeric_cols], infer.loc[:, numeric_cols], numeric_cols


def _scale_numeric_frames(
    train_numeric: pd.DataFrame, infer_numeric: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return train_scaled, infer_scaled


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
    train_numeric, infer_numeric, numeric_cols = _select_numeric_feature_frames(
        df_train, df_infer, feature_columns=feature_columns
    )
    if len(numeric_cols) == 0:
        return []

    train_scaled, infer_scaled = _scale_numeric_frames(train_numeric, infer_numeric)

    results: list[FeatureDriftResult] = []
    for col in infer_scaled.columns:
        d, p = ks_2samp(train_scaled[col], infer_scaled[col])
        results.append(
            FeatureDriftResult(feature=col, ks_stat=float(d), p_value=float(p))
        )
    return results


def build_numeric_drift_report_df(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame,
    *,
    feature_columns: t.Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Build a per-feature drift report comparing training reference vs inference cohorts.

    Summary statistics (mean, median, std, null %) are computed on the imputed feature
    values passed in. KS statistics use MinMax scaling fit on the training split.
    """
    train_numeric, infer_numeric, numeric_cols = _select_numeric_feature_frames(
        df_train, df_infer, feature_columns=feature_columns
    )
    if len(numeric_cols) == 0:
        return pd.DataFrame(columns=list(DRIFT_REPORT_COLUMNS))

    ks_by_feature = {
        r.feature: r
        for r in compute_numeric_ks_drift(
            df_train, df_infer, feature_columns=list(numeric_cols)
        )
    }

    rows: list[dict[str, t.Any]] = []
    for col in numeric_cols:
        train_s = pd.to_numeric(train_numeric[col], errors="coerce")
        infer_s = pd.to_numeric(infer_numeric[col], errors="coerce")
        train_mean = float(train_s.mean())
        infer_mean = float(infer_s.mean())
        train_median = float(train_s.median())
        infer_median = float(infer_s.median())
        ks = ks_by_feature[col]
        rows.append(
            {
                "feature": col,
                "train_mean": train_mean,
                "infer_mean": infer_mean,
                "mean_delta": infer_mean - train_mean,
                "train_median": train_median,
                "infer_median": infer_median,
                "median_delta": infer_median - train_median,
                "train_std": float(train_s.std()),
                "infer_std": float(infer_s.std()),
                "train_null_pct": float(train_s.isna().mean()),
                "infer_null_pct": float(infer_s.isna().mean()),
                "ks_stat": ks.ks_stat,
                "p_value": ks.p_value,
            }
        )

    return (
        pd.DataFrame(rows, columns=list(DRIFT_REPORT_COLUMNS))
        .sort_values("ks_stat", ascending=False)
        .reset_index(drop=True)
    )


def log_numeric_ks_drift(
    df_train: pd.DataFrame,
    df_infer: pd.DataFrame,
    *,
    top_n: int = 20,
    feature_columns: t.Sequence[str] | None = None,
    context: str = "model drift detection",
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Log top drifting features and return the full drift report DataFrame."""
    log = logger or LOGGER
    report = build_numeric_drift_report_df(
        df_train, df_infer, feature_columns=feature_columns
    )
    if report.empty:
        log.info("[%s] No numeric features to compare.", context)
        return report

    n = min(top_n, len(report))
    log.info(
        "[%s] Top %d numeric features by KS statistic (categorical not evaluated):",
        context,
        n,
    )
    for row in report.head(n).itertuples(index=False):
        log.info(
            "  %-30s  KS=%.4f  p=%.4e  mean_delta=%+.4f",
            row.feature,
            row.ks_stat,
            row.p_value,
            row.mean_delta,
        )
    return report
