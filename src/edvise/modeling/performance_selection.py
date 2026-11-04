"""Shared fit path for model-performance feature selectors.

Recursive elimination and select-by-shuffling both need one imputed numeric
frame, one forest, and the same force-include repair. Each caller supplies
the feature-engine selector.
"""

from __future__ import annotations

import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd
import sklearn.impute

LOGGER = logging.getLogger(__name__)

# Pinned with pyproject.toml. Both selectors below record features_to_drop_.
FEATURE_ENGINE_VERSION = "1.9.4"
PERFORMANCE_SCORING = "roc_auc"
PERFORMANCE_CV = 3
_BOOL = ("boolean", "bool")


def selection_estimator(random_state: int) -> t.Any:
    """Shallow forest shared by the performance selectors.

    Elimination and shuffling refit this estimator once per feature. Depth and
    leaf size keep that loop cheap and stable. ``n_jobs=1`` keeps the fit
    deterministic for a given ``random_state``.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        min_samples_leaf=5,
        random_state=random_state,
        n_jobs=1,
    )


def require_feature_engine() -> None:
    import feature_engine

    found = feature_engine.__version__
    if found != FEATURE_ENGINE_VERSION:
        raise RuntimeError(
            "feature selection requires feature-engine "
            f"{FEATURE_ENGINE_VERSION}, found {found}"
        )


def resolve_binary_target(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    target: pd.Series | None,
    step: str,
) -> pd.Series | None:
    """Return a boolean label aligned to ``df``, or None when selection should skip."""
    y = target
    if y is None and target_col is not None and target_col in df.columns:
        y = df[target_col]
    if y is None:
        LOGGER.info("skipping %s; target column %s is unavailable", step, target_col)
        return None
    if not y.index.equals(df.index):
        y = y.reindex(df.index)

    from edvise.shared.validation import is_boolean_like

    if not is_boolean_like(y):
        LOGGER.warning(
            "skipping %s; target '%s' is not boolean-like",
            step,
            target_col,
        )
        return None
    return y


def fit_model_selector(
    df: pd.DataFrame,
    y: pd.Series,
    selector: t.Any,
    *,
    force_include_cols: Collection[str] | None,
    empty: str,
    found: str,
) -> pd.DataFrame:
    """Fit ``selector`` on numeric and boolean columns and drop what it rejects.

    Categorical columns stay in ``df``. A force-included column the selector
    drops is put back. Returns ``df`` unchanged when the frame cannot support
    a ROC-AUC cross-validation fit.
    """
    prepared = _labeled_numeric_frame(df, y)
    if prepared is None:
        LOGGER.info(empty)
        return df
    features, y_fit = prepared
    selector.fit(features, y_fit)
    rejected = [
        column for column in list(selector.features_to_drop_) if column in df.columns
    ]
    return _drop_except_forced(
        df,
        rejected,
        force_include_cols=force_include_cols,
        found=found,
    )


def _labeled_numeric_frame(
    df: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series] | None:
    columns = df.select_dtypes(include=["number", *_BOOL]).columns.tolist()
    columns = [column for column in columns if bool(df[column].notna().any())]
    if len(columns) < 2:
        LOGGER.info("skipping model selection; fewer than 2 numeric columns")
        return None

    features = _imputed_numeric_bool(df.loc[:, columns])
    labeled = y.notna()
    features = (
        features.loc[labeled].replace([np.inf, -np.inf], np.nan).dropna(axis="index")
    )
    y_fit = y.loc[features.index]
    class_counts = y_fit.astype(int).value_counts()
    if len(features) < 10 or class_counts.size != 2 or int(class_counts.min()) < 3:
        LOGGER.warning(
            "skipping model selection; need at least 10 rows and 3 per class "
            "to score roc_auc, got %s rows and class counts %s",
            len(features),
            class_counts.to_dict(),
        )
        return None
    return features, y_fit.astype(int)


def _drop_except_forced(
    df: pd.DataFrame,
    columns: list[str],
    *,
    force_include_cols: Collection[str] | None,
    found: str,
) -> pd.DataFrame:
    forced = set(force_include_cols or [])
    kept = [column for column in columns if column in forced]
    if kept:
        LOGGER.info(
            "keeping force-included columns the selector dropped: %s",
            kept,
        )
    drop = [column for column in columns if column not in forced]
    if not drop:
        LOGGER.info("no model-rejected features to drop")
        return df
    LOGGER.info(found, len(drop), drop)
    return df.drop(columns=drop)


def _impute(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
        missing_values=np.nan, strategy=strategy
    ).set_output(transform="pandas")  # type: ignore[call-arg]
    imputed = imputer.fit_transform(frame)
    assert isinstance(imputed, pd.DataFrame)
    return imputed


def _imputed_numeric_bool(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-impute numbers and mode-impute booleans. Other columns are omitted."""
    frames: list[pd.DataFrame] = []
    numeric = df.select_dtypes(include=["number"])
    boolean = df.select_dtypes(include=list(_BOOL))
    if not numeric.empty:
        frames.append(_impute(numeric, "mean"))
    if not boolean.empty:
        frames.append(_impute(boolean.astype("float64"), "most_frequent"))
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1)
