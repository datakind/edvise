import logging
import typing as t
from collections.abc import Collection

import pandas as pd

from edvise.modeling.feature_families import FEATURE_FAMILIES
from edvise.modeling.performance_selection import (
    PERFORMANCE_CV,
    PERFORMANCE_SCORING,
    fit_model_selector,
    require_feature_engine,
    resolve_binary_target,
    selection_estimator,
)

LOGGER = logging.getLogger(__name__)

# Max fraction of nulls allowed when keeping a feature. Config may set a stricter
# (lower) value; values above this cap are clamped for feature selection.
MAX_INCOMPLETE_THRESHOLD = 0.2
# Compared together so 0/1 matches True/False. Kept separate from labels so "1" is not a copy of 1.
_NUMBER_OR_BOOL = ("number", "boolean")
_CATEGORICAL = ("string", "category", "object")


def select_features(
    df: pd.DataFrame,
    *,
    non_feature_cols: t.Optional[list[str]] = None,
    force_include_cols: t.Optional[list[str]] = None,
    incomplete_threshold: float = MAX_INCOMPLETE_THRESHOLD,
    low_variance_threshold: float = 0.0,
    collinear_threshold: t.Optional[float] = 10.0,
    target_col: str | None = None,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """
    Select features by dropping incomplete, near-constant, and identical
    columns, then dropping numeric columns that shuffling shows are unused.

    Keep-scores (univariate AUC, mutual information, absolute Spearman) are
    computed once on ``df`` and used only to refill an empty feature family.
    Pass the training split so those scores, and the shuffling model, do not
    see test labels.

    Args:
        df: dataframe of features to select. Callers that have a train/test
            split must pass the training rows.
        non_feature_cols: list of column names that are not features and should not
            be run through the feature selection algorithm. For example, demographics, IDs, outcome variables, etc.
        force_include_cols: list of features to force include in the final dataset.
        incomplete_threshold: Threshold for determining incomplete features.
            Values above ``MAX_INCOMPLETE_THRESHOLD`` (0.2) are capped.
        low_variance_threshold: Deprecated and ignored. Kept so existing configs
            still load. A non-zero value logs a warning and does not change selection.
        collinear_threshold: Retained so existing configs still load. Any
            non-null value runs duplicate removal and select-by-shuffling.
            Null skips both. The number is not a VIF cutoff.
        target_col: Binary target used for keep-scores and shuffling.
            When missing, those steps are skipped.
        random_state: Seed for mutual information and the shuffling model.

    Returns:
        df with non_feature_cols, force_include_cols, and any other columns selected
        by the algorithms
    """
    LOGGER.info("selecting features ...")
    non_feature_cols = [c for c in (non_feature_cols or []) if c in df.columns]
    force_include_cols = force_include_cols or []
    seed = 0 if random_state is None else random_state
    aucs, mis, corrs = _keep_metric_series(
        df,
        target_col=target_col,
        exclude=set(non_feature_cols),
        random_state=seed,
    )
    if incomplete_threshold > MAX_INCOMPLETE_THRESHOLD:
        LOGGER.warning(
            "incomplete_threshold=%s exceeds max %s; capping for feature selection",
            incomplete_threshold,
            MAX_INCOMPLETE_THRESHOLD,
        )
        incomplete_threshold = MAX_INCOMPLETE_THRESHOLD
    LOGGER.info(
        "force including %s columns: %s", len(force_include_cols), force_include_cols
    )
    if low_variance_threshold != 0.0:
        LOGGER.warning(
            "low_variance_threshold=%s is deprecated and ignored",
            low_variance_threshold,
        )
    screened = (
        df.drop(columns=non_feature_cols + force_include_cols)
        .pipe(drop_incomplete_features, threshold=incomplete_threshold)
        .pipe(drop_low_variance_features)
    )
    keep = set(screened.columns)
    keep.update(force_include_cols)
    df_selected = _columns_in_order(df, keep)
    if collinear_threshold is not None:
        for col in force_include_cols:
            if col not in df_selected.columns:
                raise ValueError(
                    f"force include column '{col}' was dropped during prior feature selection steps!"
                )
        df_selected = drop_duplicate_features(
            df_selected,
            force_include_cols=force_include_cols,
        )
        target = (
            df[target_col]
            if target_col is not None and target_col in df.columns
            else None
        )
        df_selected = drop_by_shuffling(
            df_selected,
            target_col=target_col,
            target=target,
            force_include_cols=force_include_cols,
            random_state=seed,
        )
    feature_cols = [col for col in df.columns if col not in set(non_feature_cols)]
    df_selected = apply_feature_family_quotas(
        df_selected,
        df,
        candidate_cols=feature_cols,
        aucs=aucs,
        mis=mis,
        corrs=corrs,
    )

    selected_feature_cols = set(df_selected.columns)
    orig_feature_cols = set(df.columns) - set(non_feature_cols)
    LOGGER.info(
        "selected %s out of %s (%s%%) feature columns: %s",
        len(selected_feature_cols),
        len(orig_feature_cols),
        round(100.0 * len(selected_feature_cols) / len(orig_feature_cols), 1),
        selected_feature_cols,
    )
    return _columns_in_order(df, set(non_feature_cols) | selected_feature_cols)


def _columns_in_order(df: pd.DataFrame, names: Collection[str]) -> pd.DataFrame:
    return df.loc[:, [col for col in df.columns if col in names]]


def _bound_scores(
    aucs: pd.Series | None,
    mis: pd.Series | None,
    corrs: pd.Series | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    empty = pd.Series(dtype="float64")
    return (
        empty if aucs is None else aucs,
        empty if mis is None else mis,
        empty if corrs is None else corrs,
    )


def _metric_value(series: pd.Series, column: str, default: float) -> float:
    value = series.get(column, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _keep_key(
    column: str,
    aucs: pd.Series,
    mis: pd.Series,
    corrs: pd.Series,
) -> tuple[float, float, float]:
    """Higher is better: AUC, then mutual information, then absolute Spearman."""
    return (
        _metric_value(aucs, column, 0.5),
        _metric_value(mis, column, 0.0),
        abs(_metric_value(corrs, column, 0.0)),
    )


def _keep_metric_series(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    exclude: set[str],
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """AUC, MI, and Spearman on ``df`` only. The quota step reads these series."""
    empty = pd.Series(dtype="float64")
    if target_col is None or target_col not in df.columns:
        LOGGER.info(
            "skipping keep-score metrics; target column %s is not in the frame",
            target_col,
        )
        return empty, empty, empty

    from edvise.model_prep.target_associations import (
        compute_mutual_infos,
        compute_spearman_corrs,
        compute_univariate_aucs,
    )

    aucs = compute_univariate_aucs(df, target_col=target_col, exclude=exclude)
    mis = compute_mutual_infos(
        df, target_col=target_col, exclude=exclude, random_state=random_state
    )
    corrs = compute_spearman_corrs(df, target_col=target_col, exclude=exclude)
    LOGGER.info(
        "stored keep-scores for %s features (AUC %s, MI %s, Spearman %s)",
        len(set(aucs.index) | set(mis.index) | set(corrs.index)),
        len(aucs),
        len(mis),
        len(corrs),
    )
    return aucs, mis, corrs


def drop_incomplete_features(
    df: pd.DataFrame, *, threshold: float = 0.5
) -> pd.DataFrame:
    """
    Drop columns from dataframe that have at least ``threshold`` fraction of null values.

    Args:
        df
        threshold: Fraction of null values above which columns are deemed "incomplete"
            and dropped from ``df`` .

    Returns:
        ``df`` without the incomplete feature columns
    """
    frac_null = df.isna().sum(axis="index") / df.shape[0]
    incomplete_cols = frac_null.loc[frac_null.ge(threshold)].index.tolist()
    return _drop_logged(
        df,
        incomplete_cols,
        empty=f"no features found with a fraction of null values >= {threshold}",
        found="dropping %s incomplete features: %s",
    )


def drop_low_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns where one value covers at least 99% of rows, on every dtype.

    A column whose most common value is under 99% of rows is kept. Nulls count
    in that row total.
    """
    # Lazy import: inference jobs load this module but do not select features.
    from feature_engine.selection import DropConstantFeatures

    if df.shape[1] == 0:
        return df
    selector = DropConstantFeatures(tol=0.99, missing_values="ignore")
    selector.fit(df)
    return _drop_logged(
        df,
        list(selector.features_to_drop_),
        empty="no near-constant features found",
        found="dropping %s near-constant features: %s",
    )


def drop_duplicate_features(
    df: pd.DataFrame,
    *,
    force_include_cols: t.Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Drop identical features, keeping a force-included column when one is in the group.

    Numeric and boolean columns are compared together, so a boolean that repeats a
    0/1 numeric column is removed. Categorical columns are compared together.
    Several matching force-included columns keep the earliest one in ``df``.
    """
    # Lazy import: inference jobs load this module but do not select features.
    from feature_engine.selection import DropDuplicateFeatures

    forced = set(force_include_cols or [])
    drop_columns: list[str] = []
    described: list[str] = []
    selector = DropDuplicateFeatures(missing_values="ignore")
    for include, cast in (
        (_NUMBER_OR_BOOL, "float64"),
        (_CATEGORICAL, "string"),
    ):
        columns = df.select_dtypes(include=include).columns.tolist()
        if len(columns) < 2:
            continue
        ordered = [col for col in columns if col in forced] + [
            col for col in columns if col not in forced
        ]
        selector.fit(df.loc[:, ordered].astype(cast))
        for group in selector.duplicated_feature_sets_:
            members = [col for col in ordered if col in group]
            kept, extras = members[0], members[1:]
            forced_extras = [col for col in extras if col in forced]
            if kept in forced and forced_extras:
                LOGGER.warning(
                    "Duplicate force-included columns found: keeping '%s', dropping: %s",
                    kept,
                    forced_extras,
                )
            drop_columns.extend(extras)
            described.extend(f"{col} (duplicate of {kept})" for col in extras)

    return _drop_logged(
        df,
        drop_columns,
        empty="no duplicate features found",
        found="dropping %s duplicate features: %s",
        detail=described,
    )


def _drop_logged(
    df: pd.DataFrame,
    columns: list[str],
    *,
    empty: str,
    found: str,
    detail: list[str] | None = None,
) -> pd.DataFrame:
    if not columns:
        LOGGER.info(empty)
        return df
    LOGGER.info(found, len(columns), detail if detail is not None else columns)
    return df.drop(columns=columns)


def drop_by_shuffling(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    target: pd.Series | None = None,
    force_include_cols: t.Optional[Collection[str]] = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Drop numeric features whose shuffled values barely change ROC AUC.

    Uses feature-engine ``SelectByShuffling``. One cross-validated forest is
    trained, then each numeric or boolean column is permuted. ``threshold=None``
    drops columns whose performance drop is below the mean drop. Categorical
    columns are left in place. Force-included columns are scored with the rest
    and put back if the selector drops them.
    """
    y = resolve_binary_target(
        df, target_col=target_col, target=target, step="select by shuffling"
    )
    if y is None:
        return df

    require_feature_engine()
    from feature_engine.selection import SelectByShuffling

    selector = SelectByShuffling(
        estimator=selection_estimator(random_state),
        scoring=PERFORMANCE_SCORING,
        cv=PERFORMANCE_CV,
        threshold=None,
        random_state=random_state,
    )
    return fit_model_selector(
        df,
        y,
        selector,
        force_include_cols=force_include_cols,
        empty="skipping select by shuffling",
        found="dropping %s features that shuffling did not show were useful: %s",
    )


def apply_feature_family_quotas(
    df_selected: pd.DataFrame,
    df_source: pd.DataFrame,
    *,
    candidate_cols: Collection[str],
    aucs: pd.Series | None = None,
    mis: pd.Series | None = None,
    corrs: pd.Series | None = None,
) -> pd.DataFrame:
    """Restore the best dropped member of any quota family that has no survivor."""
    aucs, mis, corrs = _bound_scores(aucs, mis, corrs)
    selected = set(df_selected.columns)
    candidates = set(candidate_cols)
    source_columns = set(df_source.columns)
    restore: list[str] = []
    for family, members in FEATURE_FAMILIES.items():
        present = [
            col for col in members if col in candidates and col in source_columns
        ]
        if not present or any(col in selected for col in present):
            continue
        best = max(present, key=lambda col: _keep_key(col, aucs, mis, corrs))
        restore.append(best)
        LOGGER.info("restoring '%s' so quota family '%s' is not empty", best, family)
    if not restore:
        return df_selected
    return _columns_in_order(df_source, set(df_selected.columns) | set(restore))
