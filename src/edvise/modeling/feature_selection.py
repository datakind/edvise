import logging
import math
import typing as t
from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd
import sklearn.impute
from statsmodels.stats.outliers_influence import variance_inflation_factor

from edvise.modeling.feature_families import FEATURE_FAMILIES

LOGGER = logging.getLogger(__name__)

# Max fraction of nulls allowed when keeping a feature. Config may set a stricter
# (lower) value; values above this cap are clamped for feature selection.
MAX_INCOMPLETE_THRESHOLD = 0.2
# |Spearman| / Cramer's V above this are one redundancy group. Matches
# feature-engine SmartCorrelatedSelection's default, which uses a strict ">".
SMART_CORRELATION_THRESHOLD = 0.8
# Compared together so 0/1 matches True/False. Kept separate from labels so "1" is not a copy of 1.
_NUMBER_OR_BOOL = ("number", "boolean")
_BOOL = ("boolean", "bool")
_CATEGORICAL = ("string", "category", "object")
_AUC_TIE_TOLERANCE = 1e-6


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
    correlation_threshold: float = SMART_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """
    Select features by dropping incomplete features, near-constant features,
    identical features, pairwise-redundant features, and variables with high
    variance inflation.

    Keep-scores (univariate AUC, mutual information, absolute Spearman) are
    computed once on ``df``. Pass the training split so those scores, and the
    smart-correlation fit, do not see test labels.

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
        collinear_threshold: Variance-inflation threshold. If null, skip duplicate
            removal, smart correlation, and VIF.
        target_col: Binary target used for keep-scores and smart correlation.
            When missing, those steps are skipped and VIF falls back to dropping
            the highest inflation factor.
        random_state: Seed for mutual information and the smart-correlation model.
        correlation_threshold: Absolute Spearman / Cramer's V above which a pair
            is redundant. Feature-engine groups pairs strictly above this value.

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
        df_selected = drop_smart_correlated_features(
            df_selected,
            target_col=target_col,
            target=target,
            aucs=aucs,
            mis=mis,
            corrs=corrs,
            force_include_cols=force_include_cols,
            threshold=correlation_threshold,
            random_state=seed,
        )
        df_selected = drop_collinear_features_iteratively(
            df_selected,
            threshold=collinear_threshold,
            force_include_cols=force_include_cols,
            aucs=aucs,
            mis=mis,
            corrs=corrs,
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


def _as_score_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype="float64")
    return values


def _metric_value(series: pd.Series, column: str, default: float) -> float:
    value = series.get(column, default)
    if value is None or pd.isna(value):
        return default
    return float(value)


def _keep_key(
    column: str, aucs: pd.Series, mis: pd.Series, corrs: pd.Series
) -> tuple[float, float, float]:
    """Higher is better: AUC, then mutual information, then |Spearman|."""
    return (
        _metric_value(aucs, column, 0.5),
        _metric_value(mis, column, 0.0),
        abs(_metric_value(corrs, column, 0.0)),
    )


def _categorical_keep_key(
    column: str, aucs: pd.Series, mis: pd.Series, corrs: pd.Series
) -> tuple[float, float, float]:
    """Higher is better for labels: mutual information, then AUC, then |Spearman|."""
    mi, auc, spearman = (
        _metric_value(mis, column, 0.0),
        _metric_value(aucs, column, 0.5),
        abs(_metric_value(corrs, column, 0.0)),
    )
    return (mi, auc, spearman)


def _keep_metric_series(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    exclude: set[str],
    random_state: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """AUC, MI, and Spearman on ``df`` only. Later steps read these series."""
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


def drop_smart_correlated_features(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    target: pd.Series | None = None,
    aucs: pd.Series | None = None,
    mis: pd.Series | None = None,
    corrs: pd.Series | None = None,
    force_include_cols: t.Optional[Collection[str]] = None,
    threshold: float = SMART_CORRELATION_THRESHOLD,
    random_state: int = 0,
) -> pd.DataFrame:
    """
    Drop pairwise-redundant numeric features, then redundant categoricals.

    Numeric and boolean columns go through feature-engine
    ``SmartCorrelatedSelection(method="spearman", selection_method="model_performance",
    scoring="roc_auc")`` with a random forest, which this feature-engine version
    requires for ``model_performance``. The package has no force-include flag, so
    force-included columns are scored with the rest. ``target`` is the label
    aligned to ``df``; it is not read from ``df`` when the column was already
    removed. After ``fit``, repair puts a force-include back when the package
    dropped it, and breaks univariate-AUC ties with mutual information then
    absolute Spearman. Categorical columns skip the package and keep the higher-MI
    column of a Cramer's V group.
    """
    aucs = _as_score_series(aucs)
    mis = _as_score_series(mis)
    corrs = _as_score_series(corrs)
    threshold = float(threshold)
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError(
            f"correlation_threshold must be in (0, 1]. Got {threshold} instead."
        )
    # The target is not a feature, so it has usually already been removed from ``df``.
    y = target
    if y is None and target_col is not None and target_col in df.columns:
        y = df[target_col]
    if y is None:
        LOGGER.info(
            "skipping smart correlation selection; target column %s is unavailable",
            target_col,
        )
        return df
    if not y.index.equals(df.index):
        y = y.reindex(df.index)

    from edvise.shared.validation import is_boolean_like

    if not is_boolean_like(y):
        LOGGER.warning(
            "skipping smart correlation selection; target '%s' is not boolean-like",
            target_col,
        )
        return df

    df = _drop_smart_correlated_numeric(
        df,
        y=y,
        aucs=aucs,
        mis=mis,
        corrs=corrs,
        force_include_cols=force_include_cols,
        threshold=threshold,
        random_state=random_state,
    )
    return drop_redundant_categoricals(
        df,
        aucs=aucs,
        mis=mis,
        corrs=corrs,
        force_include_cols=force_include_cols,
        threshold=threshold,
    )


def repair_smart_correlated_groups(
    correlated_feature_dict: Mapping[str, Collection[str]],
    *,
    aucs: pd.Series | None = None,
    mis: pd.Series | None = None,
    corrs: pd.Series | None = None,
    force_include_cols: t.Optional[Collection[str]] = None,
) -> list[str]:
    """
    Decide which members of each smart-correlation group to drop.

    ``correlated_feature_dict`` is feature-engine's ``correlated_feature_dict_``:
    the key is the column the package kept, and the values are the columns it
    would drop. A force-included member is kept instead. When two or more members
    share the best univariate AUC and the package's choice is one of them, the
    keep-score (AUC, mutual information, absolute Spearman) picks the survivor.
    A unique AUC winner chosen by the package is left alone.
    """
    aucs = _as_score_series(aucs)
    mis = _as_score_series(mis)
    corrs = _as_score_series(corrs)
    forced = set(force_include_cols or [])
    drop: list[str] = []
    for kept, others in correlated_feature_dict.items():
        group = [kept, *sorted(col for col in others if col != kept)]
        forced_members = [col for col in group if col in forced]
        if forced_members:
            rejected = [col for col in group if col not in forced]
            if len(forced_members) > 1:
                LOGGER.warning(
                    "Correlated force-included columns %s; keeping all of them "
                    "and dropping %s",
                    forced_members,
                    rejected,
                )
            elif rejected and kept not in forced:
                LOGGER.info(
                    "SCS repair puts force-included '%s' back and drops '%s'",
                    forced_members[0],
                    kept,
                )
            drop.extend(rejected)
            continue

        auc_values = {col: _metric_value(aucs, col, 0.5) for col in group}
        best_auc = max(auc_values.values())
        tied = [
            col
            for col in group
            if math.isclose(
                auc_values[col], best_auc, rel_tol=0.0, abs_tol=_AUC_TIE_TOLERANCE
            )
        ]
        if len(tied) >= 2 and kept in tied:
            winner = max(tied, key=lambda col: _keep_key(col, aucs, mis, corrs))
            if winner != kept:
                LOGGER.info(
                    "SCS repair keeps '%s' over '%s' after an AUC tie in %s",
                    winner,
                    kept,
                    group,
                )
            drop.extend(col for col in group if col != winner)
        else:
            drop.extend(col for col in group if col != kept)
    return list(dict.fromkeys(drop))


def drop_redundant_categoricals(
    df: pd.DataFrame,
    *,
    aucs: pd.Series | None = None,
    mis: pd.Series | None = None,
    corrs: pd.Series | None = None,
    force_include_cols: t.Optional[Collection[str]] = None,
    threshold: float = SMART_CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Keep the higher-MI column of each Cramer's V group. Numeric columns stay."""
    aucs = _as_score_series(aucs)
    mis = _as_score_series(mis)
    corrs = _as_score_series(corrs)
    columns = df.select_dtypes(include=_CATEGORICAL).columns.tolist()
    if len(columns) < 2:
        LOGGER.info("no categorical pairs to compare with Cramer's V")
        return df

    from edvise.data_audit.eda import _cramers_v

    groups = _cramers_v_groups(
        df.loc[:, columns].astype("string"),
        threshold=float(threshold),
        association=_cramers_v,
    )
    forced = set(force_include_cols or [])
    drop_columns: list[str] = []
    described: list[str] = []
    for group in groups:
        forced_members = [col for col in group if col in forced]
        if forced_members:
            rejected = [col for col in group if col not in forced]
            kept = forced_members
        else:
            winner = max(
                group, key=lambda col: _categorical_keep_key(col, aucs, mis, corrs)
            )
            kept = [winner]
            rejected = [col for col in group if col != winner]
        drop_columns.extend(rejected)
        described.extend(f"{col} (Cramer's V group of {kept[0]})" for col in rejected)
    return _drop_logged(
        df,
        drop_columns,
        empty="no redundant categorical features found",
        found="dropping %s redundant categorical features: %s",
        detail=described,
    )


def _cramers_v_groups(
    df: pd.DataFrame,
    *,
    threshold: float,
    association: t.Callable[[pd.Series, pd.Series], float | None],
) -> list[list[str]]:
    columns = df.columns.tolist()
    parent = {col: col for col in columns}

    def find(col: str) -> str:
        while parent[col] != col:
            parent[col] = parent[parent[col]]
            col = parent[col]
        return col

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for i, left in enumerate(columns):
        for right in columns[i + 1 :]:
            stat = association(df[left], df[right])
            if stat is not None and stat > threshold:
                union(left, right)

    grouped: dict[str, list[str]] = {}
    for col in columns:
        grouped.setdefault(find(col), []).append(col)
    return [members for members in grouped.values() if len(members) > 1]


def _drop_smart_correlated_numeric(
    df: pd.DataFrame,
    *,
    y: pd.Series,
    aucs: pd.Series,
    mis: pd.Series,
    corrs: pd.Series,
    force_include_cols: t.Optional[Collection[str]],
    threshold: float,
    random_state: int,
) -> pd.DataFrame:
    from feature_engine.selection import SmartCorrelatedSelection
    from sklearn.ensemble import RandomForestClassifier

    columns = df.select_dtypes(include=["number", *_BOOL]).columns.tolist()
    columns = [col for col in columns if bool(df[col].notna().any())]
    if len(columns) < 2:
        LOGGER.info(
            "skipping smart correlation selection; fewer than 2 numeric columns"
        )
        return df

    X = _imputed_numeric_frame(df.loc[:, columns])
    labeled = y.notna()
    X = X.loc[labeled].replace([np.inf, -np.inf], np.nan).dropna(axis="index")
    y_fit = y.loc[X.index].astype(int)
    class_counts = y_fit.value_counts()
    if len(X) < 10 or class_counts.size != 2 or int(class_counts.min()) < 3:
        LOGGER.warning(
            "skipping smart correlation selection; need at least 10 rows and "
            "3 per class to score roc_auc, got %s rows and class counts %s",
            len(X),
            class_counts.to_dict(),
        )
        return df

    selector = SmartCorrelatedSelection(
        method="spearman",
        threshold=threshold,
        missing_values="ignore",
        selection_method="model_performance",
        estimator=RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        ),
        scoring="roc_auc",
        cv=3,
    )
    selector.fit(X, y_fit)
    drop_columns = [
        col
        for col in repair_smart_correlated_groups(
            selector.correlated_feature_dict_,
            aucs=aucs,
            mis=mis,
            corrs=corrs,
            force_include_cols=force_include_cols,
        )
        if col in df.columns
    ]
    return _drop_logged(
        df,
        drop_columns,
        empty="no smart-correlated numeric features found",
        found="dropping %s smart-correlated numeric features: %s",
    )


def _imputed_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mean-impute numbers and mode-impute booleans so a model can score them."""
    numeric_df = df.select_dtypes(include=["number"])
    bool_df = df.select_dtypes(include=list(_BOOL))
    frames: list[pd.DataFrame] = []
    if not numeric_df.empty:
        imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
            missing_values=np.nan, strategy="mean"
        ).set_output(transform="pandas")  # type: ignore
        imputed = imputer.fit_transform(numeric_df)
        assert isinstance(imputed, pd.DataFrame)
        frames.append(imputed)
    if not bool_df.empty:
        bool_imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
            missing_values=np.nan, strategy="most_frequent"
        ).set_output(transform="pandas")  # type: ignore
        imputed_bool = bool_imputer.fit_transform(bool_df.astype("float64"))
        assert isinstance(imputed_bool, pd.DataFrame)
        frames.append(imputed_bool)
    if not frames:
        return pd.DataFrame(index=df.index)
    return pd.concat(frames, axis=1).astype("float64")


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
    aucs = _as_score_series(aucs)
    mis = _as_score_series(mis)
    corrs = _as_score_series(corrs)
    selected = set(df_selected.columns)
    candidates = set(candidate_cols)
    restore: list[str] = []
    for family, members in FEATURE_FAMILIES.items():
        present = [
            col for col in members if col in candidates and col in df_source.columns
        ]
        if not present or any(col in selected for col in present):
            continue
        best = max(present, key=lambda col: _keep_key(col, aucs, mis, corrs))
        restore.append(best)
        LOGGER.info("restoring '%s' so quota family '%s' is not empty", best, family)
    if not restore:
        LOGGER.info("no quota family needed a feature restored")
        return df_selected
    return _columns_in_order(df_source, set(df_selected.columns) | set(restore))


def drop_collinear_features_iteratively(
    df: pd.DataFrame,
    *,
    threshold: float = 10.0,
    force_include_cols: t.Optional[Collection[str]] = None,
    aucs: pd.Series | None = None,
    mis: pd.Series | None = None,
    corrs: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Use Variance Inflation Factor (VIF) to drop collinear features iteratively.

    The function takes the following steps:
    1. Selects only numeric and boolean features for VIF analysis. Categorical
       columns are left in place.
    2. Impute missing values - this is required by the variance_inflation_factor()
    function. Because we typically rely on default AutoML strategies for imputation,
    we expect to have null values in our data. TODO: we may want to revisit this
    decision if we want more consistent imputation between the VIF analysis and the
    modeling, perhaps deciding to do our own imputation prior to this VIF analysis.
    3. Among columns whose VIF is still >= ``threshold``, drop the one with the
    worst keep-score (lowest AUC, then mutual information, then absolute Spearman).
    When those scores tie, drop the higher VIF. Force-included columns are never
    dropped. One column is removed, then VIF is recomputed.
    4. Repeat step 3 until no non-forced column has VIF >= threshold, or only a
    force-included column is left.

    Note: the function used, variance_inflation_factor(), does not take an
    intercept into account. See: https://github.com/statsmodels/statsmodels/issues/2376
    If we want to use a centered VIF rather than an uncentered VIF, we would need to add
    a constant columns to the features dataframe and then re-calculate VIF.

    Troubleshooting: if this hangs, try disabling mlflow logging:
        import mlflow
        mlflow.autolog(disable=True)

    Args:
        df
        threshold: Variance Inflation Factor (VIF) above which columns are considered
            "multicollinear" and dropped from ``df`` . A VIF of 1 indicates no correlation
            with any other feature; a VIF of 10 is considered "very high" and contributing
            to multicollinearity.
        force_include_cols
        aucs: Univariate AUC from the training frame. Missing scores count as 0.5.
        mis: Mutual information from the training frame. Missing scores count as 0.
        corrs: Spearman correlation with the target. The absolute value is used.

    Returns:
        features not considered collinear according to a VIF threshold
    """
    np.seterr(divide="ignore", invalid="ignore")
    force_include_cols = force_include_cols or []
    aucs = _as_score_series(aucs)
    mis = _as_score_series(mis)
    corrs = _as_score_series(corrs)

    # Identify numeric and boolean columns
    numeric_df = df.select_dtypes(include=["number"])
    bool_df = df.select_dtypes(include=list(_BOOL)).astype("Int64")

    if numeric_df.empty and bool_df.empty:
        LOGGER.warning("no numeric columns found, so no collinear features to drop")
        return df

    # Impute numeric features with mean because VIF function can't handle nulls
    imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
        missing_values=np.nan, strategy="mean"
    ).set_output(transform="pandas")  # type: ignore
    df_num_imputed = imputer.fit_transform(numeric_df)
    assert isinstance(df_num_imputed, pd.DataFrame)

    df_features = df_num_imputed.copy()

    n_features_dropped_so_far = 0

    if not bool_df.empty:
        bool_imputer: sklearn.impute.SimpleImputer = sklearn.impute.SimpleImputer(
            missing_values=np.nan, strategy="most_frequent"
        ).set_output(transform="pandas")  # type: ignore

        df_bool_imputed = bool_imputer.fit_transform(bool_df)

        assert isinstance(df_bool_imputed, pd.DataFrame)

        df_features = pd.concat([df_features, df_bool_imputed], axis=1)

    forced = set(force_include_cols)
    warned_infinite = False
    while True:
        if df_features.shape[1] < 2:
            LOGGER.info(
                "stopping VIF with %s column(s) left",
                df_features.shape[1],
            )
            break
        uncentered_vif_dict = {
            col: variance_inflation_factor(df_features, col_idx)
            for col_idx, col in enumerate(df_features.columns)
            if col not in forced
        }
        if not uncentered_vif_dict:
            LOGGER.info("only force-included columns remain; stopping VIF")
            break
        vif_values = list(uncentered_vif_dict.values())
        if not warned_infinite and np.isinf(vif_values).all():
            LOGGER.warning(
                "all non-forced features have infinite VIF; "
                "dropping the weakest keep-score one at a time"
            )
            warned_infinite = True
        eligible = {
            col: vif
            for col, vif in uncentered_vif_dict.items()
            if pd.notna(vif) and float(vif) >= threshold
        }
        if not eligible:
            break
        victim = min(
            eligible,
            key=lambda col: (
                *_keep_key(col, aucs, mis, corrs),
                -float(eligible[col]),
                -int(df_features.columns.get_loc(col)),
            ),
        )
        n_features_dropped_so_far += 1
        LOGGER.info(
            "dropping '%s' with VIF %s (lowest keep-score among VIF >= %s)",
            victim,
            eligible[victim],
            threshold,
        )
        df = df.drop(columns=[victim])
        df_features = df_features.drop(columns=[victim])

    LOGGER.info("dropping %s collinear features", n_features_dropped_so_far)

    return df
