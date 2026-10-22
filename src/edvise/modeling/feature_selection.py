import logging
import typing as t
from collections.abc import Collection

import numpy as np
import pandas as pd
import sklearn.impute
from statsmodels.stats.outliers_influence import variance_inflation_factor

LOGGER = logging.getLogger(__name__)

# Max fraction of nulls allowed when keeping a feature. Config may set a stricter
# (lower) value; values above this cap are clamped for feature selection.
MAX_INCOMPLETE_THRESHOLD = 0.2
# A categorical is ID-like when nearly every non-null value is unique.
_ID_LIKE_NUNIQUE_RATIO = 0.99
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
) -> pd.DataFrame:
    """
    Select features by dropping incomplete features, near-constant features,
    identical features, and variables with high correlations to others.

    Args:
        df: dataframe of features to select
        non_features_cols: list of column names that are not features and should not
            be run through the feature selection algorithm. For example, demographics, IDs, outcome variables, etc.
        force_include_cols: list of features to force include in the final dataset.
        incomplete_threshold: Threshold for determining incomplete features.
            Values above ``MAX_INCOMPLETE_THRESHOLD`` (0.2) are capped.
        low_variance_threshold: Deprecated and ignored. Kept so existing configs
            still load. A non-zero value logs a warning and does not change selection.
        collinear_threshold: Threshold for determining collinear features;
            if null, skip this selection step.

    Returns:
        df with non_features_cols, force_include_cols, and any other columns selected
        by the algorithms
    """
    LOGGER.info("selecting features ...")
    non_feature_cols = [c for c in (non_feature_cols or []) if c in df.columns]
    force_include_cols = force_include_cols or []
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
    df_selected = (
        # we'll add these columns back in later
        df.drop(columns=non_feature_cols + force_include_cols)
        # NOTE: we should create dummy variables for categoricals prior to running this
        # if one of the dummies represents NaN, we won't drop the categoricals that have lots of nulls,
        # but instead can capture any patterns that may relate to the nulls.
        # If there are non-dummy categorical variables that get dropped here,
        # consider pre-processing into dummy variables for this reason.
        .pipe(drop_incomplete_features, threshold=incomplete_threshold)
        # One value in at least 99% of rows, on every dtype.
        .pipe(drop_low_variance_features)
    )
    # Stable order so duplicate removal keeps the earliest column, not a random set member.
    keep = set(df_selected.columns)
    keep.update(force_include_cols)
    df_selected = df.loc[:, [col for col in df.columns if col in keep]]
    if collinear_threshold is not None:
        # multi-collinearity: it may not interfere with the model's performance
        # but it does negatively affect the interpretation of the predictors
        for col in force_include_cols:
            if col not in df_selected.columns:
                raise ValueError(
                    f"force include column '{col}' was dropped during prior feature selection steps!"
                )
        # Literal copies make VIF infinite. Remove them before the VIF loop.
        df_selected = drop_duplicate_features(
            df_selected,
            force_include_cols=force_include_cols,
        )
        df_selected = drop_collinear_features_iteratively(
            df_selected,
            threshold=collinear_threshold,
            force_include_cols=force_include_cols,
        )

    orig_feature_cols = set(df.columns) - set(non_feature_cols)
    selected_feature_cols = set(df_selected.columns)
    keep_cols = set(non_feature_cols) | selected_feature_cols
    LOGGER.info(
        "selected %s out of %s (%s%%) feature columns: %s",
        len(selected_feature_cols),
        len(orig_feature_cols),
        round(100.0 * len(selected_feature_cols) / len(orig_feature_cols), 1),
        selected_feature_cols,
    )
    # maintain the original column order, in case that's meaningful
    return df.loc[:, df.columns.isin(keep_cols)]


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
    if incomplete_cols:
        LOGGER.info(
            "dropping %s incomplete features: %s",
            len(incomplete_cols),
            incomplete_cols,
        )
        return df.drop(columns=incomplete_cols)
    else:
        LOGGER.info("no features found with a fraction of null values >= %s", threshold)
        return df


def drop_low_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns where one value covers at least 99% of rows, on every dtype.

    A column whose most common value is under 99% of rows is kept. Nulls count
    in that row total. Categorical columns with a distinct value for nearly
    every row are flagged and kept.
    """
    # Lazy import: inference jobs load this module but do not select features.
    from feature_engine.selection import DropConstantFeatures

    id_like = _id_like_categorical_cols(df)
    if id_like:
        LOGGER.warning(
            "ID-like categorical columns (nunique ≈ n rows), kept: %s",
            id_like,
        )
    if df.shape[1] == 0:
        return df
    selector = DropConstantFeatures(tol=0.99, missing_values="ignore")
    selector.fit(df)
    drop_cols = list(selector.features_to_drop_)
    if not drop_cols:
        LOGGER.info("no near-constant features found")
        return df
    LOGGER.info("dropping %s near-constant features: %s", len(drop_cols), drop_cols)
    return df.drop(columns=drop_cols)


def _id_like_categorical_cols(df: pd.DataFrame) -> list[str]:
    """Categorical columns where nearly every non-null value is unique."""
    categorical = df.select_dtypes(include=list(_CATEGORICAL))
    if categorical.empty:
        return []
    non_null = categorical.notna().sum()
    nunique = categorical.nunique(dropna=True)
    id_like = (
        (non_null > 1) & (nunique > 1) & (nunique / non_null >= _ID_LIKE_NUNIQUE_RATIO)
    )
    return [str(col) for col in categorical.columns[id_like.fillna(False)]]


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
    duplicate_descriptions: list[str] = []
    selector = DropDuplicateFeatures(missing_values="ignore")
    for include, cast in (
        (_NUMBER_OR_BOOL, "float64"),
        (_CATEGORICAL, "string"),
    ):
        columns = df.select_dtypes(include=include).columns.tolist()
        if len(columns) < 2:
            continue
        # feature-engine keeps the first column of each identical group.
        ordered = [col for col in columns if col in forced]
        ordered.extend(col for col in columns if col not in forced)
        position = {col: i for i, col in enumerate(ordered)}
        selector.fit(df.loc[:, ordered].astype(cast))
        dropped = selector.features_to_drop_
        for group in selector.duplicated_feature_sets_:
            kept = next(iter(group - dropped))
            extras = sorted(group - {kept}, key=lambda col: position[col])
            forced_extras = [col for col in extras if col in forced]
            if kept in forced and forced_extras:
                LOGGER.warning(
                    "Duplicate force-included columns found: keeping '%s', dropping: %s",
                    kept,
                    forced_extras,
                )
            for col in extras:
                drop_columns.append(col)
                duplicate_descriptions.append(f"{col} (duplicate of {kept})")

    if not drop_columns:
        LOGGER.info("no duplicate features found")
        return df
    LOGGER.info(
        "dropping %s duplicate features: %s",
        len(drop_columns),
        duplicate_descriptions,
    )
    return df.drop(columns=drop_columns)


def drop_collinear_features_iteratively(
    df: pd.DataFrame,
    *,
    threshold: float = 10.0,
    force_include_cols: t.Optional[Collection[str]] = None,
) -> pd.DataFrame:
    """
    Use Variance Inflation Factor (VIF) to drop collinear features iteratively.

    The function takes the following steps:
    1. Selects only numeric features for VIF analysis
    2. Impute missing values - this is required by the variance_inflation_factor()
    function. Because we typically rely on default AutoML strategies for imputation,
    we expect to have null values in our data. TODO: we may want to revisit this
    decision if we want more consistent imputation between the VIF analysis and the
    modeling, perhaps deciding to do our own imputation prior to this VIF analysis.
    3. Find the highest VIF. If it is over threshold, it is considered too high and the
    feature is contributing to multicollinearity - we drop this feature.
    4. Repeat step 3 with the remaining features until there are no VIF >= threshold.

    Note: the function used, variance_inflation_factor(), does not take an
    intercept into account. See: https://github.com/statsmodels/statsmodels/issues/2376
    If we want to use a centered VIF rather than an uncentered VIF, we would need to add
    a constant columns to the features dataframe and then re-calculate VIF.

    Troubleshooting: if this hangs, try disabling mlflow logging:
        import mlflow
        mlflow.autolog(disable=True)

    Args:
        df
        threshold: Variance Inflaction Factor (VIF) above which columns are considered
            "multicollinear" and dropped from ``df`` . A VIF of 1 indicates no correlation
            with any other feature; a VIF of 10 is considered "very high" and contributing
            to multicollinearity.
        force_include_cols

    Returns:
        features not considered collinear according to a VIF threshold
    """
    np.seterr(divide="ignore", invalid="ignore")
    force_include_cols = force_include_cols or []

    # Identify numeric and boolean columns
    numeric_df = df.select_dtypes(include=["number"])
    bool_df = df.select_dtypes(include=["boolean"]).astype("Int64")

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

    # calculate initial VIFs for features that aren't force-included
    uncentered_vif_dict = {
        col: variance_inflation_factor(df_features, col_idx)
        for col_idx, col in enumerate(df_features.columns)
        if col not in force_include_cols
    }
    if np.isinf(list(uncentered_vif_dict.values())).all():
        LOGGER.warning(
            "all features are perfectly correlated with one another; "
            "no collinear features will be dropped ..."
        )
        return df

    while (max_vif := max(uncentered_vif_dict.values())) >= threshold:
        highest_vif_cols = [
            col for col, vif in uncentered_vif_dict.items() if vif >= max_vif
        ]
        n_features_dropped_so_far += len(highest_vif_cols)
        LOGGER.info(
            "dropping %s columns with VIF >= %s: %s ...",
            len(highest_vif_cols),
            threshold,
            highest_vif_cols,
        )
        df = df.drop(columns=highest_vif_cols)
        df_features = df_features.drop(columns=highest_vif_cols)

        # recalculate VIFs after dropping columns
        uncentered_vif_dict = {
            col: variance_inflation_factor(df_features, col_idx)
            for col_idx, col in enumerate(df_features.columns)
            if col not in force_include_cols
        }

    LOGGER.info("dropping %s collinear features", n_features_dropped_so_far)

    return df
