import logging

import re
import typing as t

import pandas as pd
import numpy as np
import numpy.typing as npt

import edvise.dataio as dataio

LOGGER = logging.getLogger(__name__)


def select_top_features_for_display(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    predicted_probabilities: list[float],
    shap_values: npt.NDArray[np.float64],
    n_features: int = 3,
    needs_support_threshold_prob: t.Optional[float] = 0.5,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Select most important features from SHAP for each student
    and format for display

    Args:
        features: features used in modeling
        unique_ids: student IDs, of length ``features.shape[0]``
        predicted_probabilities: predicted probabilities for each student, in the same
            order as unique_ids, of shape len(unique_ids)
        shap_values: array of arrays of SHAP values, of shape len(unique_ids)
        n_features: number of important features to return
        needs_support_threshold_prob: Minimum probability in [0.0, 1.0] used to compute
            a boolean "needs support" field added to output records. Values in
            ``predicted_probabilities`` greater than or equal to this threshold result in
            a True value, otherwise it's False; if this threshold is set to null,
            then no "needs support" values are added to the output records.
            Note that this doesn't have to be the "optimal" decision threshold for
            the trained model that produced ``predicted_probabilities`` , it can
            be tailored to a school's preferences and use case.
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`

    Returns:
        explainability dataframe for display

    TODO: refactor this functionality so it's vectorized and aggregates by student
    """
    pred_probs = np.asarray(predicted_probabilities)

    top_features_info = []
    for i, (unique_id, predicted_proba) in enumerate(zip(unique_ids, pred_probs)):
        instance_shap_values = shap_values[i]
        top_indices = np.argsort(-np.abs(instance_shap_values))[:n_features]
        top_features = features.columns[top_indices]
        top_feature_values = features.iloc[i][top_features]
        top_shap_values = instance_shap_values[top_indices]

        student_output = {
            "Student ID": unique_id,
            "Support Score": predicted_proba,
        }
        if needs_support_threshold_prob is not None:
            student_output["Support Needed"] = (
                predicted_proba >= needs_support_threshold_prob
            )

        for feature_rank, (feature, feature_value, shap_value) in enumerate(
            zip(top_features, top_feature_values, top_shap_values), start=1
        ):
            feature_name = (
                _get_mapped_feature_name(feature, features_table)
                if features_table is not None
                else feature
            )
            feature_value = (
                str(round(feature_value, 2))
                if isinstance(feature_value, float)
                else str(feature_value)
            )
            student_output |= {
                f"Feature_{feature_rank}_Name": feature_name,
                f"Feature_{feature_rank}_Value": feature_value,
                f"Feature_{feature_rank}_Importance": round(shap_value, 2),
            }

        top_features_info.append(student_output)
    return pd.DataFrame(top_features_info)


def generate_ranked_feature_table(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
    metadata: bool = True,
    original_dtypes: t.Optional[dict[str, t.Any]] = None,
) -> pd.DataFrame:
    """
    Creates a table of all selected features of the model ranked
    by average SHAP magnitude (aka feature importance). We utilize average
    SHAP magnitude & an absolute value because it removes directionality
    from the SHAP values and focuses specifically on importance. This table
    is used in the model cards to provide a comprehensive summary of the model's
    features.

    Args:
        features: feature data used in modeling where columns are the feature
            column names
        shap_values: array of arrays of SHAP values, of shape len(unique_ids)
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`
        metadata: whether to return short desc and long desc along with name in
            features table (applicable only to pdp)
        original_dtypes: Optional dictionary mapping feature names to their original
            dtypes before sklearn processing. Used to correctly classify boolean features
            that may have been converted to numeric types.

    Returns:
        A ranked pandas DataFrame by average shap magnitude
    """
    feature_metadata = []

    for idx, feature in enumerate(features.columns):
        if features_table is not None:
            mapped = _get_mapped_feature_name(
                feature_col=feature,
                features_table=features_table,
                metadata=metadata,
            )
        else:
            mapped = feature if not metadata else (feature, None, None)

        if metadata:
            readable_feature_name, short_feature_desc, long_feature_desc = mapped
            feature_name = readable_feature_name
        else:
            feature_name = mapped

        dtype = features[feature].dtype
        
        # Check original_dtypes first to correctly identify boolean features
        # that may have been converted to numeric during sklearn processing
        orig_dtype_raw = original_dtypes.get(feature, None) if original_dtypes else None
        if orig_dtype_raw is not None:
            try:
                orig_dtype = pd.api.types.pandas_dtype(orig_dtype_raw)
            except Exception:
                orig_dtype = None
        else:
            orig_dtype = None
        
        is_bool_from_original = (
            orig_dtype is not None and pd.api.types.is_bool_dtype(orig_dtype)
        )
        
        data_type = (
            "Boolean"
            if (pd.api.types.is_bool_dtype(dtype) or is_bool_from_original)
            else "Continuous"
            if pd.api.types.is_numeric_dtype(dtype)
            else "Categorical"
        )

        avg_shap_magnitude_raw = np.mean(np.abs(shap_values[:, idx]))

        row = {
            "feature_name": feature,
            "readable_feature_name": feature_name,
            "data_type": data_type,
            "average_shap_magnitude_raw": avg_shap_magnitude_raw,
        }

        if metadata:
            row["short_feature_desc"] = short_feature_desc
            row["long_feature_desc"] = long_feature_desc

        feature_metadata.append(row)

    df = (
        pd.DataFrame(feature_metadata)
        .sort_values(by="average_shap_magnitude_raw", ascending=False)
        .reset_index(drop=True)
    )

    df["average_shap_magnitude"] = (
        df["average_shap_magnitude_raw"]
        .apply(lambda x: "<0.0000" if round(x, 4) == 0 else round(x, 4))
        .astype(str)
    )

    return df.drop(columns=["average_shap_magnitude_raw"])


def _get_mapped_feature_name(
    feature_col: str, features_table: dict[str, dict[str, str]], metadata: bool = False
) -> t.Any:
    feature_col = feature_col.lower()  # just in case

    def _descs(entry: dict[str, str]) -> tuple[t.Optional[str], t.Optional[str]]:
        # Keep original keys first; allow new keys if present
        short_desc = entry.get("short_desc", entry.get("short_feature_desc"))
        long_desc = entry.get("long_desc", entry.get("long_feature_desc"))
        return short_desc, long_desc

    if feature_col in features_table:
        entry = features_table[feature_col]
        feature_name = entry["name"]
        if metadata:
            short_desc, long_desc = _descs(entry)
            return feature_name, short_desc, long_desc
        return feature_name
    else:
        for fkey, fval in features_table.items():
            if "(" in fkey and ")" in fkey:
                if match := re.fullmatch(fkey, feature_col):
                    feature_name = fval["name"].format(*match.groups())
                    if metadata:
                        short_desc, long_desc = _descs(fval)
                        return feature_name, short_desc, long_desc
                    return feature_name

    try:
        for _, fval in features_table.items():
            nm = fval.get("name")
            if nm and nm.strip().lower() == feature_col:
                if metadata:
                    short_desc, long_desc = _descs(fval)
                    return nm, short_desc, long_desc
                return nm
    except Exception:
        # Swallow any unexpected issues to preserve old behavior
        pass

    feature_name = feature_col
    if metadata:
        return feature_name, None, None
    return feature_name


def top_shap_features(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    shap_values: npt.NDArray[np.float64],
    top_n: int = 10,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Extracts the top N most important SHAP features across all samples.

    Args:
        features (pd.DataFrame): Input feature values.
        unique_ids (pd.Series): Unique identifiers for each sample.
        shap_values (np.ndarray): SHAP values for the input features.
        top_n (int): Number of top features to select (default is 10).
        features_table (dict, optional): Mapping of feature names to human-readable names.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns:
            - student_id
            - feature_name
            - shap_value
            - feature_value
    """

    if features.empty or shap_values.size == 0 or unique_ids.empty:
        raise ValueError("Input data cannot be empty.")

    shap_long = (
        pd.DataFrame(shap_values, columns=features.columns)
        .assign(student_id=unique_ids.values)
        .melt(id_vars="student_id", var_name="feature_name", value_name="shap_value")
    )

    feature_long = features.assign(student_id=unique_ids.values).melt(
        id_vars="student_id", var_name="feature_name", value_name="feature_value"
    )

    summary_df = shap_long.merge(feature_long, on=["student_id", "feature_name"])

    top_n_features = (
        summary_df.groupby("feature_name")["shap_value"]
        .apply(lambda x: np.mean(np.abs(x)))
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    top_features = summary_df[summary_df["feature_name"].isin(top_n_features)].copy()

    if features_table is not None:
        top_features[
            ["feature_readable_name", "feature_short_desc", "feature_long_desc"]
        ] = top_features["feature_name"].apply(
            lambda feature: pd.Series(
                _get_mapped_feature_name(feature, features_table, metadata=True)
            )
        )

    top_features["feature_value"] = top_features["feature_value"].astype(str)

    return top_features


def top_feature_boxstats(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Per-feature summary for the GLOBAL top-N features (by mean |SHAP|).
    Returns min, Q1, median, Q3, max suitable for box/whisker plotting,
    along with mean absolute SHAP for reference.
    """
    if features.empty or shap_values.size == 0:
        raise ValueError("Input data cannot be empty.")
    if shap_values.shape != (features.shape[0], features.shape[1]):
        raise ValueError(
            f"shap_values shape {shap_values.shape} must match features shape {features.shape}"
        )

    mean_abs = pd.Series(np.mean(np.abs(shap_values), axis=0), index=features.columns)
    top_feats = mean_abs.sort_values(ascending=False)

    # Restrict stats to numeric columns
    stats_source = features.select_dtypes(include=[np.number])

    rows = []
    for feat in top_feats.index:
        if feat not in stats_source.columns:
            # Non-numeric top feature — include row with NaN stats, but correct counts.
            rows.append(
                {
                    "feature_name": feat,
                    "feature_shap_value": float(top_feats[feat]),
                    "min": np.nan,
                    "Q1": np.nan,
                    "median": np.nan,
                    "Q3": np.nan,
                    "max": np.nan,
                    "count": int(features[feat].notna().sum()),
                    "n_missing": int(features[feat].isna().sum()),
                }
            )
            continue

        col = stats_source[feat]
        rows.append(
            {
                "feature_name": feat,
                "feature_shap_value": float(top_feats[feat]),
                "min": float(col.min()),
                "Q1": float(col.quantile(0.25, interpolation="linear")),
                "median": float(col.quantile(0.5, interpolation="linear")),
                "Q3": float(col.quantile(0.75, interpolation="linear")),
                "max": float(col.max()),
                "count": int(col.notna().sum()),
                "n_missing": int(col.isna().sum()),
            }
        )

    feature_boxstats = (
        pd.DataFrame(rows)
        .sort_values("feature_shap_value", ascending=False)
        .reset_index(drop=True)
    )
    if features_table is not None:
        feature_boxstats[
            ["feature_readable_name", "feature_short_desc", "feature_long_desc"]
        ] = feature_boxstats["feature_name"].apply(
            lambda feature: pd.Series(
                _get_mapped_feature_name(feature, features_table, metadata=True)
            )
        )
    return feature_boxstats


def support_score_distribution_table(
    df_serving: pd.DataFrame,
    unique_ids: t.Any,
    pred_probs: t.Any,
    shap_values: t.Any,
    inference_params: dict,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
) -> pd.DataFrame:
    """
    Selects top SHAP features for each student, and bins the support scores.

    Args:
        df_serving (pd.DataFrame): Input features used for prediction.
        unique_ids (pd.Series): Unique ids (student_id) for each student.
        pred_probs (list or np.ndarray): Predicted probabilities from the model.
        shap_values (np.ndarray or pd.DataFrame): SHAP values for the input features.
        inference_params (dict): Dictionary containing configuration for:
            - "num_top_features" (int): Number of top features to display.
            - "min_prob_pos_label" (float): Threshold to determine if support is needed.
        features_table (dict): Optional dictionary mapping feature names to understandable format.
        model_feature_names (list): List of feature names used by the model.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - bin_lower: Lower bound of the support score bin.
            - bin_upper: Upper bound of the support score bin.
            - support_score: Midpoint of the bin (used for plotting).
            - count_of_students: Number of students in the bin.
            - pct: Percentage of total students in the bin.

    """

    try:
        result = select_top_features_for_display(
            features=df_serving,
            unique_ids=unique_ids,
            predicted_probabilities=pred_probs,
            shap_values=shap_values.values,
            n_features=inference_params["num_top_features"],
            needs_support_threshold_prob=inference_params["min_prob_pos_label"],
            features_table=features_table,
        )

        # --- Bin support scores for histogram (e.g., 0.0 to 1.0 in 0.1 steps) ---
        bin_width = 0.2 / 5  # 0.04
        bins = np.arange(0.0, 1.0 + bin_width, bin_width)  # 0.00 ... 1.00

        counts, bin_edges = np.histogram(result["Support Score"], bins=bins)

        bin_lower = bin_edges[:-1]
        bin_upper = bin_edges[1:]
        support_score = (bin_lower + bin_upper) / 2
        pct = counts / counts.sum()

        bin_summary = pd.DataFrame(
            {
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "support_score": support_score,
                "count_of_students": counts,
                "pct": pct,
            }
        )

        return bin_summary

    except Exception:
        import traceback

        traceback.print_exc()
        raise  # <-- temporarily raise instead of returning None


def top_n_features(
    grouped_features: pd.DataFrame,
    unique_ids: pd.Series,
    grouped_shap_values: npt.NDArray[np.float64] | pd.DataFrame,  # relax input
    features_table_path: str,
    n: int = 10,
) -> pd.DataFrame:
    features_table = dataio.read.read_features_table(features_table_path)
    try:
        top_n_shap_features = top_shap_features(
            features=grouped_features,
            unique_ids=unique_ids,
            shap_values=(
                grouped_shap_values.values
                if isinstance(grouped_shap_values, pd.DataFrame)
                else grouped_shap_values
            ),
            top_n=n,
            features_table=features_table,
        )
        return top_n_shap_features
    except Exception as e:
        logging.error("Error computing top %d shap features table: %s", n, e)
        raise  # keep the signature honest


def features_box_whiskers_table(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table_path: str,
) -> pd.DataFrame:
    features_table = dataio.read.read_features_table(features_table_path)
    try:
        feature_boxstats = top_feature_boxstats(
            features=features,
            shap_values=shap_values,
            features_table=features_table,
        )
        return feature_boxstats

    except Exception as e:
        logging.error("Error computing box features %d shap features table: %s", e)
        return None
