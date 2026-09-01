import logging

import re
import typing as t

import pandas as pd
import numpy as np
import numpy.typing as npt

import edvise.dataio as dataio
from edvise.modeling.features_table_mapping import map_feature_col_for_features_table

LOGGER = logging.getLogger(__name__)

DEFAULT_DISPLAY_DECIMALS = 2
DEFAULT_INDICATOR_COLUMN_LABEL = "Indicator"
SUPPORT_SCORE_COL = "Support Score"
MAX_STUDENTS_FEATURES_WITH_MOST_IMPACT = 4000


def _format_display_number(
    value: float, decimals: int = DEFAULT_DISPLAY_DECIMALS
) -> str:
    """Format a number as a fixed-decimal string for display."""
    return f"{round(float(value), decimals):.{decimals}f}"


def _as_excel_csv_text(value: object) -> str:
    """
    Wrap ``value`` as an Excel/Sheets text formula so CSV apps left-align the cell.

    Plain strings like ``\"0.90\"`` are still auto-detected as numbers (right-aligned).
    ``=\"0.90\"`` forces text. Leading tabs are stripped if present from older exports.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        text = ""
    else:
        text = str(value)
        if text.startswith("\t"):
            text = text[1:]
        # Already formula-wrapped from a prior pass.
        if text.startswith('="') and text.endswith('"'):
            return text
    escaped = text.replace('"', '""')
    return f'="{escaped}"'


def format_dataframe_for_excel_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with every cell wrapped for left-aligned Excel/Sheets CSV text."""
    if df.empty:
        return df.copy()
    return df.apply(lambda col: col.map(_as_excel_csv_text))


def select_top_features_for_display(
    features: pd.DataFrame,
    unique_ids: pd.Series,
    predicted_probabilities: list[float],
    shap_values: npt.NDArray[np.float64],
    n_features: int = 3,
    needs_support_threshold_prob: t.Optional[float] = None,
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
    schema_type: str | None = None,
    *,
    support_score_decimals: int = DEFAULT_DISPLAY_DECIMALS,
    importance_decimals: int = DEFAULT_DISPLAY_DECIMALS,
    column_label: str = DEFAULT_INDICATOR_COLUMN_LABEL,
    sort_by_support_score: bool = True,
    format_numerics_as_strings: bool = True,
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
        needs_support_threshold_prob: Deprecated and ignored. Retained for call-site
            compatibility; the "Support Needed" column is no longer emitted.
        features_table: Optional mapping of column to human-friendly feature name/desc,
            loaded via :func:`utils.load_features_table()`
        support_score_decimals: Decimal places for Support Score display.
        importance_decimals: Decimal places for indicator importance display.
        column_label: Prefix for per-rank columns (e.g. ``Indicator_1_Name``).
        sort_by_support_score: If True, sort rows by Support Score descending.
        format_numerics_as_strings: If True, emit Support Score and importance as
            fixed-decimal strings.

    Returns:
        explainability dataframe for display

    TODO: refactor this functionality so it's vectorized and aggregates by student
    """
    del needs_support_threshold_prob  # deprecated; Support Needed column removed

    pred_probs = np.asarray(predicted_probabilities, dtype=float)
    feature_columns = features.columns.to_numpy()
    features_values = features.to_numpy()

    top_features_info = []
    for i, (unique_id, predicted_proba) in enumerate(zip(unique_ids, pred_probs)):
        instance_shap_values = shap_values[i]
        top_indices = np.argsort(-np.abs(instance_shap_values))[:n_features]
        top_features = feature_columns[top_indices]
        top_feature_values = features_values[i, top_indices]
        top_shap_values = instance_shap_values[top_indices]

        student_output: dict[str, t.Any] = {
            "Student ID": unique_id,
            SUPPORT_SCORE_COL: float(predicted_proba),
        }

        for feature_rank, (feature, feature_value, shap_value) in enumerate(
            zip(top_features, top_feature_values, top_shap_values), start=1
        ):
            feature_name = (
                _get_mapped_feature_name(
                    feature, features_table, schema_type=schema_type
                )
                if features_table is not None
                else feature
            )
            if isinstance(feature_value, (float, np.floating)) and not isinstance(
                feature_value, (bool, np.bool_)
            ):
                feature_value_display = str(round(float(feature_value), 2))
            else:
                feature_value_display = str(feature_value)
            student_output |= {
                f"{column_label}_{feature_rank}_Name": feature_name,
                f"{column_label}_{feature_rank}_Value": feature_value_display,
                f"{column_label}_{feature_rank}_Importance": float(shap_value),
            }

        top_features_info.append(student_output)

    df = pd.DataFrame(top_features_info)
    if df.empty:
        return df

    if sort_by_support_score:
        df = df.sort_values(
            by=SUPPORT_SCORE_COL, ascending=False, kind="mergesort"
        ).reset_index(drop=True)

    importance_cols = [
        c
        for c in df.columns
        if c.startswith(f"{column_label}_") and c.endswith("_Importance")
    ]
    if format_numerics_as_strings:
        df[SUPPORT_SCORE_COL] = df[SUPPORT_SCORE_COL].map(
            lambda v: _format_display_number(v, support_score_decimals)
        )
        for col in importance_cols:
            df[col] = df[col].map(
                lambda v: _format_display_number(v, importance_decimals)
            )
        df["Student ID"] = df["Student ID"].map(str)
    else:
        df[SUPPORT_SCORE_COL] = df[SUPPORT_SCORE_COL].round(support_score_decimals)
        for col in importance_cols:
            df[col] = df[col].round(importance_decimals)

    return df


def generate_ranked_feature_table(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
    metadata: bool = True,
    original_dtypes: t.Optional[dict[str, t.Any]] = None,
    schema_type: str | None = None,
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
                schema_type=schema_type,
            )
        else:
            mapped = feature if not metadata else (feature, None, None)

        if metadata:
            readable_feature_name, short_feature_desc, long_feature_desc = mapped
            feature_name = readable_feature_name
        else:
            feature_name = mapped

        dtype = features[feature].dtype

        # Prefer the pre-imputation dtype so display-only string sentinels do
        # not change numeric or boolean features to "Categorical".
        orig_dtype_raw = original_dtypes.get(feature, None) if original_dtypes else None
        orig_dtype = None
        if orig_dtype_raw is not None:
            try:
                orig_dtype = pd.api.types.pandas_dtype(orig_dtype_raw)
            except (TypeError, ValueError):
                pass

        type_source = orig_dtype if orig_dtype is not None else dtype
        if pd.api.types.is_bool_dtype(type_source):
            data_type = "Boolean"
        elif pd.api.types.is_numeric_dtype(type_source):
            data_type = "Continuous"
        else:
            data_type = "Categorical"

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


def _lookup_features_table_entry(
    feature_col: str,
    features_table: dict[str, dict[str, str]],
    *,
    schema_type: str | None = None,
) -> tuple[dict[str, str], re.Match[str] | None] | None:
    """Return ``(entry, regex_match)`` for an exact or regex features-table key."""
    feature_col = feature_col.lower()
    mapped_col = map_feature_col_for_features_table(feature_col, schema_type)
    candidates = [mapped_col]
    if mapped_col != feature_col:
        candidates.append(feature_col)

    for candidate in candidates:
        if candidate in features_table:
            return features_table[candidate], None
        for fkey, fval in features_table.items():
            if "(" in fkey and ")" in fkey:
                if match := re.fullmatch(fkey, candidate):
                    return fval, match
    return None


def is_feature_defined_in_table(
    feature_col: str,
    features_table: dict[str, dict[str, str]],
    *,
    schema_type: str | None = None,
) -> bool:
    """True when ``feature_col`` matches an exact or regex key in ``features_table``."""
    return (
        _lookup_features_table_entry(
            feature_col, features_table, schema_type=schema_type
        )
        is not None
    )


def _get_mapped_feature_name(
    feature_col: str,
    features_table: dict[str, dict[str, str]],
    metadata: bool = False,
    schema_type: str | None = None,
) -> t.Any:
    feature_col = feature_col.lower()  # just in case

    def _descs(entry: dict[str, str]) -> tuple[t.Optional[str], t.Optional[str]]:
        # Keep original keys first; allow new keys if present
        short_desc = entry.get("short_desc", entry.get("short_feature_desc"))
        long_desc = entry.get("long_desc", entry.get("long_feature_desc"))
        return short_desc, long_desc

    if lookup := _lookup_features_table_entry(
        feature_col, features_table, schema_type=schema_type
    ):
        entry, match = lookup
        feature_name = entry["name"].format(*match.groups()) if match else entry["name"]
        if metadata:
            short_desc, long_desc = _descs(entry)
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
    schema_type: str | None = None,
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
                _get_mapped_feature_name(
                    feature, features_table, metadata=True, schema_type=schema_type
                )
            )
        )

    top_features["feature_value"] = top_features["feature_value"].astype(str)

    return top_features


def sample_features_with_most_impact_students(
    df: pd.DataFrame,
    *,
    max_students: int = MAX_STUDENTS_FEATURES_WITH_MOST_IMPACT,
    student_id_col: str = "student_id",
    random_state: int | None = 42,
) -> pd.DataFrame:
    """
    Limit ``features_with_most_impact`` rows for webapp display.

    When the cohort has more than ``max_students`` unique students, randomly
    sample that many students and keep every top-feature row for them (e.g.
    4,000 students × 10 features → 40,000 rows). Smaller cohorts are unchanged.
    """
    if student_id_col not in df.columns:
        raise ValueError(f"Missing required column {student_id_col!r}")

    unique_students = df[student_id_col].drop_duplicates()
    if len(unique_students) <= max_students:
        return df

    sampled_ids = unique_students.sample(
        n=max_students,
        random_state=random_state,
        replace=False,
    )
    return df.loc[df[student_id_col].isin(sampled_ids)].copy()


def top_feature_boxstats(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table: t.Optional[dict[str, dict[str, str]]] = None,
    schema_type: str | None = None,
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
                _get_mapped_feature_name(
                    feature, features_table, metadata=True, schema_type=schema_type
                )
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
    schema_type: str | None = None,
) -> pd.DataFrame:
    """
    Bin support scores for histogram display.

    Args:
        df_serving (pd.DataFrame): Unused; retained for call-site compatibility.
        unique_ids (pd.Series): Unused; retained for call-site compatibility.
        pred_probs (list or np.ndarray): Predicted probabilities from the model.
        shap_values (np.ndarray or pd.DataFrame): Unused; retained for call-site compatibility.
        inference_params (dict): Unused; retained for call-site compatibility.
        features_table (dict): Unused; retained for call-site compatibility.
        schema_type: Unused; retained for call-site compatibility.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - bin_lower: Lower bound of the support score bin.
            - bin_upper: Upper bound of the support score bin.
            - support_score: Midpoint of the bin (used for plotting).
            - count_of_students: Number of students in the bin.
            - pct: Percentage of total students in the bin.

    """
    _ = (
        df_serving,
        unique_ids,
        shap_values,
        inference_params,
        features_table,
        schema_type,
    )

    try:
        # Bin raw probabilities for histogram (display formatting is handled separately).
        bin_width = 0.2 / 5  # 0.04
        bins = np.arange(0.0, 1.0 + bin_width, bin_width)  # 0.00 ... 1.00

        counts, bin_edges = np.histogram(np.asarray(pred_probs, dtype=float), bins=bins)

        bin_lower = bin_edges[:-1]
        bin_upper = bin_edges[1:]
        support_score = (bin_lower + bin_upper) / 2
        pct = counts / counts.sum()

        return pd.DataFrame(
            {
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "support_score": support_score,
                "count_of_students": counts,
                "pct": pct,
            }
        )

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
    schema_type: str | None = None,
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
            schema_type=schema_type,
        )
        return top_n_shap_features
    except Exception as e:
        logging.error("Error computing top %d shap features table: %s", n, e)
        raise  # keep the signature honest


def features_box_whiskers_table(
    features: pd.DataFrame,
    shap_values: npt.NDArray[np.float64],
    features_table_path: str,
    schema_type: str | None = None,
) -> pd.DataFrame:
    features_table = dataio.read.read_features_table(features_table_path)
    try:
        feature_boxstats = top_feature_boxstats(
            features=features,
            shap_values=shap_values,
            features_table=features_table,
            schema_type=schema_type,
        )
        return feature_boxstats

    except Exception as e:
        logging.error("Error computing box features %d shap features table: %s", e)
        return None
