import typing as t
import logging

import tempfile
import mlflow

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    log_loss,
)
from sklearn.calibration import calibration_curve
import h2o
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from h2o.estimators.estimator_base import H2OEstimator

from . import training
from . import utils
from . import imputation
from . import calibration
from . import inference

LOGGER = logging.getLogger(__name__)


PosLabelType = t.Union[bool, str]

H2O_FRAMEWORK_DISPLAY_NAMES = {
    "GBM": "Boosted Decision Trees (GBM)",
    "XGBoost": "Boosted Decision Trees (XGBoost)",
    "DRF": "Random Forest (DRF)",
    "XRT": "Random Forest (XRT)",
    "GLM": "Linear Model (GLM)",
}


def get_metrics_fixed_threshold_all_splits(
    model: H2OEstimator,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    target_col: str,
    pos_label: PosLabelType,
    threshold: float = 0.5,
    sample_weight_col: t.Optional[str] = None,
    calibrator: t.Optional[calibration.SklearnCalibratorWrapper] = None,
) -> tuple[t.Dict[str, float], dict]:
    """
    Compute metrics at a fixed threshold (default 0.5) for all splits.
    If `calibrator` is given, metrics use calibrated probabilities.
    If `return_cache=True`, also return the cached per-split arrays to avoid re-predicting.
    """

    def _compute_metrics(frame: H2OFrame, label: str) -> tuple[dict, dict]:
        preds = model.predict(frame)
        prob_col = utils._pick_pos_prob_column(preds, pos_label)

        # Pull data
        df = utils._to_pandas(frame)
        y_true = df[target_col].to_numpy()
        y_prob_raw = utils._to_pandas(preds[prob_col]).to_numpy().reshape(-1)

        # Calibrate if provided
        y_prob = (
            calibrator.transform(y_prob_raw) if calibrator is not None else y_prob_raw
        )

        w = None
        if sample_weight_col and sample_weight_col in df.columns:
            w = df[sample_weight_col].astype(float).fillna(0.0).to_numpy()

        # Unify label space {0,1}
        y_true_bin = utils._binarize_targets(y_true, pos_label)
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics (all in {0,1})
        acc = accuracy_score(y_true_bin, y_pred, sample_weight=w)
        prec = precision_score(y_true_bin, y_pred, sample_weight=w, zero_division=0)
        rec = recall_score(y_true_bin, y_pred, sample_weight=w)
        f1 = f1_score(y_true_bin, y_pred, sample_weight=w)
        auc = roc_auc_score(y_true_bin, y_prob, sample_weight=w)
        ll = log_loss(y_true_bin, y_prob, sample_weight=w, labels=[0, 1])

        tn, fp, fn, tp = confusion_matrix(
            y_true_bin, y_pred, labels=[0, 1], sample_weight=w
        ).ravel()

        metrics = {
            f"{label}_threshold": float(threshold),
            f"{label}_precision": float(prec),
            f"{label}_recall": float(rec),
            f"{label}_accuracy": float(acc),
            f"{label}_f1": float(f1),
            f"{label}_roc_auc": float(auc),
            f"{label}_log_loss": float(ll),
            f"{label}_true_positives": float(tp),
            f"{label}_true_negatives": float(tn),
            f"{label}_false_positives": float(fp),
            f"{label}_false_negatives": float(fn),
        }
        preds = {
            "y_true_bin": np.array(y_true_bin),
            "y_prob_raw": np.array(y_prob_raw),
            "y_prob": np.array(y_prob),
            "weights": np.array(w),
        }
        return metrics, preds

    metrics_train, preds_train = _compute_metrics(train, "train")
    metrics_val, preds_val = _compute_metrics(valid, "validate")
    metrics_test, preds_test = _compute_metrics(test, "test")

    out = {"model_id": model.model_id, **metrics_train, **metrics_val, **metrics_test}
    return out, {"train": preds_train, "validate": preds_val, "test": preds_test}


def generate_all_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    *,
    prefix: str = "test",
    sample_weights: t.Optional[np.ndarray] = None,
) -> None:
    """
    Generates and logs classification plots to MLflow as figures.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_proba: Predicted probabilities for the positive class
        prefix: Prefix for plot file names (e.g., "train", "test", "val")
    """
    plot_fns = {
        "confusion_matrix": (
            lambda yt, yp: create_confusion_matrix_plot(yt, yp, sample_weights),
            y_pred,
        ),
        "roc_curve": (
            lambda yt, pp: create_roc_curve_plot(yt, pp, sample_weights),
            y_proba,
        ),
        "precision_recall": (
            lambda yt, pp: create_precision_recall_curve_plot(yt, pp, sample_weights),
            y_proba,
        ),
        "calibration_curve": (
            lambda yt, pp: create_calibration_curve_plot(yt, pp),
            y_proba,
        ),
    }
    for name, (plot_fn, values) in plot_fns.items():
        fig = plot_fn(y_true, values)
        mlflow.log_figure(fig, f"{prefix}_{name}.png")


def _get_data_run_id(automl_experiment_id: str, data_runname: str) -> str:
    run_df = mlflow.search_runs(
        experiment_ids=[automl_experiment_id], output_format="pandas"
    )
    assert isinstance(run_df, pd.DataFrame)
    # pick the most recent run with this name
    matches = run_df[run_df["tags.mlflow.runName"] == data_runname]
    if matches.empty:
        raise RuntimeError(
            f"No run found with runName={data_runname!r} in experiment {automl_experiment_id}"
        )
    data_run_id = matches.sort_values("start_time", ascending=False).iloc[0]["run_id"]
    return str(data_run_id)


def extract_training_data_from_model(
    automl_experiment_id: str,
    data_runname: str = "H2O AutoML Experiment Summary and Storage",
) -> pd.DataFrame:
    """
    Load the concatenated train/val/test dataset logged by the summary run.
    (Fetches a single file to avoid slow folder listings.)
    """
    data_run_id = _get_data_run_id(automl_experiment_id, data_runname)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_fp = mlflow.artifacts.download_artifacts(
            run_id=data_run_id,
            artifact_path="inputs/full_dataset.parquet",
            dst_path=tmpdir,
        )
        return pd.read_parquet(parquet_fp)


def extract_number_of_runs_from_model_training(
    automl_experiment_id: str,
    data_runname: str = "H2O AutoML Experiment Summary and Storage",
) -> int:
    """
    Count rows in h2o_leaderboard.csv (logged by the summary run).
    (Fetches a single file to avoid slow folder listings.)
    """
    data_run_id = _get_data_run_id(automl_experiment_id, data_runname)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_fp = mlflow.artifacts.download_artifacts(
            run_id=data_run_id,
            artifact_path="leaderboard/h2o_leaderboard.csv",
            dst_path=tmpdir,
        )
        df_leaderboard = pd.read_csv(csv_fp)
        return int(df_leaderboard.shape[0])


############
## PLOTS! ##
############


def create_and_log_h2o_model_comparison(
    aml: H2OAutoML,
    artifact_path: str = "model_comparison.png",
) -> pd.DataFrame:
    """
    Plots best (lowest) logloss per framework using AutoML leaderboard metrics,
    logs the figure to MLflow, and returns the compact DataFrame used for plotting.
    """
    included_frameworks = set(training.VALID_H2O_FRAMEWORKS)

    lb = utils._to_pandas(aml.leaderboard)

    # Ensure there's a 'framework' column
    if "algo" in lb.columns:
        df = lb.rename(columns={"algo": "framework"})
    else:
        df = lb.copy()
        # infer framework by splitting model_id at '_' and taking first token
        df["framework"] = df["model_id"].str.split("_").str[0]

    # Keep only frameworks we trained with
    df = df.loc[
        df["framework"].isin(included_frameworks), ["framework", "logloss"]
    ].dropna()

    # Best (lowest) per family, sorted low→high
    best = (
        df.sort_values("logloss", ascending=True)
        .drop_duplicates(subset=["framework"], keep="first")
        .sort_values("logloss", ascending=True)
        .reset_index(drop=True)
    )

    best["framework_display"] = (
        best["framework"].map(H2O_FRAMEWORK_DISPLAY_NAMES).fillna(best["framework"])
    )

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(best["framework_display"], best["logloss"])

    if len(bars):
        bars[0].set_alpha(1.0)
        for b in bars[1:]:
            b.set_alpha(0.5)
        for i, b in enumerate(bars):
            ax.text(
                b.get_width() * 0.98,
                b.get_y() + b.get_height() / 2,
                f"{best['logloss'].iloc[i]:.4f}",
                va="center",
                ha="right",
            )

    ax.set_xlabel("log_loss")
    ax.set_xlim(left=0)
    ax.invert_yaxis()
    plt.subplots_adjust(left=0.35)
    plt.tight_layout()

    if mlflow.active_run():
        mlflow.log_figure(fig, artifact_path)

    plt.close(fig)
    return best


def create_confusion_matrix_plot(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weights: t.Optional[np.ndarray]
) -> plt.Figure:
    # Normalize confusion matrix by true labels
    cm = confusion_matrix(
        y_true, y_pred, normalize="true", sample_weight=sample_weights
    )

    fig, ax = plt.subplots(figsize=(8, 4.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Normalized Confusion Matrix")

    # Remove default annotations from ConfusionMatrixDisplay
    for txt in ax.texts:
        txt.set_visible(False)

    # Dynamic contrast-aware text overlay (cell values)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "black" if value < 0.5 else "white"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color)

    # --- Side annotations (like your screenshot) ---
    green = "#2ca02c"
    red = "#d62728"

    # Place text in axes coordinates so it stays anchored outside the plot
    # y positions correspond to row centers: row 0 ~ 0.75, row 1 ~ 0.25
    y_row0 = 0.75
    y_row1 = 0.25

    # Left side: interpret by TRUE label rows
    ax.text(
        -0.35,
        y_row0,
        "True Negatives\nDoes Not Need Support;\nCorrectly Classified",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=green,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        -0.35,
        y_row1,
        "False Negatives\nNeeds Support;\nIncorrectly Classified",
        transform=ax.transAxes,
        ha="right",
        va="center",
        color=red,
        fontsize=10,
        fontweight="bold",
    )

    # Right side: interpret by PREDICTED label columns (using the same row y-positions)
    ax.text(
        1.35,
        y_row0,
        "False Positives\nDoes NOT Need Support;\nIncorrectly Classified",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=red,
        fontsize=10,
        fontweight="bold",
    )
    ax.text(
        1.35,
        y_row1,
        "True Positives\nNeeds Support;\nCorrectly Classified",
        transform=ax.transAxes,
        ha="left",
        va="center",
        color=green,
        fontsize=10,
        fontweight="bold",
    )

    # Make room for the outside text
    fig.subplots_adjust(left=0.30, right=0.70)

    plt.tight_layout()
    plt.close(fig)
    return fig


def create_roc_curve_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weights: t.Optional[np.ndarray] = None,
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_proba, sample_weight=sample_weights)
    auc_score = roc_auc_score(y_true, y_proba, sample_weight=sample_weights)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.close(fig)
    return fig


def create_precision_recall_curve_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weights: t.Optional[np.ndarray] = None,
) -> plt.Figure:
    precision, recall, _ = precision_recall_curve(
        y_true, y_proba, sample_weight=sample_weights
    )
    ap_score = average_precision_score(y_true, y_proba, sample_weight=sample_weights)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f"Precision-Recall (AP = {ap_score:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    plt.close(fig)
    return fig


def create_calibration_curve_plot(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> plt.Figure:
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy="quantile"
    )

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(prob_pred, prob_true, marker="o", label="Model Calibration")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # Labels and legend
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.close(fig)
    return fig


def log_roc_table(
    institution_id: str,
    *,
    automl_run_id: str,
    catalog: str = "staging_sst_01",
    target_col: str = "target",
    modeling_df: pd.DataFrame,
    split_col: t.Optional[str] = None,
) -> None:
    """
    Computes and saves an ROC curve table (FPR, TPR, threshold, etc.) for a given H2O model run
    by reloading the test dataset and the trained model.

    Args:
        institution_id (str): Institution ID prefix for table name.
        automl_experiment_id (str): MLflow run ID of the trained model.
        experiment_id
        catalog (str): Destination catalog/schema for the ROC curve table.
    """
    try:
        from databricks.connect import DatabricksSession

        spark = DatabricksSession.builder.getOrCreate()
    except Exception:
        print("⚠️ Databricks Connect failed. Falling back to local Spark.")
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[*]").appName("Fallback").getOrCreate()
        )

    split_col = split_col or "split"

    table_path = f"{catalog}.{institution_id}_silver.training_{automl_run_id}_roc_curve"

    try:
        df = modeling_df
        test_df = df[df[split_col] == "test"].copy()

        # Load and transform using sklearn imputer
        test_df = imputation.SklearnImputerWrapper.load_and_transform(
            test_df,
            run_id=automl_run_id,
        )

        # Load model + features
        model = utils.load_h2o_model(automl_run_id)
        feature_names: t.List[str] = inference.get_h2o_used_features(model)

        # Prepare inputs for ROC
        y_true = test_df[target_col].values
        X_test = test_df[feature_names]
        _, y_scores = inference.predict_h2o(
            X_test,
            model=model,
        )

        # Calculate ROC table manually and plot all thresholds.
        # Down the line, we might want to specify a threshold to reduce plot density
        thresholds = np.sort(np.unique(y_scores))[::-1]
        rounded_thresholds = sorted(
            set([round(t, 4) for t in thresholds]), reverse=True
        )

        P, N = np.sum(y_true == 1), np.sum(y_true == 0)

        rows = []
        for thresh in rounded_thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            TP = np.sum((y_pred == 1) & (y_true == 1))
            FP = np.sum((y_pred == 1) & (y_true == 0))
            TN = np.sum((y_pred == 0) & (y_true == 0))
            FN = np.sum((y_pred == 0) & (y_true == 1))
            TPR = TP / P if P else 0
            FPR = FP / N if N else 0
            rows.append(
                {
                    "threshold": round(thresh, 4),
                    "true_positive_rate": round(TPR, 4),
                    "false_positive_rate": round(FPR, 4),
                    "true_positive": int(TP),
                    "false_positives": int(FP),
                    "true_negatives": int(TN),
                    "false_negatives": int(FN),
                }
            )

        roc_df = pd.DataFrame(rows)
        spark_df = spark.createDataFrame(roc_df)
        spark_df.write.mode("overwrite").saveAsTable(table_path)
        logging.info(
            "ROC table written to table '%s' for run_id=%s", table_path, automl_run_id
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to log ROC table for run {automl_run_id}: {e}"
        ) from e
