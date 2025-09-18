import yaml
import logging
import typing as t

import os
import datetime
import tempfile
import contextlib
import random

import mlflow
from mlflow.models import Model, infer_signature
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from pandas.api.types import (
    is_categorical_dtype,
    is_object_dtype,
    is_string_dtype,
    is_integer_dtype,
)

import h2o
from h2o.automl import H2OAutoML
from h2o.model.model_base import ModelBase
from h2o.frame import H2OFrame
from h2o.two_dim_table import H2OTwoDimTable

from sklearn.metrics import confusion_matrix

from . import evaluation
from . import imputation

LOGGER = logging.getLogger(__name__)


def safe_h2o_init(base_port: int = 54321, mem_per_cluster: str = "4G") -> None:
    """
    Initialize a unique H2O cluster per Databricks task (or randomly if no task id).
    Hardens the server by binding to localhost so REST endpoints (incl. /3/ImportFiles)
    are not reachable from the network.
    """
    task_id = os.environ.get("DATABRICKS_TASK_RUN_ID")
    if task_id:
        port = base_port + (int(task_id) % 10000)
    else:
        port = base_port + random.randint(0, 1000)

    LOGGER.info(f"Starting H2O cluster at 127.0.0.1:{port} (localhost only)...")
    h2o.init(
        ip="127.0.0.1",
        port=port,
        nthreads=-1,
        max_mem_size=mem_per_cluster,
        bind_to_localhost=True,  # restrict server to local machine
    )

    # Safety check: verify we really connected to localhost
    conn = h2o.connection()
    base_url = getattr(conn, "base_url", "")
    if not ("127.0.0.1" in base_url or "localhost" in base_url):
        LOGGER.warning(
            "H2O is not bound to localhost. This may expose REST endpoints on the network."
        )
    # Stop H2O progress bars globally
    h2o.no_progress()


def download_artifact_file(
    run_id: str, artifact_path: str, dst_dir: str | None = None
) -> str:
    """
    Download a single artifact file (e.g., 'model/model.h2o') and return its local path.
    This avoids slow directory listings in GCS/DBFS.
    """
    if dst_dir is None:
        dst_dir = tempfile.mkdtemp()
    # mlflow will create subdirs as needed; returns the local file path
    return mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path=artifact_path,
        dst_path=dst_dir,
    )


def download_model_artifact(run_id: str, artifact_subdir: str = "model") -> str:
    """
    Back-compat wrapper that now downloads ONLY model.h2o instead of the whole folder.
    Returns the local file path to model.h2o.
    """
    return download_artifact_file(run_id, f"{artifact_subdir}/model.h2o")


def load_h2o_model(
    run_id: str, artifact_path: str = "model"
) -> h2o.model.model_base.ModelBase:
    """
    Initializes H2O and loads the model by downloading a single file:
    artifact_path/model.h2o
    """
    if not h2o.connection():
        safe_h2o_init()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_model_file = download_artifact_file(
            run_id, f"{artifact_path}/model.h2o", tmp_dir
        )
        if not os.path.exists(local_model_file):
            raise FileNotFoundError(
                f"Expected model.h2o not found at {local_model_file}"
            )
        return h2o.load_model(local_model_file)


def get_cv_logloss_stats(model) -> t.Tuple[t.Optional[float], t.Optional[float]]:
    """Return CV logloss mean and STD, if available.

    Tries the cross-validation summary table first (column names vary across H2O
    versions), then falls back to aggregating per-fold metrics. If CV is disabled
    or stats are unavailable, returns (None, None).

    Args:
      model: H2O model (e.g., from AutoML leaderboard).

    Returns:
      Tuple[Optional[float], Optional[float]]: (cv_mean, cv_std) for logloss, or
      (None, None) if CV was not enabled with H2O training configuration.
    """
    try:
        summ = model.cross_validation_metrics_summary()
        df = summ.as_data_frame() if hasattr(summ, "as_data_frame") else summ
        if isinstance(df, pd.DataFrame):
            cols = [str(c).strip().lower() for c in df.columns]
            df.columns = cols
            key = (
                "name"
                if "name" in cols
                else ("metric" if "metric" in cols else cols[0])
            )
            mean = (
                "mean" if "mean" in cols else ("value" if "value" in cols else cols[1])
            )
            # accept sd / stddev / std (and fall back safely)
            if "sd" in cols:
                std = "sd"
            elif "stddev" in cols:
                std = "stddev"
            elif "std" in cols:
                std = "std"
            else:
                std = cols[min(2, len(cols) - 1)]
            row = df[df[key].astype(str).str.lower().eq("logloss")]
            if not row.empty:
                cv_mean = float(row[mean].iloc[0])
                cv_std = float(row[std].iloc[0])
                if np.isfinite(cv_mean) and np.isfinite(cv_std) and cv_std > 0:
                    return cv_mean, cv_std
    except Exception:
        pass

    # Fallback to per fold
    try:
        vals = [
            m.logloss()
            for m in getattr(model, "cross_validation_metrics", lambda: [])()
            if hasattr(m, "logloss")
        ]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if len(vals) >= 2:
            return float(np.mean(vals)), float(np.std(vals, ddof=1))
        if len(vals) == 1:
            return float(vals[0]), None
    except Exception:
        pass

    return None, None


def compute_overfit_score_logloss(
    model, train: H2OFrame, test: H2OFrame, valid: t.Optional[H2OFrame] = None
) -> dict:
    """Compute an overfit score between [0, 1] from logloss.

    Emphasizes generalization:
      - Generalization vs expectation (primary): penalizes when test logloss is
      materially worse than the CV mean (test - CV_mean), scaled by CV_std.
      - Capacity overfit to train: penalizes when train looks much better than
      the CV mean (CV_mean - train), scaled by CV_std.
      - If H2O's training config doesn't include CV: we then penalize based on large test vs. train gaps.
      - (Optional) If validation dataset is provided, reports a symmetric validation vs. test
      instability metric for dashboards. If a model struggles with validation vs. test, that shows it may
      generalize inconsistently.

    We utilize tolerances since fluctuations are normal and they should help avoid flagging noise.
      - For test vs CV and CV vs train: `min(cap, FRAC_OF_CV_STD * CV_std)`.
      - For test vs train when CV is off: `min(GAP_CAP_train, FRAC_OF_CV_STD * FALLBACK_STD)`.

    Both are made adaptive to the dataset's CV spread, so they scale naturally
    across datasets. If CV is not available, we fall back to a conservative scale.

    The returned `overfit.score` is capped in [0, 1] for automated model selection.

    Parameters:
      model: H2O model to score.
      train : H2OFrame for the training split.
      valid (Optional): H2OFrame for the validation split.
      test: H2OFrame for the test split.

    Returns:
      dict:
        - overfit.score (float [0,1]): higher ⇒ greater overfit risk.
        - overfit.std_excess (float): max z-like excess among the risk terms.
        - overfit.flag (str): 'green' | 'yellow' | 'red' (based on std_excess vs STD_MAX).
        - delta.test_train (float): test vs. train.
        - (optional) delta.test_cv (float): test vs. CV_mean
        - (optional) delta.cv_train (float): CV_mean vs. train
        - (optional) cv.logloss_mean (float), cv.logloss_std (float)
        - (optional) instability.score, instability.sd_gap if validation dataset is provided.
    """
    # Tolerance constants
    TOL_CV_TEST: float = 0.03  # tolerance for test vs CV mean
    TOL_CV_TRAIN: float = 0.1  # tolerance for test vs train and CV vs train
    FRAC_OF_CV_STD: float = (
        0.25  # ensures tolerance scales with CV std: min(cap, frac*cv_std)
    )
    FALLBACK_STD: float = 0.08  # we treat this as “one STD” when CV is off
    STD_MAX: float = 1.5  # setting maximum STD so overfit score is capped at 1

    # Threshold-free split losses
    train_log_loss = float(model.model_performance(train).logloss())
    test_log_loss = float(model.model_performance(test).logloss())
    if valid is not None:
        valid_log_loss = float(model.model_performance(valid).logloss())

    # CV context
    cv_mean, cv_std = get_cv_logloss_stats(model)
    cv_ok = (
        (cv_mean is not None)
        and (cv_std is not None)
        and np.isfinite(cv_std)
        and cv_std > 0
    )

    # Deltas
    delta_test_train = test_log_loss - train_log_loss
    delta_test_cv = None if not cv_ok else (test_log_loss - cv_mean)
    delta_cv_train = None if not cv_ok else (cv_mean - train_log_loss)

    # Risk metrics
    if cv_ok:
        tol_holdout = min(TOL_CV_TEST, FRAC_OF_CV_STD * cv_std)  # test vs CV
        tol_train = min(TOL_CV_TRAIN, FRAC_OF_CV_STD * cv_std)  # CV vs train

        z_test_cv = max(0.0, (delta_test_cv - tol_holdout) / cv_std)
        z_train_cv = max(0.0, (delta_cv_train - tol_train) / cv_std)
        std_excess = z_test_cv
    else:
        scale_fb = FALLBACK_STD
        tol_tt = min(TOL_CV_TRAIN, FRAC_OF_CV_STD * scale_fb)
        z_tt = max(0.0, (delta_test_train - tol_tt) / scale_fb)
        std_excess = z_tt

    # Capping score between [0, 1] for model selection
    score = float(min(1.0, std_excess / STD_MAX))

    # Bands
    yellow_threshold = 0.33 * STD_MAX
    red_threshold = 0.67 * STD_MAX
    flag = (
        "green"
        if std_excess < yellow_threshold
        else ("yellow" if std_excess < red_threshold else "red")
    )

    out = {
        "overfit.score": score,
        "overfit.std_excess": float(std_excess),
        "overfit.flag": flag,
        "delta.test_cv": (None if delta_test_cv is None else float(delta_test_cv)),
        "delta.cv_train": (None if delta_cv_train is None else float(delta_cv_train)),
        "delta.test_train": float(delta_test_train),
        "cv.logloss_mean": (None if cv_mean is None else float(cv_mean)),
        "cv.logloss_std": (None if cv_std is None else float(cv_std)),
    }

    # Optional: instability (symmetric) for dashboards if a valid split is supplied
    if valid is not None:
        valid_log_loss = float(model.model_performance(valid).logloss())
        delta_tv = test_log_loss - valid_log_loss
        delta_tv_std = None
        scale = cv_std if cv_ok else FALLBACK_STD
        delta_tv_std = (
            delta_tv / scale if scale and np.isfinite(scale) and scale > 0 else np.nan
        )
        out.update(
            {
                "delta.test_valid": float(delta_tv),
                "delta_std.test_valid": float(delta_tv_std),
            }
        )

    return out


def log_h2o_experiment(
    aml: H2OAutoML,
    *,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    target_col: str,
    experiment_id: str,
    imputer: t.Optional[imputation.SklearnImputerWrapper] = None,
) -> pd.DataFrame:
    """
    Logs evaluation metrics, plots, and model artifacts for all models in an H2O AutoML leaderboard to MLflow.

    Args:
        aml: trained H2OAutoML object.
        train: H2OFrame containing the training split.
        valid: H2OFrame containing the validation split.
        test: H2OFrame containing the test split.
        institution_id: Institution identifier, used to namespace the MLflow experiment.
        target_col: Column name of target (used for plotting and label extraction).
        target_name: Name of the target of the model from the config.
        checkpoint_name: Name of the checkpoint of the model from the config.
        workspace_path: Path prefix for experiment naming within MLflow.
        experiment_id: ID of experiment set during training call
        client: Optional MLflowClient instance. If not provided, one will be created.

    Returns:
        results_df (pd.DataFrame): DataFrame with metrics and MLflow run IDs for all successfully logged models.
    """
    LOGGER.info("Logging experiment to MLflow with classification plots...")

    leaderboard_df = _to_pandas(aml.leaderboard)

    log_h2o_experiment_summary(
        aml=aml,
        leaderboard_df=leaderboard_df,
        train=train,
        valid=valid,
        test=test,
        target_col=target_col,
    )

    # Capping # of models that we're logging to save some time
    MAX_MODELS_TO_LOG = 50
    top_model_ids = leaderboard_df["model_id"].tolist()[:MAX_MODELS_TO_LOG]

    if not top_model_ids:
        LOGGER.warning("No models found in leaderboard.")
        return experiment_id, pd.DataFrame()

    results = []
    num_models = len(top_model_ids)

    for idx, model_id in enumerate(top_model_ids):
        # Show status update
        model_num = idx + 1

        if model_num == 1 or model_num % 10 == 0 or model_num == num_models:
            LOGGER.info(
                f"Completed logging on {model_num}/{len(top_model_ids)} top models..."
            )

        # Setting threshold to 0.5 due to binary classification
        metrics = log_h2o_model(
            aml=aml,
            model_id=model_id,
            train=train,
            valid=valid,
            test=test,
            imputer=imputer,
            target_col=target_col,
            primary_metric=aml.sort_metric,
        )

        if metrics:
            results.append(metrics)

    results_df = pd.DataFrame(results)
    LOGGER.info(f"Finished logging on {len(results_df)} top model runs to MLflow.")

    return results_df


def log_h2o_experiment_summary(
    *,
    aml: H2OAutoML,
    leaderboard_df: pd.DataFrame,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    target_col: str,
    run_name: str = "H2O AutoML Experiment Summary and Storage",
) -> None:
    """
    Logs summary information about the H2O AutoML experiment to a dedicated MLflow run in
    the experiment with the leaderboard as a CSV, list of input features, training dataset
    (with splits e.g. "train", "test", "val"), target distribution, and the
    schema (column names and types).

    Args:
        aml: trained H2OAutoML object.
        leaderboard_df (pd.DataFrame): Leaderboard as DataFrame.
        train (H2OFrame): training H2OFrame.
        valid (H2OFrame): Validation H2OFrame.
        test (H2OFrame): test H2OFrame.
        target_col (str): Name of the target column.
        run_name (str): Name of the MLflow run. Defaults to "h2o_automl_experiment_summary".
    """
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name):
        # Log basic experiment metadata
        mlflow.log_metric("num_models_trained", len(leaderboard_df))
        mlflow.log_param("best_model_id", aml.leader.model_id)

        # Create tmp directory for artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log leaderboard
            leaderboard_path = os.path.join(tmpdir, "h2o_leaderboard.csv")
            leaderboard_df.to_csv(leaderboard_path, index=False)
            mlflow.log_artifact(leaderboard_path, artifact_path="leaderboard")

            # Log feature list
            features = [col for col in train.columns if col != target_col]
            features_path = os.path.join(tmpdir, "train_features.txt")
            with open(features_path, "w") as f:
                for feat in features:
                    f.write(f"{feat}\n")
            mlflow.log_artifact(features_path, artifact_path="inputs")

            # Log sampled training data
            train_df = _to_pandas(train)
            valid_df = _to_pandas(valid)
            test_df = _to_pandas(test)
            full_df = pd.concat([train_df, valid_df, test_df], axis=0)
            df_parquet_path = os.path.join(tmpdir, "full_dataset.parquet")
            full_df.to_parquet(df_parquet_path, index=False)
            mlflow.log_artifact(df_parquet_path, artifact_path="inputs")

            # Log target distribution
            target_dist_df = _to_pandas(train[target_col].table())
            target_dist_path = os.path.join(tmpdir, "target_distribution.csv")
            target_dist_df.to_csv(target_dist_path, index=False)
            mlflow.log_artifact(target_dist_path, artifact_path="inputs")

            # Log schema
            schema_df = pd.DataFrame(train.types.items(), columns=["column", "dtype"])
            schema_path = os.path.join(tmpdir, "train_schema.csv")
            schema_df.to_csv(schema_path, index=False)
            mlflow.log_artifact(schema_path, artifact_path="inputs")


@contextlib.contextmanager
def _suppress_output():
    """Silence stdout/stderr just for chatty H2O calls."""
    with (
        open(os.devnull, "w") as fnull,
        contextlib.redirect_stdout(fnull),
        contextlib.redirect_stderr(fnull),
    ):
        yield


def log_h2o_model(
    *,
    aml: H2OAutoML,
    model_id: str,
    train: h2o.H2OFrame,
    valid: h2o.H2OFrame,
    test: h2o.H2OFrame,
    threshold: float = 0.5,
    target_col: str = "target",
    imputer: t.Optional[imputation.SklearnImputerWrapper] = None,
    primary_metric: str = "logloss",
) -> dict | None:
    """
    Evaluates a single H2O model and logs metrics, plots, and artifacts to MLflow.
    Optimizations:
      - restrict output suppression to chatty calls only
      - sample small subset for signature prediction (avoid full-train predict)
    """
    try:
        # get model
        model = h2o.get_model(model_id)

        # compute scalar metrics once (no logging here)
        metrics = evaluation.get_metrics_near_threshold_all_splits(
            model, train, valid, test, threshold=threshold
        )

        # ensure clean run context
        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run():
            active_run = mlflow.active_run()
            run_id = active_run.info.run_id if active_run else None

            # primary metric tag
            mlflow.set_tag("mlflow.primaryMetric", f"validate_{primary_metric}")

            # model comparison plot
            # keep this where it is, but only silence its progress bars
            with _suppress_output():
                evaluation.create_and_log_h2o_model_comparison(aml=aml)

            # per-split predictions, confusion matrix + plots
            for split_name, frame in zip(
                ("train", "val", "test"), (train, valid, test)
            ):
                # y_true (pandas)
                y_true = _to_pandas(frame[target_col]).values.flatten()

                # predict probabilities
                with _suppress_output():
                    preds = model.predict(frame)
                positive_class_label = preds.col_names[-1]

                # pull prob column to pandas
                y_proba = _to_pandas(preds[positive_class_label]).values.flatten()
                y_pred = (y_proba >= threshold).astype(int)

                # confusion matrix counts (for FE tables)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                label = "validate" if split_name == "val" else split_name
                metrics.update(
                    {
                        f"{label}_true_positives": float(tp),
                        f"{label}_true_negatives": float(tn),
                        f"{label}_false_positives": float(fp),
                        f"{label}_false_negatives": float(fn),
                    }
                )

                # classification plots (ROC/PR/etc)
                with _suppress_output():
                    evaluation.generate_all_classification_plots(
                        y_true, y_pred, y_proba, prefix=split_name
                    )

            # Log overfit score based on log loss and CV
            ofs = compute_overfit_score_logloss(
                model=model, train=train, valid=valid, test=test
            )
            metrics.update(ofs)

            with _suppress_output():
                # params + metrics (use the batched version you implemented)
                log_model_metadata_to_mlflow(
                    model_id=model_id,
                    model=model,
                    metrics=metrics,
                    exclude_keys={"model_id"},
                )

            # signature + UC artifacts (avoid full-train predict)
            # sample a small slice from the H2OFrame for signature inference
            # prefer first 200 rows; if nrows unavailable, just slice 200
            try:
                nrows = int(getattr(train, "nrows", 200))
            except Exception:
                nrows = 200
            n = max(1, min(200, nrows))

            # Small H2OFrame sample (includes target, drop it later for X)
            sample_hf = train[:n, :]

            # Convert the sample to pandas for signature inputs
            X_df = _to_pandas(sample_hf)
            if isinstance(X_df, pd.DataFrame):
                X_sample = X_df.drop(columns=[target_col], errors="ignore")
            else:
                # very defensive fallback; shouldn't happen in normal runs
                X_sample = pd.DataFrame({"__f__": [0.0]})

            # Predict on the *same small sample* (fast) for signature outputs
            with _suppress_output():
                y_pred_sample = model.predict(sample_hf).as_data_frame()
            if hasattr(y_pred_sample, "as_data_frame"):
                y_pred_sample = y_pred_sample.as_data_frame()

            X_sample = X_sample.copy()
            maybe_na_ints = [
                c for c in X_sample.columns if is_integer_dtype(X_sample[c].dtype)
            ]
            for c in maybe_na_ints:
                X_sample[c] = X_sample[c].astype("float64")

            signature = infer_signature(X_sample, y_pred_sample)

            with _suppress_output():
                # Optimized UC logger
                log_h2o_model_metadata_for_uc(
                    h2o_model=model,
                    artifact_path="model",
                    signature=signature,
                    # include_env_files=False by default for speed
                )

            # imputer artifacts
            if imputer is not None:
                try:
                    imputer.log_pipeline(artifact_path="sklearn_imputer")
                except Exception as e:
                    LOGGER.warning(f"Failed to log imputer artifacts: {e}")

        metrics["mlflow_run_id"] = str(run_id)
        return metrics

    except Exception as e:
        LOGGER.exception(f"Failed to evaluate and log model {model_id}: {e}")
        return None


def log_h2o_model_metadata_for_uc(
    h2o_model: ModelBase,
    artifact_path: str,
    signature: mlflow.models.signature.ModelSignature,
    include_env_files: bool = False,
) -> None:
    """
    Custom H2O model logger (Unity Catalog-compatible & future-proof for MLflow 3.x).
    Mlflow 3.x will deprecate mlflow.h2o.log_model.
    Saves the H2O model + MLmodel metadata so Unity Catalog can register it.

    Args:
        h2o_model: trained H2O model to log.
        artifact_path: Subdir in MLflow run artifacts (e.g. "model").
        signature: Optional MLflow signature object (mlflow.models.signature.ModelSignature).
    """
    # Prefer fast local SSD on Databricks
    base_tmp = "/local_disk0" if os.path.exists("/local_disk0") else None
    with tempfile.TemporaryDirectory(dir=base_tmp) as tmpdir:
        # 1) Save raw H2O model to fast local disk
        model_saved_path = h2o.save_model(h2o_model, path=tmpdir, force=True)

        # Normalize filename to "model.h2o" in the same fs
        final_model_path = os.path.join(tmpdir, "model.h2o")
        if model_saved_path != final_model_path:
            os.rename(model_saved_path, final_model_path)

        # 2) Try to export MOJO
        final_mojo_path = _try_export_mojo(h2o_model, tmpdir)
        if final_mojo_path:
            mlflow.log_artifact(final_mojo_path, artifact_path=artifact_path)

        # 3) Build MLmodel metadata
        mlmodel = Model(artifact_path=artifact_path, flavors={})
        mlmodel.add_flavor(
            "h2o",
            h2o_version=h2o.__version__,
            model_data="model.h2o",
        )
        if signature is not None:
            mlmodel.signature = signature

        # Convert the MLmodel to YAML text and log directly (no dir walk)
        mlmodel_path = os.path.join(tmpdir, "MLmodel")
        mlmodel.save(mlmodel_path)
        with open(mlmodel_path, "r") as f:
            mlmodel_yaml = f.read()
        mlflow.log_text(mlmodel_yaml, artifact_file=f"{artifact_path}/MLmodel")

        # 4) (Optional) minimal env files. Skip by default for speed.
        if include_env_files:
            # Small files, but still I/O + upload; only do if you need them.
            reqs_path = os.path.join(tmpdir, "requirements.txt")
            with open(reqs_path, "w") as f:
                f.write(f"h2o=={h2o.__version__}\n")
            mlflow.log_artifact(reqs_path, artifact_path=artifact_path)

            conda_env = {
                "name": "h2o_env",
                "channels": ["defaults", "conda-forge"],
                "dependencies": [
                    f"h2o={h2o.__version__}",
                    "pip",
                    {"pip": [f"mlflow=={mlflow.__version__}"]},
                ],
            }
            conda_path = os.path.join(tmpdir, "conda.yaml")
            with open(conda_path, "w") as f:
                yaml.safe_dump(conda_env, f)
            mlflow.log_artifact(conda_path, artifact_path=artifact_path)

        # 5) Upload the big file LAST (single call, no directory walk)
        mlflow.log_artifact(final_model_path, artifact_path=artifact_path)


def log_model_metadata_to_mlflow(
    model_id: str,
    model: ModelBase,
    metrics: dict[str, t.Any],
    exclude_keys: t.Optional[set[str]] = None,
) -> None:
    exclude_keys = exclude_keys or set()

    # 1) model_id as a single param
    mlflow.log_param("model_id", model_id)

    # 2) Hyperparameters in one batch
    try:
        hyperparams = {
            k: str(v)
            for k, v in getattr(model, "_parms", {}).items()
            if (
                v is not None
                and k != "model_id"
                and not isinstance(v, (h2o.H2OFrame, list, dict))
            )
        }
        if hyperparams:
            mlflow.log_params(hyperparams)  # ← batch
    except Exception as e:
        LOGGER.warning(f"Failed to log hyperparameters for model {model_id}: {e}")

    # 3) Metrics in one batch (cast to float & drop non-numeric)
    numeric_metrics: dict[str, float] = {}
    for k, v in metrics.items():
        if k in exclude_keys:
            continue
        try:
            numeric_metrics[k] = float(v)
        except (TypeError, ValueError):
            # Skip non-numeric metrics silently or warn once if you prefer
            continue

    if numeric_metrics:
        # This is a single batch call under the hood
        mlflow.log_metrics(numeric_metrics)


def set_or_create_experiment(
    workspace_path: str,
    institution_id: str,
    target_name: str,
    checkpoint_name: str,
    client: t.Optional[MlflowClient] = None,
) -> str:
    """
    Creates or retrieves a structured MLflow experiment and sets it as the active experiment.

    Args:
        workspace_path: Base MLflow workspace path.
        institution_id: Institution or tenant identifier used for experiment naming.
        target_name: Name of the target variable.
        checkpoint_name: Name of the modeling checkpoint.
        client: MLflow client. A new one is created if not provided.

    Returns:
        MLflow experiment ID (created or retrieved).
    """
    if client is None:
        client = MlflowClient()

    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    name_parts = [institution_id, target_name, checkpoint_name, "h2o_automl", timestamp]
    experiment_name = "/".join(
        [
            workspace_path.rstrip("/"),
            "h2o_automl",
            "_".join([part for part in name_parts if part]),
        ]
    )

    try:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = client.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        mlflow.set_experiment(experiment_name)
        return experiment_id
    except Exception as e:
        raise RuntimeError(f"Failed to create or set MLflow experiment: {e}")


def correct_h2o_dtypes(
    h2o_df: h2o.H2OFrame,
    original_df: pd.DataFrame,
    force_enum_cols: t.Optional[t.List[str]] = None,
) -> h2o.H2OFrame:
    """
    Correct H2OFrame dtypes based on original pandas DataFrame, ensuring columns
    inferred as numeric in H2O are restored to categorical/enums if they were
    non-numeric in pandas.

    Args:
        h2o_df: H2OFrame created from original_df
        original_df: Original pandas DataFrame with dtype info
        force_enum_cols: Optional list of column names to forcibly convert to enum

    Returns:
        h2o_df (possibly modified)
    """
    force_enum_cols = set(force_enum_cols or [])
    converted_columns = []

    LOGGER.info("Starting H2O dtype correction.")

    for col in original_df.columns:
        if col not in h2o_df.columns:
            LOGGER.debug(f"Skipping '{col}': not found in H2OFrame.")
            continue

        orig_dtype = original_df[col].dtype
        h2o_type = h2o_df.types.get(col)
        is_non_numeric = (
            is_categorical_dtype(original_df[col])
            or is_object_dtype(original_df[col])
            or is_string_dtype(original_df[col])
        )
        h2o_is_numeric = h2o_type in ("int", "real")

        should_force = col in force_enum_cols and h2o_type not in ("enum",)
        needs_correction = (is_non_numeric and h2o_is_numeric) or should_force

        LOGGER.debug(
            f"Column '{col}': orig_dtype={orig_dtype}, h2o_dtype={h2o_type}, "
            f"non_numeric={is_non_numeric}, force={should_force}"
        )

        if needs_correction:
            try:
                h2o_df[col] = h2o_df[col].asfactor()
                converted_columns.append(col)
                LOGGER.info(
                    f"Converted '{col}' to enum "
                    f"(originally {orig_dtype}, inferred as {h2o_type})."
                )
            except Exception as e:
                LOGGER.warning(f"Failed to convert '{col}' to enum: {e}")

    LOGGER.info(
        f"H2O dtype correction complete. {len(converted_columns)} column(s) affected: {converted_columns}"
    )
    return h2o_df


def _to_h2o(
    pobj: t.Any, force_enum_cols: t.Optional[t.List[str]] = None
) -> h2o.H2OFrame:
    """Convert common Python objects to an H2OFrame.

    This function wraps multiple input types into an H2OFrame and applies
    `correct_h2o_dtypes` so that categorical columns from pandas are preserved
    as enums in H2O.

    Args:
        pobj (Any):
            The object to convert. Supported types:
              - `pandas.DataFrame`: Converted directly to H2OFrame.
              - `pandas.Series`: Converted to single-column H2OFrame.
              - `numpy.ndarray`: Converted to H2OFrame via a pandas.DataFrame wrapper.
              - `h2o.H2OFrame`: Returned as-is.
        force_enum_cols (Optional[List[str]]):
            Optional list of column names to force conversion to enum
            regardless of dtype.

    Returns:
        h2o.H2OFrame:
            The converted H2OFrame with corrected dtypes.

    Raises:
        TypeError: If the input type is unsupported or `None`.
    """
    if pobj is None:
        raise TypeError("_to_h2o: cannot convert None")

    # Already H2OFrame
    if H2OFrame is not None and isinstance(pobj, H2OFrame):
        return pobj

    # Pandas DataFrame
    if isinstance(pobj, pd.DataFrame):
        hf = h2o.H2OFrame(pobj)
        return correct_h2o_dtypes(hf, pobj, force_enum_cols=force_enum_cols)

    # Pandas Series
    if isinstance(pobj, pd.Series):
        df = pobj.to_frame()
        hf = h2o.H2OFrame(df)
        return correct_h2o_dtypes(hf, df, force_enum_cols=force_enum_cols)

    # Numpy array
    if isinstance(pobj, np.ndarray):
        if pobj.ndim == 1:
            df = pd.DataFrame({0: pobj})
        else:
            df = pd.DataFrame(pobj)
        hf = h2o.H2OFrame(df)
        return correct_h2o_dtypes(hf, df, force_enum_cols=force_enum_cols)

    raise TypeError(f"_to_h2o: unsupported object type {type(pobj)}")


def _to_pandas(hobj: t.Any) -> pd.DataFrame:
    """
    Convert common H2O objects to pandas.DataFrame.

    - H2OFrame.as_data_frame() supports `use_pandas` and `use_multi_thread` (for performance).
    - H2OTwoDimTable.as_data_frame() takes no arguments in H2O 3.46+.
    - For other objects, we'll use `as_data_frame()`.
    """
    # Case 1: Big data — use multithreaded pull for H2OFrame
    if H2OFrame is not None and isinstance(hobj, H2OFrame):
        try:
            return hobj.as_data_frame(use_pandas=True, use_multi_thread=True)
        except TypeError:
            # Very old H2O without use_multi_thread
            return hobj.as_data_frame(use_pandas=True)

    # Case 2: Metric tables such as H2OTwoDimTable doesn't support multi-thread
    if H2OTwoDimTable is not None and isinstance(hobj, H2OTwoDimTable):
        return hobj.as_data_frame()

    # Case 3: Fallback for any other hobj that supports as_dataframe
    if hasattr(hobj, "as_data_frame"):
        try:
            return hobj.as_data_frame()
        except TypeError:
            # Last-resort fallback for legacy signatures
            return hobj.as_data_frame(use_pandas=True)

    raise TypeError(f"_to_pandas: unsupported object type {type(hobj)}")


def _try_export_mojo(h2o_model: ModelBase, tmpdir: str) -> str | None:
    """
    Attempt to export a MOJO for the given model. Returns (has_mojo, final_mojo_path).
    """
    final_mojo_path = os.path.join(tmpdir, "model.zip")
    try:
        # Primary, version-agnostic path: model method
        mojo_path = h2o_model.download_mojo(path=tmpdir)
        if mojo_path and os.path.exists(mojo_path):
            if mojo_path != final_mojo_path:
                os.replace(mojo_path, final_mojo_path)
            return final_mojo_path
        else:
            logging.warning(
                "download_mojo returned no path for algo=%s",
                getattr(h2o_model, "algo", "?"),
            )
            return None
    except AttributeError as e:
        logging.warning("Model has no download_mojo(): %s", e)
        return None
    except Exception as e:
        # Some algos/params don't support MOJO
        logging.warning(
            "MOJO export failed for algo=%s: %s", getattr(h2o_model, "algo", "?"), e
        )
        return None
