"""Post-mitigation FNR for every student group, logged beside the training audit."""

from __future__ import annotations

import logging
import typing as t

import mlflow
import pandas as pd

from edvise.modeling import bias_detection

LOGGER = logging.getLogger(__name__)

POST_MITIGATION_ARTIFACT_SUBDIR = "post_mitigation"


def assign_dual_threshold_predictions(
    df: pd.DataFrame,
    *,
    group_col: str,
    target_group: str,
    group_threshold: float,
    default_threshold: float,
    pred_col: str,
    pred_prob_col: str,
) -> pd.DataFrame:
    """Flag the target group at its threshold and everyone else at the default."""
    out = df.copy()
    scores = pd.to_numeric(out[pred_prob_col], errors="coerce")
    is_target = out[group_col].astype("string").str.strip().eq(str(target_group))
    out[pred_col] = ((scores >= float(group_threshold)) & is_target) | (
        (scores >= float(default_threshold)) & ~is_target
    )
    return out


def _default_threshold(cfg: t.Any) -> float:
    inference = getattr(cfg, "inference", None)
    value = getattr(inference, "min_prob_pos_label", None) if inference else None
    if value is None:
        return 0.5
    return float(value)


def _group_columns(cfg: t.Any, mitigation_col: str) -> list[str]:
    columns: list[str] = []
    for col in [mitigation_col, *(getattr(cfg, "student_group_cols", None) or [])]:
        if isinstance(col, str) and col.strip() and col not in columns:
            columns.append(col)
    return columns


def score_test_split(cfg: t.Any) -> pd.DataFrame:
    """Score the test split and attach ``pred_prob`` without changing support scores."""
    from edvise.scripts import predictions_h2o as pred

    experiment_id = cfg.model.experiment_id
    run_id = cfg.model.run_id
    if not experiment_id or not run_id:
        raise ValueError("cfg.model.experiment_id and cfg.model.run_id are required.")

    _, df_test = pred.extract_and_split_training_data(experiment_id, cfg.split_col)
    model, feat_names = pred.load_model_and_features(run_id)
    imputer = pred.imputer_for_run(run_id)
    calibrator = pred.calibrator_for_run(run_id)
    features_df, _ = pred.align_features(
        imputer.transform(df_test), feat_names, cfg.student_id_col
    )
    pos_label = "true" if cfg.pos_label is None else cfg.pos_label
    _, probs = pred.predict_probs(
        features_df,
        model,
        feat_names,
        str(pos_label),
        calibrator=calibrator,
    )
    out = df_test.copy()
    out[cfg.pred_prob_col] = probs
    return out


def log_post_mitigation_group_bias(cfg: t.Any) -> None:
    """Recompute FNR for every group under the dual threshold and log the plots."""
    mitigation = cfg.modeling.bias_mitigation
    group_col = mitigation.student_group_col
    scored = score_test_split(cfg)
    if group_col not in scored.columns:
        raise ValueError(
            f"Bias mitigation group column {group_col!r} is not on the test split."
        )
    split_col = cfg.split_col or "split"
    if split_col not in scored.columns:
        scored[split_col] = "test"
    scored = assign_dual_threshold_predictions(
        scored,
        group_col=group_col,
        target_group=mitigation.student_group,
        group_threshold=float(mitigation.custom_threshold),
        default_threshold=_default_threshold(cfg),
        pred_col=cfg.pred_col,
        pred_prob_col=cfg.pred_prob_col,
    )
    candidate_cols = [col for col in _group_columns(cfg, group_col) if col != group_col]
    group_cols = [col for col in candidate_cols if col in scored.columns]
    missing = [col for col in candidate_cols if col not in scored.columns]
    for col in missing:
        LOGGER.warning(
            "Skipping post-mitigation FNR for %s; column is not on the test split.",
            col,
        )
    run_id = cfg.model.run_id
    active = mlflow.active_run()
    if active is not None and active.info.run_id != run_id:
        mlflow.end_run()
        active = None

    def _evaluate() -> None:
        pos_label = True if cfg.pos_label is None else cfg.pos_label
        sample_weight_col = cfg.sample_weight_col or "sample_weight"
        bias_detection.evaluate_bias(
            scored,
            student_group_cols=group_cols,
            pos_label=pos_label,
            split_col=split_col,
            target_col=cfg.target_col,
            pred_col=cfg.pred_col,
            pred_prob_col=cfg.pred_prob_col,
            sample_weight_col=sample_weight_col,
            artifact_subdir=POST_MITIGATION_ARTIFACT_SUBDIR,
            log_run_metrics=False,
            plot_all_groups=True,
            threshold_label="dual threshold",
        )

    if active is None:
        with mlflow.start_run(run_id=run_id):
            _evaluate()
    else:
        _evaluate()
