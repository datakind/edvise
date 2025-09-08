import typing as t
from dataclasses import dataclass
import logging
import pandas as pd
import mlflow

from edvise import modeling, dataio
from edvise.modeling.h2o_ml import utils as h2o_utils

# ---- Dataclasses to decouple IO and outputs ----


class RunType(str, t.Enum):
    TRAIN = "train"
    PREDICT = "predict"


@dataclass
class PredConfig:
    model_run_id: str
    experiment_id: str | None
    split_col: str | None
    student_id_col: str
    pos_label: str | bool
    min_prob_pos_label: float
    background_data_sample: int
    cfg_inference_params: dict | None
    random_state: int


@dataclass
class PredPaths:
    silver_modeling_path: str | None = None
    features_table_path: str | None = None


@dataclass
class PredOutputs:
    top_features_result: pd.DataFrame
    shap_feature_importance: pd.DataFrame | None
    support_score_distribution: pd.DataFrame
    grouped_features: pd.DataFrame
    grouped_contribs_df: pd.DataFrame
    unique_ids: pd.Series
    pred_probs: pd.Series


def load_features_table(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    try:
        ft = dataio.read.read_features_table(file_path=path)
        logging.info("Loaded features table from %s", path)
        return ft
    except Exception as e:
        logging.warning("Features table missing/unreadable (%s). Continuing.", e)
        return None


def load_model_and_features(run_id: str):
    model = h2o_utils.load_h2o_model(run_id)
    feat_names = modeling.h2o_ml.inference.get_h2o_used_features(model)
    if not feat_names:
        raise ValueError("Model reports zero used features.")
    return model, feat_names


def extract_and_split_training_data(experiment_id: str | None, split_col: str | None):
    if not experiment_id:
        raise ValueError(
            "experiment_id is required to extract training data for background SHAP."
        )
    df_all = modeling.h2o_ml.evaluation.extract_training_data_from_model(experiment_id)
    if split_col:
        if split_col not in df_all.columns:
            raise ValueError(f"split_col='{split_col}' not in data.")
        df_train = df_all.loc[df_all[split_col].eq("train"), :].copy()
        df_test = df_all.loc[df_all[split_col].eq("test"), :].copy()
    else:
        logging.warning("No split_col configured; using all rows as both train/test.")
        df_train = df_all.copy()
        df_test = df_all.copy()
    if df_train.empty or df_test.empty:
        raise ValueError("Empty train/test split.")
    return df_train, df_test


def sample_rows(df: pd.DataFrame, n: int, seed: int, where: str) -> pd.DataFrame:
    n = int(min(max(1, n), len(df)))
    if n == 0:
        raise ValueError(f"Cannot sample from empty dataframe at {where}")
    return df.sample(n=n, random_state=seed)


def imputer_for_run(run_id: str):
    return modeling.h2o_ml.imputation.SklearnImputerWrapper.load(run_id=run_id)


def align_features(
    df_imp: pd.DataFrame, model_feature_names: list[str], student_id_col: str
):
    missing = [c for c in model_feature_names if c not in df_imp.columns]
    if missing:
        raise ValueError(f"Imputed data missing model features: {missing}")
    if student_id_col not in df_imp.columns:
        raise ValueError(f"Missing student_id_col: {student_id_col}")
    return df_imp.loc[:, model_feature_names].copy(), df_imp[student_id_col].copy()


def predict_probs(
    features_df: pd.DataFrame, model, feature_names: list[str], pos_label
):
    labels, probs = modeling.h2o_ml.inference.predict_h2o(
        features=features_df,
        model=model,
        feature_names=feature_names,
        pos_label=pos_label,
    )
    return labels, probs


def compute_shap(model, features_df: pd.DataFrame, background_df: pd.DataFrame):
    return modeling.h2o_ml.inference.compute_h2o_shap_contributions(
        model=model, df=features_df, background_data=background_df
    )


def group_shap_and_features(contribs_df: pd.DataFrame, features_df: pd.DataFrame):
    grouped_contribs_df = modeling.h2o_ml.inference.group_shap_values(contribs_df)
    grouped_features = modeling.h2o_ml.inference.group_feature_values(features_df)
    return grouped_contribs_df, grouped_features


def log_shap_plot(contribs_df, features_df, original_dtypes, run_id: str):
    if mlflow.active_run():
        mlflow.end_run()
    try:
        with mlflow.start_run(run_id=run_id):
            modeling.h2o_ml.inference.plot_grouped_shap(
                contribs_df=contribs_df,
                features_df=features_df,
                original_dtypes=original_dtypes,
            )
    except Exception as e:
        logging.warning("Failed to log SHAP plot: %s", e)


# ---- main orchestration that both training & inference can call ----


def run_predictions(
    cfg: PredConfig,
    paths: PredPaths,
    *,
    run_type: RunType = RunType.TRAIN,
    df_inference: pd.DataFrame | None = None,
    test_sample_cap: int = 200,
) -> PredOutputs:
    ft = load_features_table(paths.features_table_path)
    model, model_feature_names = load_model_and_features(cfg.model_run_id)
    imp = imputer_for_run(cfg.model_run_id)

    # ----- Build df_test (the rows to score) -----
    if run_type == RunType.TRAIN:
        df_train, df_test_all = extract_and_split_training_data(
            cfg.experiment_id, cfg.split_col
        )
        df_test = sample_rows(
            df_test_all,
            min(test_sample_cap, len(df_test_all)),
            cfg.random_state,
            "df_test(train)",
        )
    else:
        # PREDICT: inference input
        if df_inference is None:
            if not paths.inference_input_path:
                raise ValueError(
                    "inference_input_path or df_inference required for predict run."
                )
            df_inference = dataio.read.read_parquet(
                file_path=paths.inference_input_path
            )
        df_test_all = df_inference
        df_test = df_test_all  # inference needs SHAP for all rows

        # get a training background unless caller provides one
        df_train, _ = extract_and_split_training_data(cfg.experiment_id, cfg.split_col)

    # ----- Impute & align for predictions -----
    df_test_imp = imp.transform(df_test)
    features_df, unique_ids = align_features(
        df_test_imp, model_feature_names, cfg.student_id_col
    )
    pred_labels, pred_probs = predict_probs(
        features_df, model, model_feature_names, cfg.pos_label
    )

    # ----- Background for SHAP -----
    df_bd_raw = sample_rows(
        df_train,
        cfg.background_data_sample,
        cfg.random_state,
        "df_train(background)",
    )
    df_bd = imp.transform(df_bd_raw).loc[:, model_feature_names].copy()

    contribs_df = compute_shap(model, features_df, df_bd)
    grouped_contribs_df, grouped_features = group_shap_and_features(
        contribs_df, features_df
    )

    log_shap_plot(
        contribs_df,
        features_df,
        getattr(imp, "input_dtypes", None),
        cfg.model_run_id,
    )

    # ----- Tables -----
    top_features_result = modeling.automl.inference.select_top_features_for_display(
        grouped_features,
        features_df,
        pred_probs,
        n_features=10,
        features_table=ft,
        needs_support_threshold_prob=cfg.min_prob_pos_label,
    )

    sfi = modeling.automl.inference.generate_ranked_feature_table(
        features=grouped_features,
        shap_values=grouped_contribs_df.to_numpy(),
        features_table=ft,
    )
    if sfi is not None and ft is not None:
        sfi[["readable_feature_name", "short_feature_desc", "long_feature_desc"]] = sfi[
            "Feature Name"
        ].apply(
            lambda f: pd.Series(
                modeling.inference._get_mapped_feature_name(f, ft, metadata=True)
            )
        )
        sfi.columns = sfi.columns.str.replace(" ", "_").str.lower()

    ssd = modeling.automl.inference.support_score_distribution_table(
        df_serving=grouped_features,
        unique_ids=unique_ids,
        pred_probs=pred_probs,
        shap_values=grouped_contribs_df,
        inference_params=cfg.inference.dict(),
    )

    return PredOutputs(
        top_features_result=top_features_result,
        shap_feature_importance=sfi,
        support_score_distribution=ssd,
        grouped_features=grouped_features,
        grouped_contribs_df=grouped_contribs_df,
        unique_ids=unique_ids,
        pred_probs=pred_probs,
        pred_labels=pred_labels,
    )
