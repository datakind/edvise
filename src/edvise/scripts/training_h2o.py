import typing as t
import logging
import argparse
import pandas as pd
import mlflow
import os

from mlflow.tracking import MlflowClient

from edvise import modeling, dataio, configs
from edvise.modeling.h2o_ml import utils as h2o_utils
from edvise import utils as edvise_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)

# comment to test


class TrainingParams(t.TypedDict, total=False):
    db_run_id: str
    institution_id: str
    student_id_col: str
    target_col: str
    split_col: t.Optional[str]
    pos_label: t.Optional[str | bool]
    primary_metric: str
    timeout_minutes: int
    exclude_cols: t.List[str]
    target_name: str
    checkpoint_name: str
    workspace_path: str
    seed: int


class TrainingTask:
    """Performs feature selection and training the model for the pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = dataio.read.read_config(
            self.args.config_file_path,
            schema=configs.pdp.PDPProjectConfig,
        )
        self.client = MlflowClient()
        h2o_utils.safe_h2o_init()
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    def feature_selection(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.modeling is None or self.cfg.modeling.feature_selection is None:
            raise ValueError(
                "FEATURE SELECTION SECTION OF MODELING DOES NOT EXIST IN THE CONFIG, PLEASE ADD"
            )

        modeling_cfg = self.cfg.modeling
        fs = modeling_cfg.feature_selection
        selection_params: t.Dict[str, t.Any] = fs.model_dump() if fs is not None else {}
        selection_params["non_feature_cols"] = self.cfg.non_feature_cols

        mlflow.autolog(disable=True)

        df_selected = modeling.feature_selection.select_features(
            (
                df_preprocessed.loc[df_preprocessed[self.cfg.split_col].eq("train"), :]
                if self.cfg.split_col
                else df_preprocessed
            ),
            **selection_params,
        )
        df_modeling = df_preprocessed.loc[:, df_selected.columns]
        return df_modeling

    def train_model(self, df_modeling: pd.DataFrame) -> str:
        mlflow.autolog(disable=False)

        # KAYLA TODO: figure out how we want to create this - create a user email field in yml to deploy?
        # Use run as for now - this will work until we set this as a service account
        workspace_path = f"/Users/{self.args.ds_run_as}"

        if self.cfg.modeling is None or self.cfg.modeling.training is None:
            raise ValueError("Missing section of the config: modeling.training")
        if self.cfg.preprocessing is None:
            raise ValueError("Missing 'preprocessing' section in config.")
        if self.cfg.preprocessing.target is None:
            raise ValueError("Missing 'preprocessing.target' section in config.")
        if self.cfg.preprocessing.checkpoint is None:
            raise ValueError("Missing 'preprocessing.checkpoint' section in config.")
        if self.cfg.pos_label is None:
            raise ValueError("Missing 'pos_label' in config.")

        pos_label = self.cfg.pos_label

        modeling_cfg = self.cfg.modeling
        training_cfg = modeling_cfg.training
        preprocessing_cfg = self.cfg.preprocessing

        db_run_id = self.args.db_run_id or ""

        timeout_minutes = training_cfg.timeout_minutes or 10
        split_col = self.cfg.split_col or "split"

        exclude_cols = list(
            set((training_cfg.exclude_cols or []) + (self.cfg.student_group_cols or []))
        )

        training_params: TrainingParams = {
            "db_run_id": db_run_id,
            "institution_id": self.cfg.institution_id,
            "student_id_col": self.cfg.student_id_col,
            "target_col": self.cfg.target_col,
            "split_col": split_col,
            "pos_label": pos_label,
            "primary_metric": training_cfg.primary_metric,
            "timeout_minutes": timeout_minutes,
            "exclude_cols": sorted(exclude_cols),
            "target_name": preprocessing_cfg.target.name,
            "checkpoint_name": preprocessing_cfg.checkpoint.name,
            "workspace_path": workspace_path,
            "seed": self.cfg.random_state or 42,  # fallback to ensure it's an int
        }

        experiment_id, *_ = modeling.h2o_ml.training.run_h2o_automl_classification(
            df=df_modeling,
            **training_params,
            client=self.client,
        )

        return experiment_id

    def evaluate_models(self, df_modeling: pd.DataFrame, experiment_id: str) -> None:
        student_group_cols: t.List[str] = t.cast(
            t.List[str], self.cfg.student_group_cols or []
        )

        if self.cfg.split_col is not None:
            df_features = df_modeling.drop(columns=self.cfg.non_feature_cols)
        else:
            raise ValueError("SPLIT COL DOES NOT EXIST IN THE CONFIG, PLEASE ADD")

        topn = 5
        if (mc := self.cfg.modeling) is not None and (ev := mc.evaluation) is not None:
            topn = ev.topn_runs_included

        top_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall",
                "test_roc_auc",
                "test_log_loss",
                "test_f1",
                "validate_log_loss",
            ],
            topn_runs_included=topn,
        )
        logging.info("top run ids = %s", top_runs)

        for run_id in top_runs.values():
            with mlflow.start_run(run_id=run_id):
                logging.info(
                    "Run %s: Starting performance evaluation%s",
                    run_id,
                    " and bias assessment",
                )
                df_features_imp = (
                    modeling.h2o_ml.imputation.SklearnImputerWrapper.load_and_transform(
                        df=df_features,
                        run_id=run_id,
                    )
                )
                model = h2o_utils.load_h2o_model(run_id=run_id)
                labels, probs = modeling.h2o_ml.inference.predict_h2o(
                    features=df_features_imp,
                    model=model,
                    pos_label=self.cfg.pos_label,
                )
                df_pred = df_modeling.assign(
                    **{
                        self.cfg.pred_col: labels,
                        self.cfg.pred_prob_col: probs,
                    }
                )
                modeling.evaluation.evaluate_performance(
                    df_pred,
                    target_col=self.cfg.target_col,
                    pos_label=self.cfg.pos_label,
                )
                modeling.bias_detection.evaluate_bias(
                    df_pred,
                    student_group_cols=student_group_cols,
                    target_col=self.cfg.target_col,
                    pos_label=self.cfg.pos_label,
                )
                logging.info("Run %s: Completed", run_id)

    def select_model(self, experiment_id: str) -> None:
        topn = 5
        if (mc := self.cfg.modeling) is not None and (ev := mc.evaluation) is not None:
            topn = ev.topn_runs_included
        selected_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall",
                "test_roc_auc",
                "test_log_loss",
                "test_bias_score_mean",
            ],
            topn_runs_included=topn,
        )
        logging.info("top run ids for model selection = %s", selected_runs)

        top_run_name, top_run_id = next(iter(selected_runs.items()))
        logging.info(f"Selected top run: {top_run_name} & {top_run_id}")

        edvise_utils.update_config.update_run_metadata(
            config_path=self.args.config_file_path,
            run_id=top_run_id,
            experiment_id=experiment_id,
        )
    def make_predictions(self):
        model = h2o_utils.load_h2o_model(self.cfg.model.run_id)
        model_feature_names = modeling.h2o_ml.inference.get_h2o_used_features(model)
        
        logging.info(
            "model uses %s features: %s", len(model_feature_names), model_feature_names
        )

        # Extract training data (all splits)
        df = modeling.h2o_ml.evaluation.extract_training_data_from_model(
            self.cfg.model.experiment_id
        )
        if self.cfg.split_col:
            df_train = df_train.loc[df_train[self.cfg.split_col].eq("train"), :]
            df_test = df.loc[df[self.cfg.split_col].eq("test"), :]

        # Need only a sample to produce SHAP plot for training
        df_test = df_test.sample(n=min(200, len(df_test)), random_state=self.cfg.random_state)

        # Load and transform using sklearn imputer
        imputer = modeling.h2o_ml.imputation.SklearnImputerWrapper.load(run_id=self.cfg.model.run_id)
        df_test = imputer.transform(df_test)

        # Extract features & unique ids
        features_df = df_test.loc[:, model_feature_names]
        unique_ids = df_test[self.cfg.student_id_col]

        # Generate predictions
        _, pred_probs = modeling.h2o_ml.inference.predict_h2o(
            features_df,
            model=model,
            feature_names=model_feature_names,
            pos_label=self.cfg.pos_label,
        )

        # Sample background data for performance optimization
        df_bd = df_train.sample(
            n=min(self.cfg.inference.background_data_sample, len(df_test)),
            random_state=self.cfg.random_state,
        )

        contribs_df = modeling.h2o_ml.inference.compute_h2o_shap_contributions(
            model=model,
            df=features_df,
            background_data=df_bd,
        )

        # Group one-hot encoding and missing value flags
        grouped_contribs_df = modeling.h2o_ml.inference.group_shap_values(contribs_df)
        grouped_features = modeling.h2o_ml.inference.group_feature_values(features_df)

        if mlflow.active_run():
            mlflow.end_run()

        with mlflow.start_run(run_id=self.cfg.model.run_id):
            # Create & log SHAP summary plot (default to group missing flags)
            modeling.h2o_ml.inference.plot_grouped_shap(
                contribs_df=contribs_df,
                features_df=features_df,
                original_dtypes=imputer.input_dtypes,
            )

        # Provide output using top features, SHAP values, and support scores
        result = modeling.inference.select_top_features_for_display(
            grouped_features,
            unique_ids,
            pred_probs,
            grouped_contribs_df.to_numpy(),
            n_features=10,
            features_table=features_table,
            needs_support_threshold_prob=self.cfg.inference.min_prob_pos_label,
        )

        # save sample advisor output dataset
        dataio.write.to_delta_table(
            result, self.cfg.datasets.gold.advisor_output.table_path, spark_session=spark
        )

        # Log MLFlow confusion matrix & roc table figures in silver schema
        with mlflow.start_run(run_id=self.cfg.model.run_id) as run:
            confusion_matrix = modeling.evaluation.log_confusion_matrix(
                institution_id=self.cfg.institution_id,
                automl_run_id=self.cfg.model.run_id,
            )

            # Log roc curve table for front-end
            roc_logs = modeling.h2o_ml.evaluation.log_roc_table(
                institution_id=self.cfg.institution_id,
                automl_run_id=self.cfg.model.run_id,
                modeling_dataset_name=self.cfg.datasets.silver.modeling.table_path,
            )

        shap_feature_importance = modeling.inference.generate_ranked_feature_table(
            features=grouped_features,
            shap_values=grouped_contribs_df.to_numpy(),
            features_table=features_table,
        )
        if shap_feature_importance is not None and features_table is not None:
            shap_feature_importance[
                ["readable_feature_name", "short_feature_desc", "long_feature_desc"]
            ] = shap_feature_importance["Feature Name"].apply(
                lambda feature: pd.Series(
                    modeling.inference._get_mapped_feature_name(
                        feature, features_table, metadata=True
                    )
                )
            )
            shap_feature_importance.columns = shap_feature_importance.columns.str.replace(
                " ", "_"
            ).str.lower()


        # save sample advisor output dataset
        dataio.write.to_delta_table(
            shap_feature_importance,
            f"{self.args.silver_volume_path}.training_{self.cfg.model.run_id}_shap_feature_importance",
            spark_session=spark,
        )

        support_score_distribution = modeling.inference.support_score_distribution_table(
            df_serving=grouped_features,
            unique_ids=unique_ids,
            pred_probs=pred_probs,
            shap_values=grouped_contribs_df,
            inference_params=cfg.inference.dict(),
        )

        # save sample advisor output dataset
        dataio.write.to_delta_table(
            support_score_distribution,
            f"{self.args.silver_volume_path}.training_{cfg.model.run_id}_support_overview",
            spark_session=spark,
        )
        dataio.write.write_parquet(
            support_score_distribution,
            file_path=f"{self.args.silver_volume_path}.training_{cfg.model.run_id}_support_overview",
        )




    def run(self):
        """Executes the target computation pipeline and saves result."""
        logging.info("Loading preprocessed data")
        df_preprocessed = pd.read_parquet(
            f"{self.args.silver_volume_path}/preprocessed.parquet"
        )
        logging.info("Selecting features")
        df_modeling = self.feature_selection(df_preprocessed)

        logging.info("Saving modeling data")
        df_modeling.to_parquet(
            f"{self.args.silver_volume_path}/modeling.parquet", index=False
        )
        logging.info(
            f"Modeling file saved to {self.args.silver_volume_path}/modeling.parquet"
        )
        logging.info("Training model")
        experiment_id = self.train_model(df_modeling)

        logging.info("Evaluating models")
        self.evaluate_models(df_modeling, experiment_id)

        logging.info("Selecting best model")
        self.select_model(experiment_id)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="H2o training for pipeline.")
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    parser.add_argument("--ds_run_as", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # try:
    #     if args.custom_schemas_path:
    #         sys.path.append(args.custom_schemas_path)
    #         schemas = importlib.import_module("schemas")
    #         logging.info("Using custom schemas")
    # except Exception:
    #     logging.info("Using default schemas")

    task = TrainingTask(args)
    task.run()
