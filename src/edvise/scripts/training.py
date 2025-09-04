import logging
import argparse
import pandas as pd
import sys
import importlib
import mlflow
import dbutils

from .. import modeling, utils, dataio, inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class TrainingTask:
    """Performs feature selection and training the model for the pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = dataio.read.read_PDP_config(self.args.toml_file_path)

    def feature_selection(self, df_preprocessed: pd.DataFrame) -> pd.Series:
        """ """
        selection_params = self.cfg.modeling.feature_selection.model_dump()
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

    def train_model(self, df_modeling: pd.DataFrame) -> None:
        mlflow.autolog(disable=False)
        # Get job run id for automl run
        job_run_id = utils.databricks.get_db_widget_param(
            "job_run_id", default="interactive"
        )
        training_params = {
            "job_run_id": job_run_id,
            "institution_id": self.cfg.institution_id,
            "student_id_col": self.cfg.student_id_col,
            "target_col": self.cfg.target_col,
            "split_col": self.cfg.split_col,
            "pos_label": self.cfg.pos_label,
            "primary_metric": self.cfg.modeling.training.primary_metric,
            "timeout_minutes": self.cfg.modeling.training.timeout_minutes,
            "exclude_frameworks": self.cfg.modeling.training.exclude_frameworks,
            "exclude_cols": sorted(
                set(
                    (self.cfg.modeling.training.exclude_cols or [])
                    + (self.cfg.student_group_cols or [])
                )
            ),
        }

        summary = modeling.training.run_automl_classification(
            df_modeling, **training_params
        )

        experiment_id = summary.experiment.experiment_id
        run_id = summary.best_trial.mlflow_run_id
        logging.info(
            f"experiment_id: {experiment_id}"
            f"\nbest trial run_id: {run_id}"
            f"\n{training_params['primary_metric']} metric distribution = {summary.metric_distribution}"
        )

        dbutils.jobs.taskValues.set(key="experiment_id", value=experiment_id)
        dbutils.jobs.taskValues.set(key="run_id", value=run_id)
        return experiment_id, run_id

    def evaluate_models(self, df_modeling: pd.DataFrame, experiment_id: str) -> None:
        split_col = self.cfg.split_col or "_automl_split_col_0000"

        # only possible to do bias evaluation if you specify a split col for train/test/validate
        # AutoML doesn't preserve student ids the training set, which we need for [reasons]
        evaluate_model_bias = split_col is not None

        if evaluate_model_bias:
            df_features = df_modeling.drop(columns=self.cfg.non_feature_cols)
        else:
            df_features = modeling.evaluation.extract_training_data_from_model(
                experiment_id
            )

        # Get top runs from experiment for evaluation
        top_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall_score",
                "test_roc_auc",
                "test_log_loss",
                "test_f1_score",
                "val_log_loss",
            ],
            topn_runs_included=self.cfg.modeling.evaluation.topn_runs_included,
        )
        logging.info("top run ids = %s", top_runs)

        for run_id in top_runs.values():
            with mlflow.start_run(run_id=run_id) as run:
                logging.info(
                    "Run %s: Starting performance evaluation%s",
                    run_id,
                    " and bias assessment" if evaluate_model_bias else "",
                )
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                df_pred = df_modeling.assign(
                    **{
                        self.cfg.pred_col: model.predict(df_features),
                        self.cfg.pred_prob_col: inference.prediction.predict_probs(
                            df_features,
                            model,
                            feature_names=list(df_features.columns),
                            pos_label=self.cfg.pos_label,
                        ),
                    }
                )
                model_comp_fig = modeling.evaluation.plot_trained_models_comparison(
                    experiment_id, self.cfg.modeling.training.primary_metric
                )
                modeling.evaluation.evaluate_performance(
                    df_pred,
                    target_col=self.cfg.target_col,
                    pos_label=self.cfg.pos_label,
                )
                if evaluate_model_bias:
                    modeling.bias_detection.evaluate_bias(
                        df_pred,
                        student_group_cols=self.cfg.student_group_cols,
                        target_col=self.cfg.target_col,
                        pos_label=self.cfg.pos_label,
                    )
                logging.info("Run %s: Completed", run_id)
        mlflow.end_run()

    def select_model(self, experiment_id: str) -> None:
        # Rank top runs again after evaluation for model selection
        selected_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall_score",
                "test_roc_auc",
                "test_log_loss",
                "test_bias_score_mean",
            ],
            topn_runs_included=self.cfg.modeling.evaluation.topn_runs_included,
        )
        logging.info("top run ids for model selection = %s", selected_runs)

        # Extract the top run
        top_run_name, top_run_id = next(iter(selected_runs.items()))
        logging.info(f"Selected top run: {top_run_name} & {top_run_id}")

        # Update config with run and experiment ids
        utils.update_run_metadata_in_toml(
            config_path="./config.toml",
            run_id=top_run_id,
            experiment_id=experiment_id,
        )

    def run(self):
        """Executes the target computation pipeline and saves result."""
        logging.info("Loading preprocessed data")
        df_preprocessed = pd.read_parquet(
            f"{self.args.student_term_path}/preprocessed.parquet"
        )

        logging.info("Selecting features")
        df_modeling = self.feature_selection(df_preprocessed)

        logging.info("Saving modeling data")
        df_modeling.to_parquet(
            f"{self.args.modeling_path}/modeling.parquet", index=False
        )
        logging.info(
            f"Modeling file saved to {self.args.modeling_path}/modeling.parquet"
        )

        logging.info("Training model")
        experiment_id, run_id = self.train_model(df_modeling)
        logging.info(
            f"Model trained with experiment_id={experiment_id}, run_id={run_id}"
        )

        logging.info("Evaluating models")
        self.evaluate_models(df_modeling, experiment_id)

        logging.info("Selecting best model")
        self.select_model(experiment_id)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Target generation for SST pipeline.")
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--custom_schemas_path", required=False, help="Path to custom schemas"
    )
    parser.add_argument(
        "--student_term_path",
        type=str,
        required=True,
        help="Path to student term parquet",
    )
    parser.add_argument(
        "--target_path", type=str, required=True, help="Path to output target parquet"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        if args.custom_schemas_path:
            sys.path.append(args.custom_schemas_path)
            schemas = importlib.import_module("schemas")
            logging.info("Using custom schemas")
    except Exception:
        logging.info("Using default schemas")

    task = TrainingTask(args)
    task.run()
