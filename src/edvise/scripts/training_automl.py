import typing as t
import logging
import argparse
import pandas as pd
import sys
import importlib
import mlflow
import dbutils

from .. import modeling, utils, dataio, configs

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class TrainingParams(t.TypedDict, total=False):
    job_run_id: str
    institution_id: str
    student_id_col: str
    target_col: str
    split_col: t.Optional[str]
    pos_label: t.Optional[str | bool]
    primary_metric: str
    timeout_minutes: int
    exclude_frameworks: t.Optional[t.List[str]]
    exclude_cols: t.List[str]


class TrainingTask:
    """Performs feature selection and training the model for the pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = dataio.read.read_config(
            self.args.toml_file_path,
            schema=configs.pdp.PDPProjectConfig,
        )

    def feature_selection(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        modeling_cfg = self.cfg.modeling
        if modeling_cfg is None or modeling_cfg.feature_selection is None:
            # No feature-selection configured: pass-through
            return df_preprocessed

        selection_params = modeling_cfg.feature_selection.model_dump()
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

    def train_model(self, df_modeling: pd.DataFrame) -> tuple[str, str]:
        mlflow.autolog(disable=False)

        modeling_cfg = self.cfg.modeling
        if modeling_cfg is None or modeling_cfg.training is None:
            raise ValueError("modeling.training must be configured in the TOML.")

        job_run_id = t.cast(
            str,
            utils.databricks.get_db_widget_param("job_run_id", default="interactive"),
        )

        # Coerce pos_label to expected type (bool | str | None)
        pos_label_cfg = self.cfg.pos_label
        if isinstance(pos_label_cfg, int):
            # If you truly mean “positive class is 1”, convert 0/1 -> False/True
            pos_label: t.Optional[bool | str] = bool(pos_label_cfg)
        else:
            pos_label = t.cast(t.Optional[bool | str], pos_label_cfg)

        timeout_minutes = modeling_cfg.training.timeout_minutes
        if timeout_minutes is None:
            timeout_minutes = 10

        training_params: TrainingParams = {
            "job_run_id": job_run_id,
            "institution_id": self.cfg.institution_id,
            "student_id_col": self.cfg.student_id_col,
            "target_col": self.cfg.target_col,
            "split_col": self.cfg.split_col,
            "pos_label": pos_label,
            "primary_metric": modeling_cfg.training.primary_metric,
            "timeout_minutes": timeout_minutes,
            "exclude_frameworks": modeling_cfg.training.exclude_frameworks,
            "exclude_cols": sorted(
                set(
                    (modeling_cfg.training.exclude_cols or [])
                    + (self.cfg.student_group_cols or [])
                )
            ),
        }

        summary = modeling.training.run_automl_classification(
            df_modeling,  # 1st positional arg
            **training_params,  # kwargs now precisely typed
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
        evaluate_model_bias = split_col is not None

        if evaluate_model_bias:
            df_features = df_modeling.drop(columns=self.cfg.non_feature_cols)
        else:
            df_features = modeling.evaluation.extract_training_data_from_model(
                experiment_id
            )

        modeling_cfg = self.cfg.modeling
        topn = 10
        if modeling_cfg is not None and modeling_cfg.evaluation is not None:
            topn = modeling_cfg.evaluation.topn_runs_included

        top_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall_score",
                "test_roc_auc",
                "test_log_loss",
                "test_f1_score",
                "val_log_loss",
            ],
            topn_runs_included=topn,
        )
        logging.info("top run ids = %s", top_runs)

        for run_id in top_runs.values():
            with mlflow.start_run(run_id=run_id):
                logging.info(
                    "Run %s: Starting performance evaluation%s",
                    run_id,
                    " and bias assessment" if evaluate_model_bias else "",
                )
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                df_pred = df_modeling.assign(
                    **{
                        self.cfg.pred_col: model.predict(df_features),
                        self.cfg.pred_prob_col: modeling.prediction.predict_probs(
                            df_features,
                            model,
                            feature_names=list(df_features.columns),
                            pos_label=t.cast(
                                t.Optional[bool | str],
                                self.cfg.pos_label
                                if not isinstance(self.cfg.pos_label, int)
                                else bool(self.cfg.pos_label),
                            ),
                        ),
                    }
                )
                modeling.evaluation.plot_trained_models_comparison(
                    experiment_id,
                    t.cast(str, modeling_cfg.training.primary_metric)
                    if (modeling_cfg and modeling_cfg.training)
                    else "test_roc_auc",
                )
                modeling.evaluation.evaluate_performance(
                    df_pred,
                    target_col=self.cfg.target_col,
                    pos_label=t.cast(
                        t.Optional[bool | str],
                        self.cfg.pos_label
                        if not isinstance(self.cfg.pos_label, int)
                        else bool(self.cfg.pos_label),
                    ),
                )
                if evaluate_model_bias:
                    # evaluate_bias expects list[Any]; give it a list (possibly empty) and cast
                    group_cols_any: t.List[t.Any] = t.cast(
                        t.List[t.Any], self.cfg.student_group_cols or []
                    )
                    modeling.bias_detection.evaluate_bias(
                        df_pred,
                        student_group_cols=group_cols_any,
                        target_col=self.cfg.target_col,
                        pos_label=t.cast(
                            t.Optional[bool | str],
                            self.cfg.pos_label
                            if not isinstance(self.cfg.pos_label, int)
                            else bool(self.cfg.pos_label),
                        ),
                    )
                logging.info("Run %s: Completed", run_id)
        mlflow.end_run()

    def select_model(self, experiment_id: str) -> None:
        modeling_cfg = self.cfg.modeling
        topn = 10
        if modeling_cfg is not None and modeling_cfg.evaluation is not None:
            topn = modeling_cfg.evaluation.topn_runs_included

        selected_runs = modeling.evaluation.get_top_runs(
            experiment_id,
            optimization_metrics=[
                "test_recall_score",
                "test_roc_auc",
                "test_log_loss",
                "test_bias_score_mean",
            ],
            topn_runs_included=topn,
        )
        logging.info("top run ids for model selection = %s", selected_runs)

        top_run_name, top_run_id = next(iter(selected_runs.items()))
        logging.info(f"Selected top run: {top_run_name} & {top_run_id}")

        utils.update_config.update_run_metadata_in_toml(
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
