import typing as t
import logging
import argparse
import pandas as pd
import sys
import importlib
import mlflow
import dbutils
import os

from mlflow.tracking import MlflowClient
from databricks.connect import DatabricksSession
from databricks.sdk.runtime import dbutils
from pyspark.dbutils import DBUtils

from src.edvise import modeling, utils, dataio, configs

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
            self.args.toml_file_path,
            schema=configs.pdp.PDPProjectConfig,
        )
        self.client = MlflowClient()
        self.dbutils = DBUtils(spark)
        modeling.h2o.utils.safe_h2o_init()
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

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
            "exclude_cols": sorted(
                set(
                    (modeling_cfg.training.exclude_cols or [])
                    + (self.cfg.student_group_cols or [])
                )
            ),
            "target_name": self.cfg.preprocessing.target.name,
            "checkpoint_name": self.cfg.preprocessing.checkpoint.name,
            "workspace_path": modeling_cfg,
            "seed": self.cfg.random_state
        }
        experiment_id, aml, train, valid, test = (
            modeling.h2o.training.run_h2o_automl_classification(
                df=df_modeling,
                **training_params,
                client=self.client,
            )
        )

        return experiment_id, aml, train, valid, test

    def evaluate_models(self, df_modeling: pd.DataFrame, experiment_id: str) -> None:
        if self.cfg.split_col is not None:
            df_features = df_modeling.drop(columns=self.cfg.non_feature_cols)
        else:
            raise ValueError("Expected 'split_col' is missing, please add.")

        modeling_cfg = self.cfg.modeling
        topn = 10
        if modeling_cfg is not None and modeling_cfg.evaluation is not None:
            topn = modeling_cfg.evaluation.topn_runs_included

        #Kayla TODO: should this be the same evaluation or new one?
        top_runs = modeling.automl.evaluation.get_top_runs(
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
        # KAYLA TO PICK UP HERE 
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

    ##KAYLA: already edited this fn other than the update metadata function
    def select_model(self, experiment_id: str) -> None:
        modeling_cfg = self.cfg.modeling
        topn = 10
        if modeling_cfg is not None and modeling_cfg.evaluation is not None:
            topn = modeling_cfg.evaluation.topn_runs_included

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

        ##KAYLA TO LOOK AT THIS
        utils.update_run_metadata_in_toml.update_run_metadata_in_toml(
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
    parser = argparse.ArgumentParser(description="H2o training for pipeline.")
    #KAYLA ADD ARGS
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
