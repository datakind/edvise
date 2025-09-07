import typing as t
import logging
import argparse
import pandas as pd
import sys
import importlib
import mlflow
import os

from mlflow.tracking import MlflowClient

from src.edvise import modeling, dataio, configs
from src.edvise.modeling.h2o import utils as h2o_utils
from src.edvise import utils as edvise_utils

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


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
            raise ValueError("FEATURE SELECTION SECTION OF MODELING DOES NOT EXIST IN THE CONFIG, PLEASE ADD")
        
        modeling_cfg = self.cfg.modeling
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
        #KAYLA TODO: figure out how we want to create this - create a user email field in yml to deploy?
        # Use run as for now - this will work until we set this as a service account
        workspace_path = f"/Users/{self.args.ds_run_as}"
       
        if self.cfg.modeling is None or self.cfg.modeling.training is None:
            raise ValueError("TRAINING SECTION OF MODELING DOES NOT EXIST IN THE CONFIG, PLEASE ADD")
        
        modeling_cfg = self.cfg.modeling
        db_run_id = self.args.db_run_id

        #Assert this is a boolean - KAYLA to double check with VISH this is true 
        if not isinstance(self.cfg.pos_label, bool):
            raise ValueError("POSITIVE LABEL MUST BE BOOLEAN IN THE CONFIG, PLEASE ADD")

        timeout_minutes = modeling_cfg.training.timeout_minutes
        if timeout_minutes is None:
            timeout_minutes = 10

        training_params: TrainingParams = {
            "db_run_id": db_run_id,
            "institution_id": self.cfg.institution_id,
            "student_id_col": self.cfg.student_id_col,
            "target_col": self.cfg.target_col,
            "split_col": self.cfg.split_col,
            "pos_label": self.cfg.pos_label,
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
            "workspace_path": workspace_path,
            "seed": self.cfg.random_state
        }
        experiment_id, *_ = (
            modeling.h2o.training.run_h2o_automl_classification(
                df=df_modeling,
                **training_params,
                client=self.client,
            )
        )

        return experiment_id

    def evaluate_models(self, df_modeling: pd.DataFrame, experiment_id: str) -> None:
        if self.cfg.split_col is not None:
            df_features = df_modeling.drop(columns=self.cfg.non_feature_cols)
        else:
            raise ValueError("SPLIT COL DOES NOT EXIST IN THE CONFIG, PLEASE ADD")
        
        topn = 10
        if self.cfg.modeling is not None and self.cfg.modeling.evaluation is not None:
            modeling_cfg = self.cfg.modeling
            topn = modeling_cfg.evaluation.topn_runs_included

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
                    modeling.h2o.imputation.SklearnImputerWrapper.load_and_transform(
                        df=df_features,
                        run_id=run_id,
                    )
                )
                model = h2o_utils.load_h2o_model(run_id=run_id)
                labels, probs = modeling.h2o.inference.predict_h2o(
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
                    pos_label=self.cfg.pos_label
                )
                modeling.bias_detection.evaluate_bias(
                    df_pred,
                    student_group_cols=self.cfg.student_group_cols,
                    target_col=self.cfg.target_col,
                    pos_label=self.cfg.pos_label,
                )
                logging.info("Run %s: Completed", run_id)

    def select_model(self, experiment_id: str) -> None:
        if self.cfg.modeling is not None and self.cfg.modeling.evaluation is not None:
                    modeling_cfg = self.cfg.modeling
                    topn = modeling_cfg.evaluation.topn_runs_included

        topn = 10
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
