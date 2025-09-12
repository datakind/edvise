import typing as t
import logging
import argparse
import pandas as pd
import mlflow
import os
import sys

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Debug info
print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
from mlflow.tracking import MlflowClient

from edvise import modeling, dataio, configs
from edvise.modeling.h2o_ml import utils as h2o_utils
from edvise.reporting.model_card.h2o_pdp import H2OPDPModelCard
from edvise import utils as edvise_utils

from edvise.scripts.predictions_h2o import (
    PredConfig,
    PredPaths,
    RunType,
    run_predictions,
)

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
        self.spark_session = self.get_spark_session()
        self.cfg = dataio.read.read_config(
            self.args.config_file_path,
            schema=configs.pdp.PDPProjectConfig,
        )
        self.client = MlflowClient()
        self.features_table_path = "shared/assets/pdp_features_table.toml"
        h2o_utils.safe_h2o_init()
        os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"

    def get_spark_session(self) -> SparkSession:
        """
        Attempts to create a Spark session.
        Returns:
            DatabricksSession | None: A Spark session if successful, None otherwise.
        """
        try:
            spark_session = DatabricksSession.builder.getOrCreate()
            logging.info("Spark session created successfully.")
            return spark_session
        except Exception:
            logging.error("Unable to create Spark session.")
            raise

    def write_delta(
        self,
        df: pd.DataFrame,
        table_name_suffix: str,
        label: t.Optional[str] = "Delta table",
    ) -> None:
        """Writes a DataFrame to a Delta Lake table in the institution's silver schema."""
        if df is None:
            msg = f"{label} is empty: cannot write inference summary tables."
            logging.error(msg)
            raise ValueError(msg)
        table_path = f"{self.args.DB_workspace}.{self.cfg.institution_id}_silver.{table_name_suffix}"
        dataio.write.to_delta_table(
            df=df, table_path=table_path, spark_session=self.spark_session
        )
        logging.info("%s data written to: %s", table_name_suffix, table_path)

    def feature_selection(self, df_preprocessed: pd.DataFrame) -> pd.DataFrame:
        if self.cfg.modeling is None or self.cfg.modeling.feature_selection is None:
            raise ValueError(
                "FEATURE SELECTION SECTION OF MODELING DOES NOT EXIST IN THE CONFIG, PLEASE ADD"
            )

        modeling_cfg = self.cfg.modeling
        fs = modeling_cfg.feature_selection
        force_include_vars = modeling_cfg.feature_selection.force_include_vars
        #confirm each of the force_include_vars are in df_preprocessed columns
        if force_include_vars is not None:
            for var in force_include_vars:
                if var not in df_preprocessed.columns:
                    raise ValueError(
                        f"FORCE INCLUDE VAR {var} NOT FOUND IN PREPROCESSED DATA COLUMNS"
                    )
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

        # create parameter for updated config post model-selection
        self.cfg = dataio.read.read_config(
            self.args.config_file_path,
            schema=configs.pdp.PDPProjectConfig,
        )

    def make_predictions(self):
        cfg = PredConfig(
            model_run_id=self.cfg.model.run_id,
            experiment_id=self.cfg.model.experiment_id,
            split_col=self.cfg.split_col,
            student_id_col=self.cfg.student_id_col,
            pos_label=self.cfg.pos_label,
            min_prob_pos_label=self.cfg.inference.min_prob_pos_label,
            background_data_sample=self.cfg.inference.background_data_sample,
            cfg_inference_params=self.cfg.inference.dict(),
            random_state=self.cfg.random_state,
        )
        paths = PredPaths(
            features_table_path="shared/assets/pdp_features_table.toml",
        )

        try:
            out = run_predictions(
                pred_cfg=cfg, pred_paths=paths, run_type=RunType.TRAIN
            )

            # write gold artifacts
            dataio.write.to_delta_table(
                df=out.top_features_result,
                table_path=self.cfg.datasets.gold.advisor_output.table_path,
                spark_session=self.spark_session,
            )

            # write to silver for FE tables
            self.write_delta(
                df=out.shap_feature_importance,
                table_name_suffix=f"training_{self.cfg.model.run_id}_shap_feature_importance",
                label="Training SHAP Feature Importance table",
            )

            self.write_delta(
                df=out.support_score_distribution,
                table_name_suffix=f"training_{self.cfg.model.run_id}_support_overview",
                label="Training Support Overview table",
            )

            # read modeling parquet for roc table
            modeling_df = dataio.read.read_parquet(
                f"{self.args.silver_volume_path}/modeling.parquet"
            )

            # training-only logging
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_id=self.cfg.model.run_id):
                _ = modeling.evaluation.log_confusion_matrix(
                    institution_id=self.cfg.institution_id,
                    automl_run_id=self.cfg.model.run_id,
                )
                _ = modeling.h2o_ml.evaluation.log_roc_table(
                    institution_id=self.cfg.institution_id,
                    automl_run_id=self.cfg.model.run_id,
                    modeling_df=modeling_df,
                )

        finally:
            if mlflow.active_run():
                try:
                    mlflow.end_run()
                except Exception:
                    pass

    def register_model(self):
        model_name = modeling.registration.get_model_name(
            institution_id=self.cfg.institution_id,
            target=self.cfg.preprocessing.target.name,
            checkpoint=self.cfg.preprocessing.checkpoint.name,
        )
        try:
            modeling.registration.register_mlflow_model(
                model_name,
                self.cfg.institution_id,
                run_id=self.cfg.model.run_id,
                catalog=self.args.DB_workspace,
                registry_uri="databricks-uc",
                mlflow_client=self.client,
            )
        except Exception as e:
            logging.warning("Failed to register model: %s", e)

        return model_name

    def create_model_card(self, model_name):
        # Initialize card
        card = H2OPDPModelCard(
            config=self.cfg,
            catalog=self.args.DB_workspace,
            model_name=model_name,
            mlflow_client=self.client,
        )
        # Build context and download artifacts
        card.build()

        # Reload & publish
        card.reload_card()
        card.export_to_pdf()

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

        logging.info("Generating training predictions & SHAP values")
        self.make_predictions()

        logging.info("Registering model in UC gold volume")
        model_name = self.register_model()

        logging.info("Generating model card")
        self.create_model_card(model_name)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="H2o training for pipeline.")
    parser.add_argument("--DB_workspace", type=str, required=True)
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
