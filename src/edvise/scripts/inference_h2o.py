"""
Databricks job: Model inference for Edvise

Refactored to use the shared predictions pipeline (run_predictions) for:
- loading model + imputer
- aligning features
- scoring
- SHAP computation + grouping
- building output tables

Writes results to Unity Catalog Delta tables and exports CSV.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import typing as t
from email.headerregistry import Address

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

import mlflow
import pandas as pd

from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

# Project modules
import edvise.dataio as dataio
import edvise.modeling as modeling
from edvise.configs.pdp import PDPProjectConfig
from edvise.utils import emails
from edvise.utils.databricks import get_spark_session
from edvise.modeling.inference import top_n_features, features_box_whiskers_table
from edvise.shared.logger import resolve_run_path, local_fs_path
from edvise.shared.validation import (
    validate_tables_exist,
    ExpectedTable,
)

# Shared predictions pipeline (your extracted module)
from edvise.scripts.predictions_h2o import (
    PredConfig,
    PredPaths,
    RunType,
    run_predictions,
)

# Disable mlflow autologging (prevents conflicts in Databricks)
mlflow.autolog(disable=True)

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class ModelInferenceTask:
    """Encapsulates the model inference logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.school_type = args.school_type.strip().lower()
        if self.school_type not in {"pdp", "edvise_custom"}:
            raise ValueError("school_type must be one of: 'pdp', 'edvise_custom'")
        self.spark_session = get_spark_session()
        self.cfg = dataio.read.read_config(
            self.args.config_file_path, schema=PDPProjectConfig
        )
        self.features_table_path = "shared/assets/pdp_features_table.toml"
        # Populated by load_mlflow_model()
        self.model_run_id: str | None = None
        self.model_experiment_id: str | None = None

    def read_config(self, config_file_path: str) -> PDPProjectConfig:
        try:
            return dataio.read.read_config(config_file_path, schema=PDPProjectConfig)
        except FileNotFoundError:
            logging.error("Configuration file not found at %s", config_file_path)
            raise
        except Exception as e:
            logging.error("Error reading configuration file: %s", e)
            raise

    def load_mlflow_model_metadata(self, *, school_type: str) -> None:
        """Discover UC model latest version -> run_id + experiment_id (no model object needed here)."""
        client = MlflowClient(registry_uri="databricks-uc")

        # Assert preprocessing is not None (should be validated by config loading)
        assert self.cfg.preprocessing is not None, "preprocessing config is required"

        if self.school_type == "pdp":
            model_name = modeling.registration.pdp_get_model_name(
                target=self.cfg.preprocessing.target,
                checkpoint=self.cfg.preprocessing.checkpoint,
                student_criteria=self.cfg.preprocessing.selection.student_criteria,
            )
        else:  # edvise_custom
            model_name = modeling.registration.get_model_name(
                institution_id=self.cfg.institution_id,
                target=self.cfg.preprocessing.target.name,
                checkpoint=self.cfg.preprocessing.checkpoint.name,
            )

        full_model_name = (
            f"{self.args.DB_workspace}."
            f"{self.args.databricks_institution_name}_gold."
            f"{model_name}"
        )

        mv = max(
            client.search_model_versions(f"name='{full_model_name}'"),
            key=lambda v: int(v.version),
        )
        self.model_run_id = mv.run_id
        run = client.get_run(self.model_run_id)
        self.model_experiment_id = run.info.experiment_id

        logging.info(
            "Using UC model %s (version=%s) with run_id=%s, experiment_id=%s",
            full_model_name,
            mv.version,
            self.model_run_id,
            self.model_experiment_id,
        )

    def validate_inference_tables(
        self,
        *,
        catalog: str,
        institution_id: str,
        run_id: str,
    ) -> None:
        silver_schema = f"{catalog}.{institution_id}_silver"
        inference_tables = [
            ExpectedTable(
                f"{silver_schema}.inference_{run_id}_features_with_most_impact",
                "inference features with most impact (silver)",
            ),
            ExpectedTable(
                f"{silver_schema}.inference_{run_id}_shap_feature_importance",
                "inference shap table (silver)",
            ),
            ExpectedTable(
                f"{silver_schema}.inference_{run_id}_support_overview",
                "inference support overview table (silver)",
            ),
            ExpectedTable(
                f"{silver_schema}.inference_{run_id}_box_plot_table",
                "inference box plot table (silver)",
            ),
        ]

        validate_tables_exist(self.spark_session, inference_tables)

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

    def run(self) -> None:
        # Enforce inference mode
        if getattr(self.args, "job_type", "inference") != "inference":
            raise ValueError(
                "ModelInferenceTask must be run with --job_type inference."
            )

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

        if self.cfg.model is None or self.cfg.model.run_id is None:
            raise ValueError("cfg.model.run_id must be set for inference runs.")
        # Use canonical per-run folder: <silver>/<run_id>/inference/
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)

        logging.info("Loading UC model metadata (run_id + experiment_id)")
        self.load_mlflow_model_metadata()
        assert self.model_run_id and self.model_experiment_id

        logging.info("Reading the processed dataset")
        preproc_path = os.path.join(current_run_path, "preprocessed.parquet")
        preproc_path_local = local_fs_path(preproc_path)
        if not os.path.exists(preproc_path_local):
            raise FileNotFoundError(
                f"Missing preprocessed.parquet at: {preproc_path} (local: {preproc_path_local})"
            )
        df_processed = dataio.read.read_parquet(preproc_path_local)

        logging.info("Sending kickoff email")
        self._send_kickoff_email()

        logging.info("Configuring + calling the shared predictions pipeline")
        inf = self.cfg.inference
        min_prob_pos_label = (
            0.5
            if inf is None or inf.min_prob_pos_label is None
            else inf.min_prob_pos_label
        )

        background_data_sample = (
            500
            if inf is None or inf.background_data_sample is None
            else inf.background_data_sample
        )
        inference_params = (
            {"num_top_features": 5, "min_prob_pos_label": 0.5}
            if inf is None or inf.dict() is None
            else inf.dict()
        )
        pred_cfg = PredConfig(
            model_run_id=self.model_run_id,
            experiment_id=self.model_experiment_id,
            split_col=self.cfg.split_col,
            student_id_col=self.cfg.student_id_col,
            pos_label=("true" if self.cfg.pos_label is None else self.cfg.pos_label),
            min_prob_pos_label=min_prob_pos_label,
            background_data_sample=background_data_sample,
            random_state=(
                12345 if self.cfg.random_state is None else self.cfg.random_state
            ),
            cfg_inference_params=inference_params,
        )
        # Choose the correct features table path your project uses
        features_table_path = self.features_table_path

        pred_paths = PredPaths(
            features_table_path=features_table_path,
        )

        out = run_predictions(
            pred_cfg=pred_cfg,
            pred_paths=pred_paths,
            run_type=RunType.PREDICT,
            df_inference=df_processed,
        )

        logging.info("Building & writing the 'predicted_dataset' table")
        predicted_df = pd.DataFrame(
            {
                self.cfg.student_id_col: out.unique_ids.values,
                "predicted_prob": out.pred_probs,
                "predicted_label": out.pred_labels,
            }
        )
        self.write_delta(
            df=predicted_df,
            table_name_suffix=f"predicted_dataset_{self.args.db_run_id}",
            label="Prediction dataset",
        )
        support_scores = pd.DataFrame(
            {
                "student_id": out.unique_ids.values,
                "support_score": out.pred_probs,
            }
        )

        logging.info("Creating FE tables")
        inference_features_with_most_impact = top_n_features(
            grouped_features=out.grouped_features,
            unique_ids=out.unique_ids,
            grouped_shap_values=out.grouped_contribs_df,
            features_table_path=features_table_path,
        ).merge(support_scores, on="student_id", how="left")

        box_whiskers_table = features_box_whiskers_table(
            features=out.grouped_features,
            shap_values=out.grouped_contribs_df.values,
            features_table_path=features_table_path,
        )

        logging.info("Writing FE tables")
        tables = {
            f"inference_{self.args.db_run_id}_features_with_most_impact": (
                inference_features_with_most_impact,
                "Inference features with most impact",
            ),
            f"inference_{self.args.db_run_id}_shap_feature_importance": (
                out.shap_feature_importance,
                "Shap Feature Importance",
            ),
            f"inference_{self.args.db_run_id}_support_overview": (
                out.support_score_distribution,
                "Support overview table",
            ),
            f"inference_{self.args.db_run_id}_box_plot_table": (
                box_whiskers_table,
                "Box plot table",
            ),
        }

        for suffix, (df, label) in tables.items():
            self.write_delta(df, suffix, label)

        logging.info("Validating inference tables were created for FE")
        self.validate_inference_tables(
            catalog=self.args.DB_workspace,
            institution_id=self.cfg.institution_id,
            run_id=self.args.db_run_id,  # NOTE: using inference job ID here as the "run id"
        )

        # 8) Export a CSV of the main “inference_output” (use top_features_result here)
        self._export_csv_with_spark(
            df=out.top_features_result,
            dest_dir=f"{self.args.job_root_dir}/ext/",
            basename="inference_output",
        )

        logging.info("Inference task completed successfully.")

    def _send_kickoff_email(self) -> None:
        w = WorkspaceClient()
        MANDRILL_USERNAME = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
        MANDRILL_PASSWORD = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
        SENDER_EMAIL = Address("Datakind Info", "help", "datakind.org")

        emails.send_inference_kickoff_email(
            str(SENDER_EMAIL),
            [self.args.datakind_notification_email],
            [self.args.DK_CC_EMAIL],
            MANDRILL_USERNAME,
            MANDRILL_PASSWORD,
        )

    def _export_csv_with_spark(
        self, df: pd.DataFrame, dest_dir: str, basename: str
    ) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        spark_df = self.spark_session.createDataFrame(df)
        spark_df.coalesce(1).write.format("csv").option("header", "true").mode(
            "overwrite"
        ).save(os.path.join(dest_dir, basename))
        logging.info("Exported CSV to %s", os.path.join(dest_dir, basename))


# ------------------------------
# CLI
# ------------------------------
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Perform model inference for the SST pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--databricks_institution_name", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=True)
    parser.add_argument("--DB_workspace", type=str, required=True)
    parser.add_argument("--job_root_dir", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--datakind_notification_email", type=str, required=True)
    parser.add_argument("--DK_CC_EMAIL", type=str, required=True)
    parser.add_argument("--features_table_path", type=str, required=False)
    parser.add_argument("--ds_run_as", type=str, required=False)
    parser.add_argument(
        "--job_type", type=str, choices=["inference"], default="inference"
    )
    parser.add_argument(
        "--school_type",
        type=str,
        choices=["pdp", "edvise_custom"],
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # # Optional schema hook (kept from your original)
    # try:
    #     if args.custom_schemas_path:
    #         sys.path.append(args.custom_schemas_path)
    #         import importlib

    #         _ = importlib.import_module(f"{args.databricks_institution_name}.schemas")
    #         logging.info("Running task with custom schema")
    # except Exception:
    #     logging.info("Running task with default schema")

    task = ModelInferenceTask(args)
    task.run()
