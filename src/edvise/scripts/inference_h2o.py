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

import mlflow
import numpy as np
import numpy.typing as npt
import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession

from databricks.sdk import WorkspaceClient
from mlflow.tracking import MlflowClient

# Project modules
import edvise.dataio as dataio
import edvise.modeling as modeling
from edvise.configs.pdp import PDPProjectConfig
from edvise.utils import emails


# Shared predictions pipeline (your extracted module)
from .predictions_h2o import (
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
        self.spark_session = self.get_spark_session()
        self.cfg = dataio.read.read_config(
            self.args.toml_file_path, schema=PDPProjectConfig
        )

        # Populated by load_mlflow_model()
        self.model_run_id: str | None = None
        self.model_experiment_id: str | None = None

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

    def read_config(self, toml_file_path: str) -> PDPProjectConfig:
        try:
            return dataio.read.read_config(toml_file_path, schema=PDPProjectConfig)
        except FileNotFoundError:
            logging.error("Configuration file not found at %s", toml_file_path)
            raise
        except Exception as e:
            logging.error("Error reading configuration file: %s", e)
            raise

    def top_n_features(
        self,
        grouped_features: pd.DataFrame,
        unique_ids: pd.Series,
        grouped_shap_values: npt.NDArray[np.float64] | pd.DataFrame,  # relax input
        features_table_path: str,
        n: int = 10,
    ) -> pd.DataFrame:
        features_table = dataio.read.read_features_table(features_table_path)
        try:
            top_n_shap_features = modeling.automl.inference.top_shap_features(
                features=grouped_features,
                unique_ids=unique_ids,
                shap_values=(
                    grouped_shap_values.values
                    if isinstance(grouped_shap_values, pd.DataFrame)
                    else grouped_shap_values
                ),
                top_n=n,
                features_table=features_table,
            )
            return top_n_shap_features
        except Exception as e:
            logging.error("Error computing top %d shap features table: %s", n, e)
            raise  # keep the signature honest

    def features_box_whiskers_table(
        self,
        features: pd.DataFrame,
        shap_values: npt.NDArray[np.float64],
    ) -> pd.DataFrame:
        features_table = dataio.read.read_features_table(
            "assets/pdp/features_table.toml"
        )
        try:
            feature_boxstats = modeling.automl.inference.top_feature_boxstats(
                features=features,
                shap_values=shap_values,
                features_table=features_table,
            )
            return feature_boxstats

        except Exception as e:
            logging.error("Error computing box features %d shap features table: %s", e)
            return None

    def load_mlflow_model_metadata(self) -> None:
        """Discover UC model latest version -> run_id + experiment_id (no model object needed here)."""
        client = MlflowClient(registry_uri="databricks-uc")
        full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{self.args.model_name}"

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
        table_path = f"{self.args.catalog}.{self.cfg.institution_id}_silver.{table_name_suffix}"
        dataio.write.to_delta_table(
            df=df, table_path=table_path, spark_session=self.spark_session
        )
        logging.info("%s data written to: %s", table_name_suffix, table_path)

    def run(self) -> None:
        # 1) Load UC model metadata (run_id + experiment_id)
        self.load_mlflow_model_metadata()
        assert self.model_run_id and self.model_experiment_id

        # 2) Read the processed dataset
        df_processed = dataio.read.read_parquet(self.args.processed_dataset_path)

        # 3) Notify via email
        self._send_kickoff_email()

        # 4) Configure + call the shared predictions pipeline
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
        features_table_path = (
            getattr(self.args, "features_table_path", None)
            or "assets/pdp/features_table.toml"
        )

        pred_paths = PredPaths(
            silver_modeling_path=self.cfg.datasets.silver.modeling.table_path
            if hasattr(self.cfg, "datasets")
            else None,
            features_table_path=features_table_path,
        )

        out = run_predictions(
            pred_cfg=pred_cfg,
            pred_paths=pred_paths,
            run_type=RunType.PREDICT,
            df_inference=df_processed,
        )

        # 5) Build + write the "predicted_dataset" table
        predicted_df = pd.DataFrame(
            {
                self.cfg.student_id_col: out.unique_ids.values,
                "predicted_prob": out.pred_probs.values,
                "predicted_label": out.pred_labels.values,
            }
        )
        self.write_delta(
            df=predicted_df,
            table_name_suffix="predicted_dataset",
            label="Prediction dataset",
        )
        support_scores = pd.DataFrame(
            {
                "student_id": out.unique_ids.values,
                "support_score": out.pred_probs.values,
            }
        )
        # 6) Create FE tables
        inference_features_with_most_impact = self.top_n_features(
            grouped_features=out.grouped_features,
            unique_ids=out.unique_ids,
            grouped_shap_values=out.grouped_contribs_df,
            features_table_path=features_table_path,
        ).merge(support_scores, on="student_id", how="left")
        box_whiskers_table = self.features_box_whiskers_table(
            features=out.grouped_features,
            shap_values=out.grouped_contribs_df.values,
        )

        # 7) Write FE tables
        tables = {
            f"inference_{self.model_run_id}_features_with_most_impact": (
                inference_features_with_most_impact,
                "Inference features with most impact",
            ),
            f"inference_{self.model_run_id}_shap_feature_importance": (
                out.shap_feature_importance,
                "Shap Feature Importance",
            ),
            f"inference_{self.model_run_id}_support_overview": (
                out.support_score_distribution,
                "Support overview table",
            ),
            f"inference_{self.model_run_id}_box_plot_table": (
                box_whiskers_table,
                "Box plot table",
            ),
        }

        for suffix, (df, label) in tables.items():
            self.write_delta(df, suffix, label)

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
            [self.args.notification_email],
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
    parser.add_argument("--DB_workspace", type=str, required=True)
    parser.add_argument("--databricks_institution_name", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=True)
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--job_root_dir", type=str, required=True)
    parser.add_argument("--toml_file_path", type=str, required=True)
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--processed_dataset_path", type=str, required=True)
    parser.add_argument("--notification_email", type=str, required=True)
    parser.add_argument("--DK_CC_EMAIL", type=str, required=True)
    # Optional / legacy:
    parser.add_argument("--modeling_table_path", type=str, required=False)
    parser.add_argument("--custom_schemas_path", type=str, required=False)
    parser.add_argument("--features_table_path", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Optional schema hook (kept from your original)
    try:
        if args.custom_schemas_path:
            sys.path.append(args.custom_schemas_path)
            import importlib

            _ = importlib.import_module(f"{args.databricks_institution_name}.schemas")
            logging.info("Running task with custom schema")
    except Exception:
        logging.info("Running task with default schema")

    task = ModelInferenceTask(args)
    task.run()
