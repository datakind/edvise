from __future__ import annotations
import argparse
import logging
import os
import sys
from email.headerregistry import Address
import typing as t
from joblib import Parallel, delayed

import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import shap
from sklearn.base import ClassifierMixin

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

# Project modules
import edvise.dataio as dataio
import edvise.modeling as modeling
from edvise.configs.pdp import PDPProjectConfig
from edvise.utils import emails
from edvise.utils.databricks import get_spark_session
from edvise.modeling.inference import top_n_features, features_box_whiskers_table
from edvise.modeling.evaluation import plot_shap_beeswarm

from edvise.dataio.read import read_config

# Disable mlflow autologging (prevents conflicts in Databricks)
mlflow.autolog(disable=True)

# Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class ModelInferenceTask:
    """Encapsulates the model inference logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        """Initializes the ModelInferenceTask."""
        self.args = args
        self.spark_session = get_spark_session()
        self.cfg = read_config(
            file_path=self.args.config_file_path, schema=PDPProjectConfig
        )
        self.model_type = "sklearn"

    def load_mlflow_model(self):
        """Loads the MLflow model."""
        client = MlflowClient(registry_uri="databricks-uc")
        model_name = modeling.registration.get_model_name(
            institution_id=self.cfg.institution_id,
            target=self.cfg.preprocessing.target.name,  # type: ignore
            checkpoint=self.cfg.preprocessing.checkpoint.name,  # type: ignore
        )
        full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{model_name}"

        # List all versions of the model
        all_versions = client.search_model_versions(f"name='{full_model_name}'")

        # Sort by version number and create uri for latest model
        latest_version = max(all_versions, key=lambda v: int(v.version))
        model_uri = f"models:/{full_model_name}/{latest_version.version}"

        try:
            load_model_func = {
                "sklearn": mlflow.sklearn.load_model,
                "xgboost": mlflow.xgboost.load_model,
                "lightgbm": mlflow.lightgbm.load_model,
                "pyfunc": mlflow.pyfunc.load_model,  # Default
            }.get(self.model_type, mlflow.pyfunc.load_model)
            model = load_model_func(model_uri)
            logging.info(
                "MLflow '%s' model loaded from '%s'", self.model_type, model_uri
            )
            return model
        except Exception as e:
            logging.error("Error loading MLflow model: %s", e)
            raise  # Critical error; re-raise to halt execution

    def predict(self, model: ClassifierMixin, df: pd.DataFrame) -> pd.DataFrame:
        """Performs inference and adds predictions to the DataFrame."""
        try:
            model_feature_names = model.named_steps["column_selector"].get_params()[
                "cols"
            ]
        except AttributeError:
            model_feature_names = model.metadata.get_input_schema().input_names()
        df_serving = df[model_feature_names]
        df_predicted = df_serving.copy()
        df_predicted["predicted_label"] = model.predict(df_serving)
        df_predicted["predicted_prob"] = model.predict_proba(df_serving)[:, 1]
        return df_predicted

    def write_data_to_delta(self, df: pd.DataFrame, table_name_suffix: str) -> None:
        """Writes a DataFrame to a Delta Lake table."""
        write_schema = f"{self.args.databricks_institution_name}_silver"
        table_path = f"{self.args.DB_workspace}.{write_schema}.{table_name_suffix}"

        try:
            dataio.write.to_delta_table(
                df, table_path, spark_session=self.spark_session
            )
            logging.info(
                "%s data written to: %s", table_name_suffix.capitalize(), table_path
            )
        except Exception as e:
            logging.error(
                "Error writing %s data to Delta Lake: %s", table_name_suffix, e
            )
            raise

    @staticmethod
    def predict_proba(
        X: pd.DataFrame | NDArray[np.floating],
        model: ClassifierMixin,
        feature_names: t.Optional[t.Sequence[str]] = None,
        pos_label: t.Optional[bool | str] = None,
    ) -> NDArray[np.floating]:
        if feature_names is None:
            feature_names = model.named_steps["column_selector"].get_params()["cols"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(data=X, columns=feature_names)

        pred_probs = t.cast(NDArray[np.floating], model.predict_proba(X))

        if pos_label is not None:
            idx = t.cast(list[object], model.classes_.tolist()).index(
                t.cast(object, pos_label)
            )
            return t.cast(NDArray[np.floating], pred_probs[:, idx])
        else:
            return t.cast(NDArray[np.floating], pred_probs)

    def parallel_explanations(
        self,
        model: ClassifierMixin,
        df_features: pd.DataFrame,
        explainer: shap.Explainer,
        model_feature_names: t.List[str],
        n_jobs: t.Optional[int] = -1,
    ) -> shap.Explanation:
        """
        Calculates SHAP explanations in parallel using joblib.

        Args:
            model: mlflow.pyfunc.PyFuncModel.
            df_features pd.DataFrame: The feature dataset to calculate SHAP values for.
            explainer (shap.Explainer): The SHAP explainer object.
            model_feature_names (List[str]): List of feature names corresponding to the columns in `df_features`.
            n_jobs (Optional[int]): The number of jobs to run in parallel. Defaults to -1 (use all available CPUs).

        Returns:
            shap.Explanation: The combined SHAP explanation object.
        """

        logging.info("Calculating SHAP values for %s records", len(df_features))

        chunk_size = 10
        chuncks_count = max(1, len(df_features) // chunk_size)
        chunks = np.array_split(df_features, chuncks_count)

        results = Parallel(n_jobs=n_jobs)(
            delayed(lambda model, chunk, explainer: explainer(chunk))(
                model, chunk, explainer
            )
            for chunk in chunks
        )

        combined_values = np.concatenate([r.values for r in results], axis=0)
        combined_data = np.concatenate([r.data for r in results], axis=0)
        combined_explanation = shap.Explanation(
            values=combined_values,
            data=combined_data,
            feature_names=model_feature_names,
        )
        return combined_explanation

    def calculate_shap_values(
        self,
        model: ClassifierMixin,
        df_processed: pd.DataFrame,
        model_feature_names: list[str],
    ) -> pd.DataFrame | None:
        """Calculates SHAP values."""

        try:
            # --- SHAP Values Calculation ---
            # TODO: Consider saving the explainer during training.
            shap_ref_data_size = 100  # Consider getting from config.

            df_train = dataio.read.read_parquet(
                f"{self.args.silver_volume_path}/preprocessed.parquet"
            )
            train_mode = df_train.mode().iloc[0]  # Use .iloc[0] for single row
            df_ref = (
                df_train.sample(
                    n=min(shap_ref_data_size, df_train.shape[0]),
                    random_state=self.cfg.random_state,
                )
                .fillna(train_mode)
                .loc[:, model_feature_names]
            )
            logging.info(
                f"Object dtype columns: {df_ref.select_dtypes(include=['object']).columns.tolist()}"
            )

            ref_dtypes = df_ref.dtypes.apply(lambda dt: dt.name).to_dict()

            explainer = shap.explainers.KernelExplainer(
                lambda x: self.predict_proba(
                    pd.DataFrame(x, columns=model_feature_names).astype(ref_dtypes),
                    model=model,
                    feature_names=model_feature_names,
                    pos_label=self.cfg.pos_label,
                ),
                df_ref.astype(ref_dtypes),
                link="identity",
            )

            shap_values_explanation = self.parallel_explanations(
                model=model,
                df_features=df_processed[model_feature_names],
                explainer=explainer,
                model_feature_names=model_feature_names,
                n_jobs=-1,
            )

            return shap_values_explanation
        except Exception as e:
            logging.error("Error during SHAP value calculation: %s", e)
            raise

    def support_score_distribution(
        self, df_serving, unique_ids, df_predicted, shap_values, model_feature_names
    ):
        """
        Selects top features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None

        # --- Load features table ---
        features_table = dataio.read.read_features_table(
            "assets/pdp/features_table.toml"
        )

        # --- Inference Parameters ---
        inference_params = {
            "num_top_features": 5,
            "min_prob_pos_label": 0.5,
        }

        pred_probs = df_predicted["predicted_prob"]
        # --- Feature Selection for Display ---

        try:
            result = modeling.automl.inference.support_score_distribution_table(
                df_serving,
                unique_ids,
                pred_probs,
                shap_values,
                inference_params=inference_params,
                features_table=features_table,
                model_feature_names=model_feature_names,
            )

            return result

        except Exception as e:
            logging.error("Error computing support score distribution table: %s", e)
            return None

    def inference_shap_feature_importance(self, df_serving, shap_values):
        """
        Selects top important features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None
        features_table = dataio.read.read_features_table(
            "assets/pdp/features_table.toml"
        )
        shap_feature_importance = (
            modeling.automl.inference.generate_ranked_feature_table(
                df_serving, shap_values.values, features_table
            )
        )

        return shap_feature_importance

    def get_top_features_for_display(
        self, df_serving, unique_ids, df_predicted, shap_values, model_feature_names
    ):
        """
        Selects top features to display and store
        """
        if not self.spark_session:
            logging.error(
                "Spark session not initialized. Cannot post process shap values."
            )
            return None

        # --- Load features table ---
        features_table = dataio.read.read_features_table(
            "assets/pdp/features_table.toml"
        )

        # --- Inference Parameters ---
        inference_params = {
            "num_top_features": 5,
            "min_prob_pos_label": 0.5,
        }

        pred_probs = df_predicted["predicted_prob"]
        # --- Feature Selection for Display ---
        try:
            result = modeling.automl.inference.select_top_features_for_display(
                df_serving,
                unique_ids,
                pred_probs,
                shap_values.values,
                n_features=inference_params["num_top_features"],
                features_table=features_table,
                needs_support_threshold_prob=inference_params["min_prob_pos_label"],
            )
            return result

        except Exception as e:
            logging.error("Error top features to display: %s", e)
            return None

    def run(self):
        """Executes the model inference pipeline."""
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

        # 2) Read the processed dataset
        df_processed = dataio.read.read_parquet(
            f"{self.args.silver_volume_path}/preprocessed.parquet"
        )
        # subset for testing
        df_processed = df_processed[:30]
        unique_ids = df_processed[self.cfg.student_id_col]

        model = self.load_mlflow_model()
        model_feature_names = model.named_steps["column_selector"].get_params()["cols"]

        # --- Email notify users ---
        # Uncomment below once we want to enable CC'ing to DK's email.
        # Secrets from Databricks
        w = WorkspaceClient()
        MANDRILL_USERNAME = w.dbutils.secrets.get(scope="sst", key="MANDRILL_USERNAME")
        MANDRILL_PASSWORD = w.dbutils.secrets.get(scope="sst", key="MANDRILL_PASSWORD")
        SENDER_EMAIL = Address("Datakind Info", "help", "datakind.org")
        emails.send_inference_kickoff_email(
            SENDER_EMAIL,
            [self.args.datakind_notification_email],
            [self.args.DK_CC_EMAIL],
            MANDRILL_USERNAME,
            MANDRILL_PASSWORD,
        )

        df_predicted = self.predict(model, df_processed)
        self.write_data_to_delta(df_predicted, "predicted_dataset")

        # --- SHAP Values Calculation ---
        shap_values = self.calculate_shap_values(
            model, df_processed, model_feature_names
        )

        if shap_values is not None:  # Proceed only if SHAP values were calculated
            logging.info(f"now cfg.model.run_id = {self.cfg.model.run_id}")
            logging.info(
                f"now cfg.model.experiment_id = {self.cfg.model.experiment_id}"
            )
            with mlflow.start_run(run_id=self.cfg.model.run_id):
                # full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{self.args.model_name}"
                # --- SHAP Summary Plot ---
                shap_fig = plot_shap_beeswarm(shap_values)

                # Inference_features_with_most_impact TABLE
                inference_features_with_most_impact = top_n_features(
                    df_processed[model_feature_names], unique_ids, shap_values.values
                )
                support_scores = pd.DataFrame(
                    {
                        "student_id": unique_ids.values,  # From the original df_test
                        "support_score": df_predicted["predicted_prob"].values,
                    }
                )
                inference_features_with_most_impact = (
                    inference_features_with_most_impact.merge(
                        support_scores, on="student_id", how="left"
                    )
                )

                # print or log the inference_features_with_most_impact
                logging.info(
                    "Inference features with most impact:\n%s",
                    inference_features_with_most_impact,
                )
                # shap_feature_importance TABLE
                shap_feature_importance = self.inference_shap_feature_importance(
                    df_processed[model_feature_names], shap_values
                )
                # # support_overview TABLE
                support_overview_table = self.support_score_distribution(
                    df_processed[model_feature_names],
                    unique_ids,
                    df_predicted,
                    shap_values,
                    model_feature_names,
                )

                box_whiskers_table = features_box_whiskers_table(
                    features=df_processed[model_feature_names],
                    shap_values=shap_values.values,
                )

                if inference_features_with_most_impact is None:
                    msg = "Inference features with most impact is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                if shap_feature_importance is None:
                    msg = "Shap Feature Importance is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                if support_overview_table is None:
                    msg = "Support overview table is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)
                if box_whiskers_table is None:
                    msg = "Box plot table is empty: cannot write inference summary tables."
                    logging.error(msg)
                    raise Exception(msg)

                self.write_data_to_delta(
                    inference_features_with_most_impact,
                    f"inference_{self.args.db_run_id}_features_with_most_impact",
                )
                self.write_data_to_delta(
                    shap_feature_importance,
                    f"inference_{self.args.db_run_id}_shap_feature_importance",
                )
                self.write_data_to_delta(
                    support_overview_table,
                    f"inference_{self.args.db_run_id}_support_overview",
                )
                self.write_data_to_delta(
                    box_whiskers_table,
                    f"inference_{self.args.db_run_id}_box_plot_table",
                )

                # Shap Result Table
                shap_results = self.get_top_features_for_display(
                    df_processed[model_feature_names],
                    unique_ids,
                    df_predicted,
                    shap_values,
                    model_feature_names,
                )

                # --- Save Results to ext/ folder in Gold volume. ---
                if shap_results is not None:
                    # Specify the folder for the output files to be stored.
                    result_path = f"{self.args.job_root_dir}/ext/"
                    os.makedirs(result_path, exist_ok=True)
                    print("result_path:", result_path)

                    # TODO What is the proper name for the table with the results?
                    # Write the DataFrame to Unity Catalog table
                    self.write_data_to_delta(shap_results, "inference_output")

                    # Write the DataFrame to CSV in the specified volume
                    spark_df = self.spark_session.createDataFrame(shap_results)
                    spark_df.coalesce(1).write.format("csv").option(
                        "header", "true"
                    ).mode("overwrite").save(result_path + "inference_output")
                    # Write the SHAP chart png to the volume
                    shap_fig.savefig(
                        result_path + "shap_chart.png", bbox_inches="tight"
                    )
                else:
                    logging.error(
                        "Empty Shap results, cannot create the SHAP chart and table"
                    )
                    raise Exception(
                        "Empty Shap results, cannot create the SHAP chart and table"
                    )


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform model inference for the SST pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument(
        "--config_file_path", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--db_run_id", type=str, required=True, help="Databricks run ID"
    )
    parser.add_argument(
        "--DB_workspace",
        type=str,
        required=True,
        help="Databricks workspace identifier",
    )
    parser.add_argument(
        "--databricks_institution_name",
        type=str,
        required=True,
        help="Databricks institution name",
    )
    parser.add_argument(
        "--datakind_notification_email",
        type=str,
        required=True,
        help="DK's email used for notifications",
    )
    parser.add_argument(
        "--DK_CC_EMAIL", type=str, required=True, help="Datakind email address CC'd"
    )
    parser.add_argument(
        "--job_root_dir",
        type=str,
        required=True,
        help="Folder path to store job output files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     schemas = importlib.import_module(f"{args.databricks_institution_name}.schemas")
    #     logging.info("Running task with custom schema")
    # except Exception:
    #     print("Running task with default schema")
    #     logging.info("Running task with default schema")
    task = ModelInferenceTask(args)
    task.run()
