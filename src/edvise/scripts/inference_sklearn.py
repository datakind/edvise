from __future__ import annotations
import argparse
import logging
import os
import sys
from email.headerregistry import Address
import typing as t
import cloudpickle
from joblib import Parallel, delayed
import pydantic as pyd
import pathlib
S = t.TypeVar("S", bound=pyd.BaseModel)

import mlflow
from mlflow.tracking import MlflowClient
from databricks.sdk import WorkspaceClient

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import shap
from sklearn.base import ClassifierMixin

try:
    import tomllib  # type: ignore
except ImportError:
    import tomli as tomllib  # noqa

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

# Project modules
import edvise.modeling as modeling
from edvise.configs.pdp import PDPProjectConfig
from edvise.utils import emails
from edvise.utils.databricks import get_spark_session
from edvise.modeling.inference import top_n_features, features_box_whiskers_table

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
        self.cfg = self.read_config(
            file_path=self.args.config_file_path, schema=PDPProjectConfig
        )
        self.model_type = "sklearn"
        self.features_table = self.read_features_table("shared/assets/pdp_features_table.toml")
        self.inference_params = {"num_top_features": 5, "min_prob_pos_label": 0.5}

    def read_config(self, file_path: str, *, schema: type[S]) -> S:
        """
        Read config from ``file_path`` and validate it using ``schema`` ,
        returning an instance with parameters accessible by attribute.
        """
        try:
            cfg = self.from_toml_file(file_path)
            return schema.model_validate(cfg)
        except FileNotFoundError:
            logging.error("Configuration file not found at %s", file_path)
            raise
        except Exception as e:
            logging.error("Error reading configuration file: %s", e)
            raise

    def from_toml_file(self, file_path: str) -> dict[str, object]:
        """
        Read data from ``file_path`` and return it as a dict.

        Args:
            file_path: Path to file on disk from which data will be read.
        """
        fpath = pathlib.Path(file_path).resolve()
        with fpath.open(mode="rb") as f:
            data = tomllib.load(f)
        logging.info("loaded config from '%s'", fpath)
        assert isinstance(data, dict)  # type guard
        return data
    
    def read_features_table(self, file_path: str) -> dict[str, dict[str, str]]:
        """
        Read a features table mapping columns to readable names and (optionally) descriptions
        from a TOML file located at ``fpath``, which can either refer to a relative path in this
        package or an absolute path loaded from local disk.

        Args:
            file_path: Path to features table TOML file relative to package root or absolute;
                for example: "assets/pdp/features_table.toml" or "/path/to/features_table.toml".
        """
        current_path = pathlib.Path().resolve()
        pkg_root_dir = next(
            p for p in current_path.parents if p.parts[-1] == "edvise"
        )
        fpath = (
            pathlib.Path(file_path)
            if pathlib.Path(file_path).is_absolute()
            else pkg_root_dir / file_path
        )
        features_table = self.from_toml_file(str(fpath))
        return features_table  # type: ignore


    def load_mlflow_model(self):
        """Loads the MLflow model."""
        client = MlflowClient(registry_uri="databricks-uc")
        model_name = modeling.registration.get_model_name(
            institution_id=self.cfg.institution_id,
            target=self.cfg.preprocessing.target.name,  # type: ignore
            checkpoint=self.cfg.preprocessing.checkpoint.name,  # type: ignore
        )
        full_model_name = f"{self.args.DB_workspace}.{self.args.databricks_institution_name}_gold.{model_name}"
        logging.info("Loading model '%s' from MLflow Model Registry", full_model_name)
        # List all versions of the model
        all_versions = client.search_model_versions(f"name='{full_model_name}'")

        # Sort by version number and create uri for latest model
        latest_version = max(all_versions, key=lambda v: int(v.version))
        model_uri = f"models:/{full_model_name}/{latest_version.version}"

        logging.info(f"Pandas version: {pd.__version__}")
        logging.info(f"Cloudpickle version: {cloudpickle.__version__}")

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
        chunks_count = max(1, len(df_features) // chunk_size)
        chunks = np.array_split(df_features, chunks_count)

        results = Parallel(n_jobs=n_jobs)(
            delayed(explainer)(chunk) for chunk in chunks
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
            shap_ref_data_size = 100  # Consider getting from config.
            #make sure self.cfg.model.experiment_id and self.cfg.split_col are not None
            if self.cfg.model.experiment_id is None:
                raise ValueError("Missing 'model.experiment_id' in config.")
            if self.cfg.split_col is None:
                raise ValueError("Missing 'split_col' in config.")
            #pull "train" observations from the training dataset for ref dataset
            df_train = modeling.evaluation.extract_training_data_from_model(self.cfg.model.experiment_id).loc[
                lambda df: df[self.cfg.split_col].eq("train")
            ]           
            # SHAP can't explain models using data with nulls
            # so, impute nulls using the mode (most frequent values)
            train_mode = df_train.mode().iloc[0] 
            # sample training dataset as "reference" data for SHAP Explainer
            df_ref = (
                df_train.sample(
                    n=min(shap_ref_data_size, df_train.shape[0]),
                    random_state=self.cfg.random_state,
                )
                .fillna(train_mode)
                .loc[:, model_feature_names]
            )
            
            # Ensure df_ref has the same dtypes as df_processed
            ref_dtypes = df_ref.dtypes.apply(lambda dt: dt.name).to_dict()

            # Initialize SHAP KernelExplainer
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

            # Calculate SHAP values in parallel
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
    
    def _export_csv_with_spark(
        self, df: pd.DataFrame, dest_dir: str, basename: str
    ) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        spark_df = self.spark_session.createDataFrame(df)
        spark_df.coalesce(1).write.format("csv").option("header", "true").mode(
            "overwrite"
        ).save(os.path.join(dest_dir, basename))
        logging.info("Exported CSV to %s", os.path.join(dest_dir, basename))


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

        # 1) Read the processed dataset
        df_processed = dataio.read.read_parquet(
            f"{self.args.silver_volume_path}/preprocessed.parquet"
        )
        # HACK: subset for testing
        df_processed = df_processed[:30]

        if not hasattr(self.cfg, "student_id_col") or self.cfg.student_id_col is None:
            raise ValueError("Missing 'student_id_col' in config.")
        unique_ids = df_processed[self.cfg.student_id_col]

        # 2. Load model from MLflow
        model = self.load_mlflow_model()
        model_feature_names = model.named_steps["column_selector"].get_params()["cols"]

        # 3. Email users 
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

        # 4. Perform predictions and save out predicted values
        df_predicted = self.predict(model, df_processed)
        dataio.write.to_delta_table(df_predicted, f"{self.args.silver_table_path}.predicted_dataset", self.spark_session)   

        # 5.SHAP Values Calculation 
        shap_values = self.calculate_shap_values(
            model, df_processed, model_feature_names
        )

        # 6. Generate and save inference summary tables and SHAP plots
        if shap_values is not None:  # Proceed only if SHAP values were calculated
            logging.info(f"now cfg.model.run_id = {self.cfg.model.run_id}")
            logging.info(
                f"now cfg.model.experiment_id = {self.cfg.model.experiment_id}"
            )
            with mlflow.start_run(run_id=self.cfg.model.run_id):

                # Inference Features with Most Impact Table
                support_scores = pd.DataFrame(
                    {
                        "student_id": unique_ids.values, 
                        "support_score": df_predicted["predicted_prob"].values,
                    }
                )          
                inference_features_with_most_impact = top_n_features(
                    df_processed[model_feature_names], unique_ids, shap_values.values
                ).merge(support_scores, on="student_id", how="left")

                # SHAP Feature Importance Table
                shap_feature_importance = modeling.automl.inference.generate_ranked_feature_table(
                    features = df_processed[model_feature_names], 
                    shap_values = shap_values.values, 
                    features_table = self.features_table
                )

                # Support Overview Table
                support_overview_table = modeling.automl.inference.support_score_distribution_table(
                    df_serving = df_processed[model_feature_names],
                    unique_ids = unique_ids,
                    pred_probs = df_predicted['predicted_prob'],
                    shap_values = shap_values,
                    inference_params  = self.inference_params,
                    features_table = self.features_table,
                )

                # Box and Whiskers Table
                box_whiskers_table = features_box_whiskers_table(
                    features=df_processed[model_feature_names],
                    shap_values=
                    shap_values.values,
                ) 

                # Save tables to Delta Lake              
                tables = {
                    f"inference_{self.args.db_run_id}_features_with_most_impact": (
                        inference_features_with_most_impact,
                        "Inference features with most impact",
                    ),
                    f"inference_{self.args.db_run_id}_shap_feature_importance": (
                        shap_feature_importance,
                        "Shap Feature Importance",
                    ),
                    f"inference_{self.args.db_run_id}_support_overview": (
                        support_overview_table,
                        "Support overview table",
                    ),
                    f"inference_{self.args.db_run_id}_box_plot_table": (
                        box_whiskers_table,
                        "Box plot table",
                    ),
                }

                for suffix, (df, label) in tables.items():
                    if df is None:
                        msg = f"{label} is empty: cannot write inference summary tables."
                        logging.error(msg)
                        raise ValueError(msg)
                    table_path = f"{self.args.silver_table_path}.{suffix}"
                    dataio.write.to_delta_table(df, table_path, self.spark_session)   

                
                # Export a CSV of the main inference output
                shap_results = modeling.automl.inference.select_top_features_for_display(
                    df_serving = df_processed[model_feature_names],
                    unique_ids = unique_ids,
                    pred_probs = df_predicted['predicted_prob'],
                    shap_values = shap_values.values,
                    n_features = self.inference_params["num_top_features"],
                    features_table = self.features_table,
                    needs_support_threshold_prob = self.inference_params["min_prob_pos_label"],
                )

                # --- Save Results to ext/ folder in Gold volume. ---
                if shap_results is not None:
                    self._export_csv_with_spark(
                        shap_results,
                        dest_dir=f"{self.args.job_root_dir}/ext/",
                        basename="inference_output",
                    )

                    # Write the DataFrame to Unity Catalog table
                    dataio.write.to_delta_table(shap_results, f"{self.args.silver_table_path}.inference_output")

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
    parser.add_argument("--silver_table_path", type=str, required=True)
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--db_run_id", type=str, required=True, help="Databricks run ID")
    parser.add_argument("--ds_run_as", type=str, required=False)
    parser.add_argument("--DB_workspace", type=str, required=True)
    parser.add_argument("--databricks_institution_name", type=str, required=True)
    parser.add_argument("--gold_table_path", type=str, required=True)
    parser.add_argument("--datakind_notification_email",type=str,required=True,help="DK's email used for notifications")
    parser.add_argument("--DK_CC_EMAIL", type=str, required=True, help="Datakind email address CC'd")
    parser.add_argument("--job_root_dir", type=str, required=True, help="Folder path to store job output files")
    
   
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = ModelInferenceTask(args)
    task.run()
