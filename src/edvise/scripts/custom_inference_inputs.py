import os
import argparse
import logging
import pathlib
from mlflow.tracking import MlflowClient


def local_fs_path(p: str) -> str:
    """Convert dbfs path to local filesystem path."""
    return p.replace("dbfs:/", "/dbfs/") if p and p.startswith("dbfs:/") else p


def get_dbutils():
    """Get dbutils if available (Databricks environment)."""
    try:
        from databricks.sdk.runtime import dbutils  # type: ignore

        return dbutils
    except Exception:
        return None


class CustomInferenceInputs:
    """
    Retrieves inference inputs by looking up the trained model and finding paths
    to config and features_table that were used during training. This ensures
    inference uses the exact same configuration as the trained model.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils()

    def get_latest_model_run_id(
        self,
        model_name: str,
        workspace: str,
        institution: str,
        registry_uri: str = "databricks-uc",
    ) -> str:
        """
        Returns the run ID of the latest version of a model registered in Unity Catalog.

        Args:
            model_name: Short name of the model (without UC path).
            workspace: Unity Catalog workspace (e.g. 'edvise').
            institution: Institution name used in the UC model path.
            registry_uri: Registry URI; defaults to 'databricks-uc'.

        Returns:
            run_id: The run ID associated with the latest version of the model.
        """
        client = MlflowClient(registry_uri=registry_uri)
        full_model_name = f"{workspace}.{institution}_gold.{model_name}"

        versions = client.search_model_versions(f"name='{full_model_name}'")
        if not versions:
            raise ValueError(
                f"No registered versions found for model: {full_model_name}"
            )

        latest_version = max(versions, key=lambda v: int(v.version))
        logging.info(
            "Found model '%s' version %s with run_id: %s",
            full_model_name,
            latest_version.version,
            latest_version.run_id,
        )
        return str(latest_version.run_id)

    def find_file_in_run_folder(
        self, run_root: str, file_name: str, description: str
    ) -> str:
        """
        Finds a file with the given name under the run root.
        Searches in: run_root, run_root/training/, run_root/inference/

        Args:
            run_root: Root directory of the training run
            file_name: Exact filename to search for (e.g. "config.toml")
            description: Human-readable description for error messages

        Returns:
            Path to the matching file

        Raises:
            FileNotFoundError: If no matching file is found
        """
        candidates = [
            run_root,
            os.path.join(run_root, "training"),
            os.path.join(run_root, "inference"),
        ]

        for cand in candidates:
            base = pathlib.Path(local_fs_path(cand))
            if not base.exists():
                continue
            file_path = base / file_name
            if file_path.exists():
                found_path = str(file_path)
                logging.info("Found %s: %s", description, found_path)
                return found_path

        raise FileNotFoundError(
            f"No {description} named '{file_name}' found under: {run_root} "
            f"(searched run root, training/, inference/)"
        )

    def run(self):
        """
        Main execution:
        1. Look up the model and get its run_id
        2. Find the config file from the training run
        3. Find the features_table file from the training run
        4. Return both paths via dbutils.jobs.taskValues (if available)
        """
        # Get the model's run_id
        model_run_id = self.get_latest_model_run_id(
            self.args.model_name,
            self.args.DB_workspace,
            self.args.databricks_institution_name,
        )

        # Construct the run root path in silver volume
        silver_run_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_silver/silver_volume/{model_run_id}"
        )
        logging.info("Looking for training artifacts in: %s", silver_run_root)

        # Find config file by exact name
        config_file_path = self.find_file_in_run_folder(
            silver_run_root, self.args.config_file_name, "config file"
        )

        # Find features_table file by exact name
        features_table_path = self.find_file_in_run_folder(
            silver_run_root, self.args.features_table_name, "features table"
        )

        logging.info("Using config file: %s", config_file_path)
        logging.info("Using features table: %s", features_table_path)

        # Set task values for downstream tasks (if dbutils available)
        if self.dbutils:
            self.dbutils.jobs.taskValues.set(
                key="config_file_path", value=str(config_file_path)
            )
            self.dbutils.jobs.taskValues.set(
                key="features_table_path", value=str(features_table_path)
            )
            logging.info("Task values set for downstream tasks")
        else:
            logging.warning(
                "dbutils not available - task values not set. "
                "This is expected in local/testing environments."
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve inference inputs (config and features table paths) from trained model."
    )
    parser.add_argument(
        "--model_name",
        required=True,
        help="Name of the registered model (without UC path prefix)",
    )
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace name"
    )
    parser.add_argument(
        "--databricks_institution_name", required=True, help="Institution identifier"
    )
    parser.add_argument(
        "--config_file_name",
        required=True,
        help="Name of the config file to search for (e.g., 'config.toml')",
    )
    parser.add_argument(
        "--features_table_name",
        required=True,
        help="Name of the features table file to search for (e.g., 'features_table.toml')",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    task = CustomInferenceInputs(args)
    task.run()
