import argparse
import logging
import os
import sys

# Ensure repo src/ is on sys.path so `import edvise.*` works in Databricks Jobs.
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from edvise.utils.databricks import (
    get_dbutils_or_none,
    get_latest_uc_model_run_id,
    find_file_in_run_folder as find_file_in_run_folder_shared,
)


class LegacyInferenceInputs:
    """
    Retrieves inference inputs by looking up the trained model and finding paths
    to config and features_table that were used during training. This ensures
    inference uses the exact same configuration as the trained model.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils_or_none()

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
        latest_version = get_latest_uc_model_run_id(
            model_name=model_name,
            workspace=workspace,
            institution=institution,
            registry_uri=registry_uri,
        )
        logging.info(
            "Found latest run_id for model '%s.%s_gold.%s': %s",
            workspace,
            institution,
            model_name,
            latest_version,
        )
        return latest_version

    def find_file_in_run_folder(
        self,
        run_root: str,
        file_name: str | None,
        description: str,
        keyword: str | None = None,
    ) -> str:
        """
        Finds a file with the given name under the run root.
        Searches in: run_root, run_root/training/, run_root/inference/

        If exact match fails, falls back to searching for .toml files containing
        the specified keyword in the filename (e.g., "config" or "features_table").

        Args:
            run_root: Root directory of the training run
            file_name: Optional exact filename to search for (e.g. "config.toml").
                If None or not found, falls back to searching for files containing keyword.
            description: Human-readable description for error messages
            keyword: Keyword to search for in filename when doing fallback search
                (e.g. "config" or "features_table"). If None, searches for any .toml file.

        Returns:
            Path to the matching file

        Raises:
            FileNotFoundError: If no matching file is found
        """
        try:
            found_path = find_file_in_run_folder_shared(
                run_root,
                file_name=file_name,
                keyword=keyword,
            )
            logging.info("Found %s at: %s", description, found_path)
            return found_path
        except FileNotFoundError as exc:
            error_msg = (
                f"No {description} found under: {run_root} "
                f"(searched run root, training/, inference/)"
            )
            if file_name:
                error_msg += f" with name '{file_name}'"
            if keyword:
                error_msg += f" or containing '{keyword}' in filename"
            raise FileNotFoundError(error_msg) from exc

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

        # Same pattern as PDP ingestion: resolve artifacts from the registered model's
        # silver run folder only (no job parameters). Legacy runs store two TOMLs
        # (project config + features table), so we disambiguate by filename keyword.
        config_file_path = self.find_file_in_run_folder(
            silver_run_root,
            None,
            "config file",
            keyword="config",
        )
        features_table_path = self.find_file_in_run_folder(
            silver_run_root,
            None,
            "features table",
            keyword="features_table",
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
        help=(
            "Registered model short name, or full UC three-level name "
            "(catalog.institution_gold.<short>)."
        ),
    )
    parser.add_argument(
        "--DB_workspace", required=True, help="Databricks workspace name"
    )
    parser.add_argument(
        "--databricks_institution_name", required=True, help="Institution identifier"
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = parse_arguments()

    task = LegacyInferenceInputs(args)
    task.run()
