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
    find_file_in_run_folder,
)


class ESInferenceInputs:
    """
    Resolve inference config from the registered model's silver run folder.

    Same pattern as legacy_inference_inputs, but ES inference only needs config.toml
    (no features_table). GenAI input files come from inputs.toml execute_files on bronze;
    ES data_audit reads GenAI parquets from the execute run via genai_active_registry.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dbutils = get_dbutils_or_none()

    def run(self) -> None:
        model_run_id = get_latest_uc_model_run_id(
            model_name=self.args.model_name,
            workspace=self.args.DB_workspace,
            institution=self.args.databricks_institution_name,
        )
        silver_run_root = (
            f"/Volumes/{self.args.DB_workspace}/"
            f"{self.args.databricks_institution_name}_silver/silver_volume/{model_run_id}"
        )
        logging.info("Looking for inference config in: %s", silver_run_root)

        config_file_path = find_file_in_run_folder(
            silver_run_root,
            keyword="config",
        )
        logging.info("Using config file: %s", config_file_path)

        if self.dbutils:
            self.dbutils.jobs.taskValues.set(
                key="config_file_path",
                value=str(config_file_path),
            )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve ES inference config from the trained model's silver artifacts."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--DB_workspace", required=True)
    parser.add_argument("--databricks_institution_name", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ESInferenceInputs(parse_arguments()).run()
