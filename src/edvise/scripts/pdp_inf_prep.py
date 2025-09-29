import argparse
import pandas as pd
import logging
from databricks.sdk.runtime import dbutils

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

from edvise.model_prep import cleanup_features as cleanup
from edvise.dataio.read import read_parquet, read_config
from edvise.dataio.write import write_parquet
from edvise.configs.pdp import PDPProjectConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferencePrepTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(args.config_file_path, schema=PDPProjectConfig)

    def merge_data(
        self,
        checkpoint_df: pd.DataFrame,
        selected_students: pd.DataFrame,
    ) -> pd.DataFrame:
        # student id col is reqd in config
        student_id_col = self.cfg.student_id_col
        df_inf = pd.merge(
            checkpoint_df,
            pd.Series(selected_students.index, name=student_id_col),
            how="inner",
            on=student_id_col,
        )
        return df_inf

    def cleanup_features(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
        cleaner = cleanup.PDPCleanup()
        return cleaner.clean_up_labeled_dataset_cols_and_vals(df_labeled)

    def run(self):
        if self.cfg.model.run_id is None:
            raise ValueError("cfg.model.run_id must be set for inference runs.")
        current_run_path = f"{self.args.silver_volume_path}/{self.cfg.model.run_id}"
        # Read inputs using custom function
        checkpoint_df = read_parquet(f"{current_run_path}/checkpoint.parquet")
        selected_students = read_parquet(
            f"{current_run_path}/selected_students.parquet"
        )

        df_labeled = self.merge_data(checkpoint_df, selected_students)
        df_preprocessed = self.cleanup_features(df_labeled)

        # Write output using custom function
        write_parquet(
            df_preprocessed,
            file_path=f"{current_run_path}/preprocessed.parquet",
            index=False,
            overwrite=True,
            verbose=True,
        )

        # Setting h2o vs sklearn based on config
        model_type = (
            "h2o" if getattr(self.cfg.model, "framework", None) == "h2o" else "sklearn"
        )
        dbutils.jobs.taskValues.set(key="model_type", value=model_type)
        logger.info(f"Setting task parameter for 'model_type' as '{model_type}'")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model preparation task for SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = InferencePrepTask(args)
    task.run()
