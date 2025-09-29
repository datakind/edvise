## TODO : edit so it works for training or inference (with and without targets)

import typing as t
import argparse
import pandas as pd
import logging
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

from edvise.model_prep import cleanup_features as cleanup, training_params
from edvise.dataio.read import read_parquet, read_config
from edvise.dataio.write import write_parquet
from edvise.configs.pdp import PDPProjectConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPrepTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(args.config_file_path, schema=PDPProjectConfig)

    def merge_data(
        self,
        checkpoint_df: pd.DataFrame,
        target_df: pd.DataFrame,
        selected_students: pd.DataFrame,
    ) -> pd.DataFrame:
        # student id col is reqd in config
        student_id_col = self.cfg.student_id_col
        df_labeled = pd.merge(
            checkpoint_df,
            pd.Series(selected_students.index, name=student_id_col),
            how="inner",
            on=student_id_col,
        )
        df_labeled = pd.merge(df_labeled, target_df, how="inner", on=student_id_col)

        target_counts = df_labeled["target"].value_counts(dropna=False)
        logging.info("Target breakdown (counts):\n%s", target_counts.to_string())

        target_percents = df_labeled["target"].value_counts(
            normalize=True, dropna=False
        )
        logging.info("Target breakdown (percents):\n%s", target_percents.to_string())

        cohort_counts = df_labeled["cohort"].value_counts(dropna=False).sort_index()
        logging.info("Cohort breakdown (counts):\n%s", cohort_counts.to_string())

        cohort_target_pct = (
            df_labeled[["cohort", "target"]]
            .value_counts(dropna=False, normalize=True)
            .sort_index()
        )
        logging.info(
            "Cohort Target breakdown (percents):\n%s", cohort_target_pct.to_string()
        )

        return df_labeled

    def cleanup_features(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
        cleaner = cleanup.PDPCleanup()
        return cleaner.clean_up_labeled_dataset_cols_and_vals(df_labeled)

    def apply_dataset_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessing_cfg = self.cfg.preprocessing
        splits: t.Dict[str, float]

        if preprocessing_cfg is not None and preprocessing_cfg.splits is not None:
            splits = t.cast(t.Dict[str, float], preprocessing_cfg.splits)
        else:
            splits = {"train": 0.6, "test": 0.2, "validate": 0.2}

        if self.cfg.split_col is not None:
            split_col = self.cfg.split_col
        else:
            split_col = "split"

        df[split_col] = training_params.compute_dataset_splits(
            df,
            label_fracs=splits,
            seed=self.cfg.random_state,
            stratify_col=self.cfg.target_col,
        )
        logger.info(
            "Dataset split distribution:\n%s",
            df[split_col].value_counts(normalize=True),
        )
        return df

    def apply_sample_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        prep = self.cfg.preprocessing
        sample_class_weight = None
        if prep is not None and prep.sample_class_weight is not None:
            sample_class_weight = prep.sample_class_weight
        else:
            sample_class_weight = "balanced"
        if self.cfg.sample_weight_col is not None:
            sample_weight_col = self.cfg.sample_weight_col
        else:
            sample_weight_col = "sample_weight"

        df[sample_weight_col] = training_params.compute_sample_weights(
            df,
            target_col=self.cfg.target_col,
            class_weight=sample_class_weight,
        )
        logger.info(
            "Sample weight distribution:\n%s",
            df[sample_weight_col].value_counts(normalize=True),
        )
        return df

    def run(self):
        # Read inputs using custom function
        current_run_path = f"{self.args.silver_volume_path}/current_run"

        checkpoint_df = read_parquet(
            f"{current_run_path}/checkpoint.parquet"
        )
        target_df = read_parquet(f"{current_run_path}/target.parquet")
        selected_students = read_parquet(
            f"{current_run_path}/selected_students.parquet"
        )

        df_labeled = self.merge_data(checkpoint_df, target_df, selected_students)
        df_preprocessed = self.cleanup_features(df_labeled)
        df_preprocessed = self.apply_dataset_splits(df_preprocessed)
        df_preprocessed = self.apply_sample_weights(df_preprocessed)

        # Write output using custom function
        write_parquet(
            df_preprocessed,
            file_path=f"{current_run_path}/preprocessed.parquet",
            index=False,
            overwrite=True,
            verbose=True,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model preparation task for SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = ModelPrepTask(args)
    task.run()
