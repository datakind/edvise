## TODO : edit so it works for training or inference (with and without targets)

import argparse
import pandas as pd
import logging

from src.edvise.model_prep import cleanup_features as cleanup, training_params
from src.edvise.dataio.read import read_parquet, read_config
from src.edvise.dataio.write import write_parquet
from src.edvise.configs.pdp import PDPProjectConfig


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
        student_id_col = self.cfg.student_id_col
        df_labeled = pd.merge(
            checkpoint_df,
            pd.Series(selected_students.index, name=student_id_col),
            how="inner",
            on=student_id_col,
        )
        df_labeled = pd.merge(df_labeled, target_df, how="inner", on=student_id_col)
        return df_labeled

    def cleanup_features(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
        cleaner = cleanup.PDPCleanup()
        return cleaner.clean_up_labeled_dataset_cols_and_vals(df_labeled)

    def apply_dataset_splits(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            splits = self.cfg.preprocessing.splits
            split_col = self.cfg.split_col
        except AttributeError:
            splits = {"train": 0.6, "test": 0.2, "validate": 0.2}
            split_col = "split"

        df[split_col] = training_params.compute_dataset_splits(
            df, label_fracs=splits, seed=self.cfg.random_state
        )
        logger.info(
            "Dataset split distribution:\n%s",
            df[split_col].value_counts(normalize=True),
        )
        return df

    def apply_sample_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            sample_class_weight = self.cfg.preprocessing.sample_class_weight
            sample_weight_col = self.cfg.sample_weight_col
        except AttributeError:
            sample_class_weight = "balanced"
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
        checkpoint_df = read_parquet(f"{self.args.silver_volume_path}/checkpoint.parquet")
        target_df = read_parquet(f"{self.args.silver_volume_path}/target.parquet")
        selected_students = read_parquet(
            f"{self.args.silver_volume_path}/selected_students.parquet"
        )

        df_labeled = self.merge_data(checkpoint_df, target_df, selected_students)
        df_preprocessed = self.cleanup_features(df_labeled)
        df_preprocessed = self.apply_dataset_splits(df_preprocessed)
        df_preprocessed = self.apply_sample_weights(df_preprocessed)

        # Write output using custom function
        write_parquet(
            df_preprocessed,
            file_path=f"{self.args.silver_volume_path}/preprocessed.parquet",
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
