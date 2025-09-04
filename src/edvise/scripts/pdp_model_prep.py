## TODO : edit so it works for training or inference (with and without targets)

import argparse
import pandas as pd
import logging

# TODO fix imports
from ..shared.shared import read_config
from ..model_prep import cleanup_features as cleanup, training_params
from dataio import read_parquet, write_parquet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelPrepTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(args.toml_file_path)

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
        checkpoint_df = read_parquet(f"{self.args.checkpoint_path}/checkpoint.parquet")
        target_df = read_parquet(f"{self.args.target_path}/target.parquet")
        selected_students = read_parquet(
            f"{self.args.selection_path}/selected_students.parquet"
        )

        df_labeled = self.merge_data(checkpoint_df, target_df, selected_students)
        df_preprocessed = self.cleanup_features(df_labeled)
        df_preprocessed = self.apply_dataset_splits(df_preprocessed)
        df_preprocessed = self.apply_sample_weights(df_preprocessed)

        # Write output using custom function
        write_parquet(
            df_preprocessed,
            file_path=f"{self.args.output_path}/preprocessed.parquet",
            index=False,
            overwrite=True,
            verbose=True,
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Model preparation task for SST pipeline."
    )
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to config TOML file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to checkpoint data"
    )
    parser.add_argument(
        "--target_path", type=str, required=True, help="Path to target data"
    )
    parser.add_argument(
        "--selection_path", type=str, required=True, help="Path to selected students"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write preprocessed dataset",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = ModelPrepTask(args)
    task.run()
