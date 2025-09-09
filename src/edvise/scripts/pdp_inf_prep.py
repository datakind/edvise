## TODO : edit so it works for training or inference (with and without targets)

import typing as t
import argparse
import pandas as pd
import logging

from edvise.model_prep import cleanup_features as cleanup, training_params
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
        # Read inputs using custom function
        checkpoint_df = read_parquet(
            f"{self.args.silver_volume_path}/checkpoint.parquet"
        )
        selected_students = read_parquet(
            f"{self.args.silver_volume_path}/selected_students.parquet"
        )

        df_labeled = self.merge_data(checkpoint_df, selected_students)
        df_preprocessed = self.cleanup_features(df_labeled)

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
    task = InferencePrepTask(args)
    task.run()
