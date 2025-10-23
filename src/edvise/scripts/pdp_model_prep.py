## TODO : edit so it works for training or inference (with and without targets)
## Noreen- I took a stab at this, but idk if it works?

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
from edvise.shared.logger import local_fs_path, resolve_run_path, init_file_logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


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
        LOGGER.info(
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
        LOGGER.info(
            "Sample weight distribution:\n%s",
            df[sample_weight_col].value_counts(normalize=True),
        )
        return df

    def run(self):
        # Ensure correct folder: training or inference
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )

        # Ensure local run path exists
        local_run_path = local_fs_path(current_run_path)
        os.makedirs(local_run_path, exist_ok=True)

        # --- Load required inputs (DBFS-safe) ---
        ckpt_path = os.path.join(current_run_path, "checkpoint.parquet")
        sel_path = os.path.join(current_run_path, "selected_students.parquet")
        ckpt_path_local = local_fs_path(ckpt_path)
        sel_path_local = local_fs_path(sel_path)

        if not os.path.exists(ckpt_path_local):
            raise FileNotFoundError(
                f"Missing checkpoint.parquet at: {ckpt_path} (local: {ckpt_path_local})"
            )
        if not os.path.exists(sel_path_local):
            raise FileNotFoundError(
                f"Missing selected_students.parquet at: {sel_path} (local: {sel_path_local})"
            )

        selected_students = read_parquet(sel_path_local)
        LOGGER.info(
            "Loaded selected_students.parquet with shape %s",
            getattr(selected_students, "shape", None),
        )
        checkpoint_df = read_parquet(ckpt_path_local)
        LOGGER.info(
            "Loaded checkpoint.parquet with shape %s",
            getattr(checkpoint_df, "shape", None),
        )

        # target.parquet may be absent during inference; handle gracefully
        target_df = None
        target_path = os.path.join(current_run_path, "target.parquet")
        target_path_local = local_fs_path(target_path)
        if os.path.exists(target_path_local):
            try:
                target_df = read_parquet(target_path_local)
                LOGGER.info(
                    "Loaded target.parquet with shape %s",
                    getattr(target_df, "shape", None),
                )
            except Exception as e:
                LOGGER.warning(
                    "target.parquet present but failed to read (%s); proceeding without targets.",
                    e,
                )

        # Merge & preprocess
        if target_df is not None:
            df_labeled = self.merge_data(checkpoint_df, target_df, selected_students)
            df_preprocessed = self.cleanup_features(df_labeled)
            # Splits/weights require targets; only apply when present
            df_preprocessed = self.apply_dataset_splits(df_preprocessed)
            df_preprocessed = self.apply_sample_weights(df_preprocessed)
            out_name = "preprocessed.parquet"
            LOGGER.info(
                "Merged target.parquet with selected_students.parquet and checkpoint.parquet into preprocessed.parquet"
            )
            LOGGER.info(
                "preprocessed.parquet with shape %s",
                getattr(df_preprocessed, "shape", None),
            )
        else:
            # Unlabeled path (e.g., inference): merge only checkpoint + selected students, then cleanup
            student_id_col = self.cfg.student_id_col
            df_unlabeled = pd.merge(
                checkpoint_df,
                pd.Series(selected_students.index, name=student_id_col),
                how="inner",
                on=student_id_col,
            )
            df_preprocessed = self.cleanup_features(df_unlabeled)
            out_name = "preprocessed_unlabeled.parquet"

        target_counts = df_preprocessed["target"].value_counts(dropna=False)
        logging.info("Target breakdown (counts):\n%s", target_counts.to_string())

        target_percents = df_preprocessed["target"].value_counts(
            normalize=True, dropna=False
        )
        logging.info("Target breakdown (percents):\n%s", target_percents.to_string())

        cohort_counts = (
            df_preprocessed[["cohort", "cohort_term"]]
            .value_counts(dropna=False)
            .sort_index()
        )
        logging.info(
            "Cohort & Cohort Term breakdowns (counts):\n%s", cohort_counts.to_string()
        )

        cohort_target_counts = (
            df_labeled[["cohort", "target"]].value_counts(dropna=False).sort_index()
        )
        logging.info(
            "Cohort Target breakdown (counts):\n%s", cohort_target_counts.to_string()
        )

        # Write output using custom function
        out_path = os.path.join(current_run_path, out_name)
        write_parquet(
            df_preprocessed,
            file_path=local_fs_path(out_path),
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
    parser.add_argument("--db_run_id", type=str, required=False)
    parser.add_argument(
        "--job_type", type=str, choices=["training", "inference"], required=False
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if not getattr(args, "job_type", None):
        args.job_type = "training"
        logging.info("No --job_type passed; defaulting to job_type='training'.")
    task = ModelPrepTask(args)
    # Attach per-run file logging (log lives under resolved run folder)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="pdp_model_prep.log",
    )
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
