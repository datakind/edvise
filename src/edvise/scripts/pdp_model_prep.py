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
from edvise.shared.logger import (
    local_fs_path,
    resolve_run_path,
    init_file_logging,
)
from edvise.shared.validation import require, warn_if
from edvise.utils.update_config import update_training_cohorts

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
        """
        Merge checkpoint and target data for the selected students, with logging:
        1) Which selected students meet the checkpoint
        2) Which of those have target evaluable
        """
        # student id col is read in config
        student_id_col = self.cfg.student_id_col

        # Build a Series of selected IDs with the right column name to merge on
        selected_ids = pd.Series(selected_students.index, name=student_id_col)

        total_selected = selected_ids.shape[0]

        # Subset 1: selected students who meet the checkpoint
        df_checkpoint_ok = pd.merge(
            checkpoint_df,
            selected_ids,
            how="inner",
            on=student_id_col,
        )

        n_checkpoint_ok = df_checkpoint_ok[student_id_col].nunique()
        pct_checkpoint_ok = (
            (n_checkpoint_ok / total_selected * 100) if total_selected else 0.0
        )

        LOGGER.info(
            "Checkpoint subset: %d/%d criteria-selected students (%.2f%%) meet the checkpoint.",
            n_checkpoint_ok,
            total_selected,
            pct_checkpoint_ok,
        )

        # Subset 2: from those, who can have a target evaluated
        df_labeled = pd.merge(
            df_checkpoint_ok,
            target_df,
            how="inner",
            on=student_id_col,
        )

        require(
            not df_labeled.empty,
            "Merge produced 0 labeled rows (checkpoint ∩ selected ∩ target is empty).",
        )

        n_target_ok = df_labeled[student_id_col].nunique()
        base_for_target = max(n_checkpoint_ok, 1)  # avoid div by zero
        pct_target_ok = n_target_ok / base_for_target * 100

        LOGGER.info(
            "Target-evaluable subset: %d/%d criteria-selected and checkpoint-met students (%.2f%%) can have a target evaluated.",
            n_target_ok,
            n_checkpoint_ok,
            pct_target_ok,
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
            cohort_counts = (
                df_labeled[["cohort", "cohort_term"]]
                .value_counts(dropna=False)
                .sort_index()
            )
            logging.info(
                "Cohort & Cohort Term breakdowns (counts):\n%s",
                cohort_counts.to_string(),
            )
            cohort_target_counts = (
                df_labeled[["cohort", "target"]].value_counts(dropna=False).sort_index()
            )
            logging.info(
                "Cohort Target breakdown (counts):\n%s",
                cohort_target_counts.to_string(),
            )

            logging.info("Updating training cohorts in config")
            training_cohorts = (
                df_labeled[["cohort", "cohort_term"]]
                .dropna()
                .astype(str)
                .agg(" ".join, axis=1)
                .str.lower()
                .unique()
                .tolist()
            )
            if training_cohorts is not None:
                update_training_cohorts(
                    config_path=self.args.config_file_path,
                    training_cohorts=training_cohorts,
                    extra_save_paths=[
                        os.path.join(current_run_path, self.args.config_file_name)
                    ],
                )
            else:
                logging.info("There are no cohorts selected for training.")

            df_preprocessed = self.cleanup_features(df_labeled)
            # Splits/weights require targets; only apply when present
            df_preprocessed = self.apply_dataset_splits(df_preprocessed)
            df_preprocessed = self.apply_sample_weights(df_preprocessed)
            out_name = "preprocessed.parquet"

            require(
                not df_preprocessed.empty,
                "preprocessed.parquet is empty after cleanup/splits/weights.",
            )

            # Must-have columns
            require(
                self.cfg.target_col in df_preprocessed.columns,
                f"Missing target column '{self.cfg.target_col}' in preprocessed dataset.",
            )
            require(
                (df_preprocessed[self.cfg.target_col].dtype == bool)
                or (
                    df_preprocessed[self.cfg.target_col]
                    .dropna()
                    .isin([0, 1, True, False])
                    .all()
                ),
                "Target column is not boolean-like after preprocessing.",
            )

            # Degenerate target (warn)
            vc = df_preprocessed[self.cfg.target_col].value_counts(dropna=False)
            warn_if(
                len(vc) == 1,
                f"Target is degenerate after preprocessing (all {vc.index[0]}).",
            )

            # Split sanity
            split_col = self.cfg.split_col or "split"
            require(
                split_col in df_preprocessed.columns,
                f"Missing split column '{split_col}' after apply_dataset_splits.",
            )
            split_fracs = df_preprocessed[split_col].value_counts(
                normalize=True, dropna=False
            )
            warn_if(
                split_fracs.min() < 0.05,
                f"One split has <5% of rows: {split_fracs.to_dict()}",
            )

            # Sample weight sanity
            sw_col = self.cfg.sample_weight_col or "sample_weight"
            require(
                sw_col in df_preprocessed.columns,
                f"Missing sample weight column '{sw_col}' after apply_sample_weights.",
            )
            require(
                (df_preprocessed[sw_col] > 0).all(), "Sample weights must be positive."
            )

            LOGGER.info(
                "Merged target.parquet with selected_students.parquet and checkpoint.parquet into preprocessed.parquet"
            )
            LOGGER.info(
                "preprocessed.parquet with shape %s",
                getattr(df_preprocessed, "shape", None),
            )
            target_counts = df_preprocessed["target"].value_counts(dropna=False)
            logging.info("Target breakdown (counts):\n%s", target_counts.to_string())

            target_percents = df_preprocessed["target"].value_counts(
                normalize=True, dropna=False
            )
            logging.info(
                "Target breakdown (percents):\n%s", target_percents.to_string()
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

        require(
            not df_preprocessed.empty, f"Refusing to write empty output: {out_name}"
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
    parser.add_argument("--config_file_name", type=str, required=True)

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
