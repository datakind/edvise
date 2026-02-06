import argparse
import pandas as pd
import logging
import json
import typing as t
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
from edvise.data_audit.cohort_selection import select_inference_cohort
from edvise.dataio.write import write_parquet
from edvise.configs.pdp import PDPProjectConfig
from edvise.shared.logger import resolve_run_path, local_fs_path, init_file_logging
from edvise.shared.validation import (
    require,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger(__name__)
logging.getLogger("py4j").setLevel(logging.WARNING)

def parse_term_filter_param(value: t.Optional[str]) -> t.Optional[list[str]]:
    """Parse --term_filter job param. Treat None, '', 'null' as not provided; else json.loads.
    Empty list after parse -> not provided (use config). Invalid JSON -> raise."""
    if value is None:
        return None
    s = value.strip()
    if s in ("", "null", "None"):
        return None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        LOGGER.error("Invalid JSON for term_filter param: %s", value)
        raise ValueError(f"Invalid JSON for --term_filter: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError("--term_filter must be a JSON list of strings")
    labels = [str(item).strip() for item in parsed if str(item).strip()]
    if not labels:
        return None  # empty list -> use config
    return labels


class InferencePrepTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(args.config_file_path, schema=PDPProjectConfig)

        # Resolve inference cohort from job param or config (term_filter is generic for cohort/graduation)
        if getattr(self.args, "job_type", None) == "inference":
            param_cohort = parse_term_filter_param(
                getattr(self.args, "term_filter", None)
            )
            if param_cohort is not None:
                if self.cfg.inference is None:
                    from edvise.configs.pdp import InferenceConfig

                    self.cfg.inference = InferenceConfig(cohort=param_cohort)
                else:
                    self.cfg.inference.cohort = param_cohort
                LOGGER.info(
                    "Inference cohort source: job param; term_filter=%s", param_cohort
                )
            else:
                LOGGER.info(
                    "Inference cohort source: config; cohort=%s",
                    self.cfg.inference.cohort if self.cfg.inference else None,
                )



    def merge_data(
        self,
        checkpoint_df: pd.DataFrame,
        selected_students: pd.DataFrame,
    ) -> pd.DataFrame:
        # student id col is read in config
        student_id_col = self.cfg.student_id_col

        # Build a Series of selected IDs with the right column name to merge on
        selected_ids = pd.Series(selected_students.index, name=student_id_col)

        total_selected = selected_ids.shape[0]

        # Subset: selected students who meet the checkpoint (checkpoint-evaluable)
        df_inf = pd.merge(
            checkpoint_df,
            selected_ids,
            how="inner",
            on=student_id_col,
        )

        n_checkpoint_ok = df_inf[student_id_col].nunique()
        pct_checkpoint_ok = (
            (n_checkpoint_ok / total_selected * 100) if total_selected else 0.0
        )

        LOGGER.info(
            "Checkpoint-evaluable subset: %d/%d criteria-selected students (%.2f%%) meet the checkpoint.",
            n_checkpoint_ok,
            total_selected,
            pct_checkpoint_ok,
        )
        return df_inf

    def cleanup_features(self, df_labeled: pd.DataFrame) -> pd.DataFrame:
        cleaner = cleanup.PDPCleanup()
        return cleaner.clean_up_labeled_dataset_cols_and_vals(df_labeled)

    def run(self):
        # Enforce inference mode & resolve <silver>/<run_id>/inference/
        if self.cfg.model.run_id is None:
            raise ValueError("cfg.model.run_id must be set for inference runs.")
        # If the script is ever reused, require job_type=inference
        if getattr(self.args, "job_type", "inference") != "inference":
            raise ValueError("InferencePrepTask must be run with --job_type inference.")

        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        log_path = init_file_logging(
            self.args,
            self.cfg,
            logger_name=__name__,
            log_file_name="pdp_inference_prep.log",
        )
        LOGGER.info("Per-run log file initialized at %s", log_path)

        # Read inputs (DBFS-safe)
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

        checkpoint_df = read_parquet(ckpt_path_local)
        logging.info(
            "Loaded checkpoint.parquet with shape %s",
            getattr(checkpoint_df, "shape", None),
        )
        selected_students = read_parquet(sel_path_local)
        logging.info(
            "Loaded selected_students.parquet with shape %s",
            getattr(selected_students, "shape", None),
        )

        df_labeled = self.merge_data(checkpoint_df, selected_students)

        require(
            not df_labeled.empty,
            "Merge produced 0 labeled rows (checkpoint ∩ selected ∩ selected_students is empty).",
        )

        LOGGER.info(" Selecting inference cohort")
        if self.cfg.inference is None or self.cfg.inference.cohort is None:
            raise ValueError("cfg.inference.cohort must be configured.")

        inf_cohort = self.cfg.inference.cohort
        df_course_validated = select_inference_cohort(
            df_course_validated, cohorts_list=inf_cohort
        )

        cohort_counts = (
            df_labeled[["cohort", "cohort_term"]]
            .value_counts(dropna=False)
            .sort_index()
        )
        logging.info(
            "Cohort & Cohort Term breakdowns (counts):\n%s",
            cohort_counts.to_string(),
        )
        df_preprocessed = self.cleanup_features(df_labeled)

        # Write output using custom function
        out_path = os.path.join(current_run_path, "preprocessed.parquet")
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
        "--job_type",
        type=str,
        choices=["inference"],
        required=False,
        default="inference",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = InferencePrepTask(args)
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
