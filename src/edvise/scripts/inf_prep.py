import argparse
import pandas as pd
import logging
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

from edvise.configs.schema_type import is_edvise_schema, project_config_class
from edvise.configs.pdp import InferenceConfig as PDPInferenceConfig
from edvise.configs.es import InferenceConfig as ESInferenceConfig
from edvise.model_prep import cleanup_features as cleanup
from edvise.dataio.read import read_parquet, read_config
from edvise.student_selection.filter_inference import (
    filter_inference_term,
    parse_term_filter_param,
)
from edvise.dataio.write import write_parquet
from edvise.shared.logger import resolve_run_path, local_fs_path, init_file_logging
from edvise.shared.utils import cohort_pair_columns
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


class InferencePrepTask:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg_cls = project_config_class(args.schema_type)
        self.cfg = read_config(args.config_file_path, schema=cfg_cls)
        inference_config_cls = (
            ESInferenceConfig
            if is_edvise_schema(args.schema_type)
            else PDPInferenceConfig
        )

        # Resolve inference cohort from job param or config (term_filter is generic for cohort/graduation)
        if getattr(self.args, "job_type", None) == "inference":
            param_cohort = parse_term_filter_param(
                getattr(self.args, "term_filter", None)
            )
            if param_cohort is not None:
                if self.cfg.inference is None:
                    self.cfg.inference = inference_config_cls(cohort=param_cohort)
                else:
                    self.cfg.inference.term = param_cohort
                LOGGER.info(
                    "Inference cohort source: job param; term_filter=%s", param_cohort
                )
            else:
                LOGGER.info(
                    "Inference cohort source: config; cohort=%s",
                    self.cfg.inference.term if self.cfg.inference else None,
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
        cleaner: cleanup.BaseCleanup
        if is_edvise_schema(self.args.schema_type):
            cleaner = cleanup.ESCleanup()
        else:
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

        log_file_name = (
            "es_inference_prep.log"
            if is_edvise_schema(self.args.schema_type)
            else "pdp_inference_prep.log"
        )
        log_path = init_file_logging(
            self.args,
            self.cfg,
            logger_name=__name__,
            log_file_name=log_file_name,
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

        LOGGER.info(
            " Selecting students for inference, i.e. met the checkpoint in term(s) of interest"
        )
        if self.cfg.inference is None or self.cfg.inference.term is None:
            raise ValueError("cfg.inference.term must be configured.")

        inf_terms = self.cfg.inference.term
        df_selected_terms = filter_inference_term(df_labeled, term_list=inf_terms)

        cohort_pair = cohort_pair_columns(df_selected_terms)
        if cohort_pair is not None:
            cy, ct = cohort_pair
            cohort_counts = (
                df_selected_terms[[cy, ct]].value_counts(dropna=False).sort_index()
            )
            logging.info(
                "Cohort & Cohort Term breakdowns (counts):\n%s",
                cohort_counts.to_string(),
            )

        term_counts = (
            df_selected_terms[["academic_year", "academic_term"]]
            .value_counts(dropna=False)
            .sort_index()
        )
        logging.info(
            "Term breakdowns (counts):\n%s",
            term_counts.to_string(),
        )
        df_preprocessed = self.cleanup_features(df_selected_terms)

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
    parser.add_argument(
        "--schema_type",
        type=str,
        default="pdp",
        help="pdp | edvise | es — selects PDP vs ES project config schema.",
    )
    parser.add_argument("--db_run_id", type=str, required=False)
    parser.add_argument(
        "--term_filter",
        type=str,
        default=None,
        help='JSON list of term/cohort labels (e.g. ["fall 2024-25"]). Omit or null for config default. Used for cohort and graduation models.',
    )
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
