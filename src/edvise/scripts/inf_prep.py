import argparse
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

from edvise.shared.schema_type import is_edvise_schema, project_config_class
from edvise.dataio.read import read_parquet, read_config
from edvise.student_selection.filter_inference import (
    log_inference_selection_breakdown,
    resolve_inference_terms_from_param,
    select_inference_students,
)
from edvise.dataio.write import write_parquet
from edvise.shared.logger import resolve_run_path, local_fs_path, init_file_logging
from edvise.shared.utils import cohort_pair_columns, feature_cleanup_for_schema
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
        resolve_inference_terms_from_param(
            self.cfg,
            schema_type=args.schema_type,
            term_filter=getattr(args, "term_filter", None),
            job_type=getattr(args, "job_type", None) or "inference",
        )

    def run(self):
        if self.cfg.model.run_id is None:
            raise ValueError("cfg.model.run_id must be set for inference runs.")
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
        LOGGER.info(
            "Loaded checkpoint.parquet with shape %s",
            getattr(checkpoint_df, "shape", None),
        )
        selected_students = read_parquet(sel_path_local)
        LOGGER.info(
            "Loaded selected_students.parquet with shape %s",
            getattr(selected_students, "shape", None),
        )

        student_id_col = self.cfg.student_id_col
        selected_ids = selected_students.index.to_series(name=student_id_col)
        total_selected = selected_ids.shape[0]
        df_labeled = checkpoint_df.merge(selected_ids, how="inner", on=student_id_col)
        n_checkpoint_ok = df_labeled[student_id_col].nunique()
        LOGGER.info(
            "Checkpoint-evaluable subset: %d/%d criteria-selected students (%.2f%%) meet the checkpoint.",
            n_checkpoint_ok,
            total_selected,
            (n_checkpoint_ok / total_selected * 100) if total_selected else 0.0,
        )

        require(
            not df_labeled.empty,
            "Merge produced 0 labeled rows (checkpoint ∩ selected ∩ selected_students is empty).",
        )
        if self.cfg.inference is None or self.cfg.inference.term is None:
            raise ValueError("cfg.inference.term must be configured.")

        inference = self.cfg.inference
        cohort_pair = cohort_pair_columns(df_labeled)
        training = getattr(getattr(self.cfg.modeling, "training", None), "cohort", None)
        df_selected = select_inference_students(
            df_labeled,
            inf_terms=inference.term,
            preprocessing=self.cfg.preprocessing,
            cohort_pair=cohort_pair,
            training_cohorts=training,
        )
        log_inference_selection_breakdown(df_selected, cohort_pair)

        cleaner = feature_cleanup_for_schema(self.args.schema_type)
        df_preprocessed = cleaner.clean_up_labeled_dataset_cols_and_vals(
            df_selected, cfg=self.cfg
        )

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
