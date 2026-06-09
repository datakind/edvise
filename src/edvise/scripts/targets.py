"""Target generation for SST pipeline (PDP or Edvise ES configs).

Module :mod:`edvise.scripts.targets` (file ``targets.py``) must not be confused with
the :mod:`edvise.targets` package (compute/retention helpers).

``--schema_type`` selects the project config model (see
:mod:`edvise.configs.schema_type`) and optional preprocessing:

- ``pdp``: :class:`edvise.configs.pdp.PDPProjectConfig` (default).
- ``edvise`` or ``es``: :class:`edvise.configs.es.ESProjectConfig`, and when the
  configured target type is ``retention``, :func:`assign_retention_column` runs
  first so downstream retention logic matches PDP-style inputs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

# Go up 3 levels from the current file's directory to reach repo root
script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")

if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

print("Script dir:", script_dir)
print("Repo root:", repo_root)
print("src_path:", src_path)
print("sys.path:", sys.path)

from edvise.configs.schema_type import is_edvise_schema, project_config_class
from edvise.dataio.read import read_config
from edvise.shared.logger import (
    init_file_logging,
    local_fs_path,
    resolve_run_path,
)
from edvise.shared.validation import require, warn_if
from edvise.targets.invoke import compute_target_from_config
from edvise.targets.retention_edvise import assign_retention_column

logging.getLogger("py4j").setLevel(logging.WARNING)


class TargetsTask:
    """Computes the target variable for an SST pipeline run."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg_cls = project_config_class(args.schema_type)
        self.cfg = read_config(self.args.config_file_path, schema=cfg_cls)

    def target_generation(
        self, df_student_terms: pd.DataFrame, df_ckpt: pd.DataFrame
    ) -> pd.Series:
        preproc = self.cfg.preprocessing
        if preproc is None or preproc.target is None:
            raise ValueError("cfg.preprocessing.target must be configured.")

        target_cfg = preproc.target
        df = df_student_terms
        if is_edvise_schema(self.args.schema_type) and target_cfg.type_ == "retention":
            df = assign_retention_column(df, student_id_col=self.cfg.student_id_col)

        return compute_target_from_config(
            target_cfg,
            df,
            df_ckpt,
            student_id_col=self.cfg.student_id_col,
        )

    def run(self) -> None:
        logging.info("Loading student-terms data...")
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        st_terms_path = os.path.join(current_run_path, "student_terms.parquet")
        ckpt_path = os.path.join(current_run_path, "checkpoint.parquet")
        st_terms_path_local = local_fs_path(st_terms_path)
        ckpt_path_local = local_fs_path(ckpt_path)

        if not os.path.exists(st_terms_path_local):
            raise FileNotFoundError(
                f"Missing student_terms.parquet at: {st_terms_path} (local: {st_terms_path_local})"
            )
        if not os.path.exists(ckpt_path_local):
            raise FileNotFoundError(
                f"Missing checkpoint.parquet at: {ckpt_path} (local: {ckpt_path_local})"
            )

        df_student_terms = pd.read_parquet(st_terms_path_local)
        df_ckpt = pd.read_parquet(ckpt_path_local)

        logging.info("Generating target labels...")
        target_series = self.target_generation(df_student_terms, df_ckpt)

        require(not target_series.empty, "Target generation produced an empty Series.")
        require(
            target_series.dtype == bool,
            f"Target dtype must be bool, got {target_series.dtype}",
        )
        require(
            not target_series.index.has_duplicates,
            "Target Series index contains duplicates; expected one target per entity.",
        )

        vc = target_series.value_counts(dropna=False)
        logging.info("Target distribution:\n%s", vc.to_string())

        warn_if(
            len(vc) == 1,
            f"Target is degenerate (all values are {vc.index[0]}).",
        )

        logging.info("Saving target data...")
        df_target = target_series.reset_index().rename(
            columns={target_series.name: "target"}
        )

        out_path = os.path.join(current_run_path, "target.parquet")
        df_target.to_parquet(local_fs_path(out_path), index=False)
        logging.info("Target file saved to %s", out_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target generation for SST pipeline (PDP or Edvise ES)."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument(
        "--schema_type",
        type=str,
        default="pdp",
        help="pdp | edvise | es — selects config class and Edvise retention preprocessing.",
    )
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

    task = TargetsTask(args)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="targets.log",
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
