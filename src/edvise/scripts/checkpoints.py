"""Build ``checkpoint.parquet`` from ``student_terms.parquet`` for PDP or Edvise ES configs.

Checkpoint dispatch uses ``checkpoint.type_`` (not ``isinstance`` on config classes) so the same
logic applies whether the config was parsed as :class:`~edvise.configs.pdp.PDPProjectConfig` or
:class:`~edvise.configs.es.ESProjectConfig` (those modules define duplicate checkpoint classes).

See :mod:`edvise.scripts.targets` for ``--schema_type`` semantics.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Type

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

from edvise import checkpoints
from edvise.configs.es import ESProjectConfig
from edvise.configs.pdp import PDPProjectConfig
from edvise.dataio.read import read_config
from edvise.shared.logger import init_file_logging, local_fs_path, resolve_run_path
from edvise.shared.validation import require

logging.getLogger("py4j").setLevel(logging.WARNING)


def _normalize_schema_type(raw: str) -> str:
    return raw.strip().lower()


def _project_config_class(schema_type: str) -> Type[PDPProjectConfig | ESProjectConfig]:
    s = _normalize_schema_type(schema_type)
    if s == "pdp":
        return PDPProjectConfig
    if s in ("edvise", "es"):
        return ESProjectConfig
    raise ValueError(
        f"Unknown --schema_type {schema_type!r}; expected 'pdp', 'edvise', or 'es'."
    )


def _log_cohort_term_breakdown(df_ckpt: pd.DataFrame) -> None:
    """Log cohort distribution using PDP or Edvise column names when present."""
    if {"cohort", "cohort_term"}.issubset(df_ckpt.columns):
        cols = ["cohort", "cohort_term"]
    elif {"entry_year", "entry_term"}.issubset(df_ckpt.columns):
        cols = ["entry_year", "entry_term"]
    else:
        logging.info(
            "Skipping cohort-term breakdown log "
            "(expected cohort/cohort_term or entry_year/entry_term on checkpoint frame)."
        )
        return
    counts = df_ckpt[cols].value_counts(dropna=False).sort_index()
    logging.info("Checkpoint cohort-term breakdown:\n%s", counts.to_string())


class CheckpointsTask:
    """Compute one row per student for the configured checkpoint."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        cfg_cls = _project_config_class(args.schema_type)
        self.cfg = read_config(self.args.config_file_path, schema=cfg_cls)

    def checkpoint_generation(self, df_student_terms: pd.DataFrame) -> pd.DataFrame:
        preprocessing_cfg = self.cfg.preprocessing
        if preprocessing_cfg is None or preprocessing_cfg.checkpoint is None:
            raise ValueError("cfg.preprocessing.checkpoint must be configured.")

        cp = preprocessing_cfg.checkpoint
        student_id_col: str = self.cfg.student_id_col

        sort_cols = cp.sort_cols
        include_cols = cp.include_cols

        sort_cols_list = [sort_cols] if isinstance(sort_cols, str) else list(sort_cols)
        missing = [c for c in sort_cols_list if c not in df_student_terms.columns]
        require(
            not missing, f"Checkpoint sort_cols not found in student_terms: {missing}"
        )

        if include_cols:
            missing_inc = [c for c in include_cols if c not in df_student_terms.columns]
            require(
                not missing_inc,
                f"Checkpoint include_cols not found in student_terms: {missing_inc}",
            )

        # Dispatch by type_ so ES and PDP config classes both work (duplicate class defs).
        if cp.type_ == "nth":
            return checkpoints.nth_student_terms.nth_student_terms(
                df_student_terms,
                n=cp.n,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
                enrollment_year_col=cp.enrollment_year_col,
                valid_enrollment_year=cp.valid_enrollment_year,
            )

        if cp.type_ == "first":
            return checkpoints.nth_student_terms.first_student_terms(
                df_student_terms,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        if cp.type_ == "first_at_num_credits_earned":
            return checkpoints.nth_student_terms.first_student_terms_at_num_credits_earned(
                df_student_terms,
                min_num_credits=cp.min_num_credits,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
                num_credits_col=cp.num_credits_col,
            )

        if cp.type_ == "first_within_cohort":
            return checkpoints.nth_student_terms.first_student_terms_within_cohort(
                df_student_terms,
                term_is_pre_cohort_col=cp.term_is_pre_cohort_col,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        raise ValueError(f"Unknown checkpoint type: {cp.type_!r}")

    def run(self) -> None:
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        student_terms_path = os.path.join(current_run_path, "student_terms.parquet")
        student_terms_path_local = local_fs_path(student_terms_path)
        if not os.path.exists(student_terms_path_local):
            raise FileNotFoundError(
                f"student_terms.parquet not found at: {student_terms_path} (local: {student_terms_path_local})"
            )
        df_student_terms = pd.read_parquet(student_terms_path_local)

        student_id_col = self.cfg.student_id_col
        require(
            student_id_col in df_student_terms.columns,
            f"student_terms missing {student_id_col}",
        )
        require(
            df_student_terms[student_id_col].isna().sum() == 0,
            f"student_terms has null {student_id_col}",
        )

        df_ckpt = self.checkpoint_generation(df_student_terms)

        require(len(df_ckpt) > 0, "Checkpoint generation produced 0 rows.")

        dup = df_ckpt.duplicated(subset=[student_id_col]).sum()
        require(
            dup == 0, f"Checkpoint output has {dup} duplicate {student_id_col} rows."
        )

        _log_cohort_term_breakdown(df_ckpt)

        out_ckpt_path = os.path.join(current_run_path, "checkpoint.parquet")
        df_ckpt.to_parquet(local_fs_path(out_ckpt_path), index=False)
        logging.info("Checkpoint file saved to %s", out_ckpt_path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Checkpoint generation from student_terms (PDP or Edvise ES)."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument(
        "--schema_type",
        type=str,
        default="pdp",
        help="pdp | edvise | es — selects PDPProjectConfig vs ESProjectConfig.",
    )
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    task = CheckpointsTask(args)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="checkpoints.log",
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
