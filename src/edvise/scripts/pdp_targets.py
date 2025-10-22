import logging
import argparse
import pandas as pd
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

from edvise import targets as _targets
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig
from edvise.shared.logger import local_fs_path, resolve_run_path, init_file_logging


# Configure logging
logging.getLogger("py4j").setLevel(logging.WARNING)


class PDPTargetsTask:
    """Computes the target variable for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def target_generation(
        self, df_student_terms: pd.DataFrame, df_ckpt: pd.DataFrame
    ) -> pd.Series:
        """
        Computes the target variable based on config.
        Returns a Series indexed by student ID(s) with boolean values.
        """
        preproc = self.cfg.preprocessing
        if preproc is None or preproc.target is None:
            raise ValueError("cfg.preprocessing.target must be configured.")

        target_cfg = preproc.target
        target_type = target_cfg.type_

        target_modules = {
            "credits_earned": _targets.credits_earned,
            "graduation": _targets.graduation,
            "retention": _targets.retention,
        }
        if target_type not in target_modules:
            raise ValueError(f"Unknown target type: {target_type}")

        compute_func = target_modules[target_type].compute_target
        kwargs = target_cfg.model_dump()
        kwargs.pop("name", None)
        kwargs.pop("type_", None)
        if target_type == "credits_earned":
            kwargs["checkpoint"] = df_ckpt

        s = compute_func(df_student_terms, **kwargs)
        if not isinstance(s, pd.Series):
            raise TypeError(f"compute_target must return pd.Series, got {type(s)}")
        return s.astype(bool) if s.dtype != "bool" else s

    def run(self):
        """Executes the target computation pipeline and saves result."""
        logging.info("Loading student-terms data...")
        # Resolve <silver>/<run_id>/<training|inference>/
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

        logging.info("Saving target data...")
        # Convert Series to DataFrame for saving
        df_target = target_series.reset_index().rename(
            columns={target_series.name: "target"}
        )

        out_path = os.path.join(current_run_path, "target.parquet")
        df_target.to_parquet(local_fs_path(out_path), index=False)
        logging.info(f"Target file saved to {out_path}")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Target generation for SST pipeline.")
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    parser.add_argument(
        "--job_type", type=str, choices=["training", "inference"], required=False
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # Infer job_type if not provided: if cfg.model.run_id exists ⇒ inference, else training
    try:
        cfg_for_infer = read_config(args.config_file_path, schema=PDPProjectConfig)
        if not getattr(args, "job_type", None):
            inferred = (
                "inference"
                if getattr(getattr(cfg_for_infer, "model", None), "run_id", None)
                else "training"
            )
            logging.info(
                f"No --job_type passed; inferring job_type='{inferred}' from config."
            )
            args.job_type = inferred
    except Exception as e:
        # If config read fails here, fall back to training (or re-raise if you prefer)
        if not getattr(args, "job_type", None):
            logging.warning(
                f"Could not infer job_type from config ({e}); defaulting to 'training'."
            )
            args.job_type = "training"
    # try:
    #     if args.custom_schemas_path:
    #         sys.path.append(args.custom_schemas_path)
    #         schemas = importlib.import_module("schemas")
    #         logging.info("Using custom schemas")
    # except Exception:
    #     logging.info("Using default schemas")

    task = PDPTargetsTask(args)
    # Attach per-run file logging under the resolved run folder
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="pdp_targets.log",  # optional; omit to use default
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()
    # Ensure logs are written to disk
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()

