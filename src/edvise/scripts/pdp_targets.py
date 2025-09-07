import logging
import argparse
import pandas as pd
import sys
import importlib

from src.edvise import targets as _targets  # assumes targets/__init__.py imports the modules
from src.edvise.dataio.read import read_config
from src.edvise.configs.pdp import PDPProjectConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class PDPTargetsTask:
    """Computes the target variable for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def target_generation(self, df_student_terms: pd.DataFrame) -> pd.Series:
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

        s = compute_func(df_student_terms, **kwargs)
        if not isinstance(s, pd.Series):
            raise TypeError(f"compute_target must return pd.Series, got {type(s)}")
        return s.astype(bool) if s.dtype != "bool" else s

    def run(self):
        """Executes the target computation pipeline and saves result."""
        logging.info("Loading student-terms data...")
        df_student_terms = pd.read_parquet(
            f"{self.args.silver_volume_path}/student_terms.parquet"
        )

        logging.info("Generating target labels...")
        target_series = self.target_generation(df_student_terms)

        logging.info("Saving target data...")
        # Convert Series to DataFrame for saving
        df_target = target_series.reset_index().rename(
            columns={target_series.name: "target"}
        )
        df_target.to_parquet(f"{self.args.silver_volume_path}/target.parquet", index=False)
        logging.info(f"Target file saved to {self.args.silver_volume_path}/target.parquet")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Target generation for SST pipeline.")
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # try:
    #     if args.custom_schemas_path:
    #         sys.path.append(args.custom_schemas_path)
    #         schemas = importlib.import_module("schemas")
    #         logging.info("Using custom schemas")
    # except Exception:
    #     logging.info("Using default schemas")

    task = PDPTargetsTask(args)
    task.run()
