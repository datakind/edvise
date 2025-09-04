import logging
import argparse
import pandas as pd
import sys
import importlib

from .. import targets as _targets  # assumes targets/__init__.py imports the modules
from edvise.dataio.read import read_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class PDPTargetsTask:
    """Computes the target variable for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.toml_file_path)

    def target_generation(self, df_student_terms: pd.DataFrame) -> pd.Series:
        """
        Computes the target variable based on the method specified in the config.
        Returns a Series indexed by student ID(s) with boolean values.
        """
        target_type = self.cfg.preprocessing.target.type_

        # Map target type from config to appropriate module
        target_modules = {
            "credits_earned": _targets.credits_earned,
            "graduation": _targets.graduation,
            "retention": _targets.retention,
        }

        if target_type not in target_modules:
            raise ValueError(f"Unknown target type: {target_type}")

        logging.info(f"Computing target using method: {target_type}")
        compute_func = target_modules[target_type].compute_target

        # Call compute_target with the student-term dataframe and kwargs from config
        target_series = compute_func(df_student_terms, **self.cfg.preprocessing.target)

        if not isinstance(target_series, pd.Series):
            raise TypeError(
                f"Expected pd.Series from compute_target, got {type(target_series)}"
            )

        return target_series

    def run(self):
        """Executes the target computation pipeline and saves result."""
        logging.info("Loading student-terms data...")
        df_student_terms = pd.read_parquet(
            f"{self.args.student_term_path}/student_terms.parquet"
        )

        logging.info("Generating target labels...")
        target_series = self.target_generation(df_student_terms)

        logging.info("Saving target data...")
        # Convert Series to DataFrame for saving
        df_target = target_series.reset_index().rename(
            columns={target_series.name: "target"}
        )
        df_target.to_parquet(f"{self.args.target_path}/target.parquet", index=False)
        logging.info(f"Target file saved to {self.args.target_path}/target.parquet")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Target generation for SST pipeline.")
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to config file"
    )
    parser.add_argument(
        "--custom_schemas_path", required=False, help="Path to custom schemas"
    )
    parser.add_argument(
        "--student_term_path",
        type=str,
        required=True,
        help="Path to student term parquet",
    )
    parser.add_argument(
        "--target_path", type=str, required=True, help="Path to output target parquet"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        if args.custom_schemas_path:
            sys.path.append(args.custom_schemas_path)
            schemas = importlib.import_module("schemas")
            logging.info("Using custom schemas")
    except Exception:
        logging.info("Using default schemas")

    task = PDPTargetsTask(args)
    task.run()
