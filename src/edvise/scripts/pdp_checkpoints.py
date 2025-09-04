import logging
import argparse
import pandas as pd
import sys
import importlib

from .. import checkpoints
from edvise.utils.databricks import read_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)  # Ignore Databricks logger


class PDPCheckpointsTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        """
        Initializes the DataProcessingTask.

        Args:
            args: The parsed command-line arguments.
        """
        self.args = args
        self.cfg = read_config(self.args.toml_file_path)

    def checkpoint_generation(self, df_student_terms: pd.DataFrame) -> pd.DataFrame:
        """
        Inputs: df_student_terms
        Outputs: df_student_terms parquet file
        """
        # Read preprocessing features from config (could move to own fn but maybe later)
        checkpoint_type = self.cfg.preprocessing.checkpoint.type_
        student_id_col = self.cfg.student_id_col
        n = self.cfg.preprocessing.checkpoint.n
        sort_cols = self.cfg.preprocessing.checkpoint.sort_cols
        include_cols = self.cfg.preprocessing.checkpoint.include_cols
        enrollment_year = self.cfg.preprocessing.checkpoint.enrollment_year
        enrollment_year_col = self.cfg.preprocessing.checkpoint.enrollment_year_col
        min_num_credits = self.cfg.preprocessing.checkpoint.min_num_credits
        term_is_pre_cohort_col = (
            self.cfg.preprocessing.checkpoint.term_is_pre_cohort_col
        )
        valid_enrollment_year = self.cfg.preprocessing.checkpoint.valid_enrollment_year

        # Set up dictionary of checkpoint functions
        checkpoint_functions = {
            "nth": lambda: checkpoints.nth_student_terms.nth_student_terms(
                df_student_terms,
                n=n,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
                enrollment_year_col=enrollment_year_col,
                valid_enrollment_year=valid_enrollment_year,
            ),
            "first": lambda: checkpoints.nth_student_terms.first_student_terms(
                df_student_terms,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            ),
            "last": lambda: checkpoints.nth_student_terms.last_student_terms_in_enrollment_year(
                df_student_terms,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            ),
            "first_at_num_credits_earned": lambda: checkpoints.nth_student_terms.first_student_terms_at_num_credits_earned(
                df_student_terms,
                min_num_credits=min_num_credits,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            ),
            "first_within_cohort": lambda: checkpoints.nth_student_terms.first_student_terms_within_cohort(
                df_student_terms,
                term_is_pre_cohort_col=term_is_pre_cohort_col,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            ),
            "last_in_enrollment_year": lambda: checkpoints.nth_student_terms.last_student_terms_in_enrollment_year(
                df_student_terms,
                enrollment_year=enrollment_year,
                enrollment_year_col=enrollment_year_col,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            ),
        }

        if checkpoint_type not in checkpoint_functions:
            raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")
        logging.info(f"Checkpoint type: {checkpoint_type}")

        df_ckpt = checkpoint_functions[checkpoint_type]()

        return df_ckpt

    def run(self):
        """Executes the data preprocessing pipeline."""
        df_student_terms = pd.read_parquet(
            f"{self.args.student_term_path}/student_terms.parquet"
        )
        df_ckpt = self.checkpoint_generation(df_student_terms)
        df_ckpt.to_parquet(
            f"{self.args.checkpoint_path}/checkpoint.parquet", index=False
        )


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument(
        "--toml_file_path", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--custom_schemas_path",
        required=False,
        help="Folder path to store custom schemas folders",
    )
    parser.add_argument(
        "--student_term_path",
        required=False,
        help="Folder path to store student term file",
    )
    parser.add_argument(
        "--checkpoint_path",
        required=False,
        help="Folder path to store checkpoint file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    try:
        sys.path.append(args.custom_schemas_path)
        sys.path.append(
            f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
        )
        schemas = importlib.import_module("schemas")
        logging.info("Running task with custom schema")
    except Exception:
        from dataio.schemas import pdp as schemas

        logging.info("Running task with default schema")
    task = PDPCheckpointsTask(args)
    task.run()
