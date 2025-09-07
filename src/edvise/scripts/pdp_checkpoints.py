import logging
import argparse
import pandas as pd
import sys
import importlib

from edvise import checkpoints
from edvise.configs.pdp import PDPProjectConfig
from edvise.dataio.read import read_config

from edvise.configs.pdp import (
    CheckpointNthConfig,
    CheckpointFirstConfig,
    CheckpointLastConfig,
    CheckpointFirstAtNumCreditsEarnedConfig,
    CheckpointFirstWithinCohortConfig,
    CheckpointLastInEnrollmentYearConfig,
)

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
        self.cfg: PDPProjectConfig = read_config(
            self.args.config_file_path, schema=PDPProjectConfig
        )

    def checkpoint_generation(self, df_student_terms: pd.DataFrame) -> pd.DataFrame:
        """
        Inputs: df_student_terms
        Outputs: df_student_terms parquet file
        """
        preprocessing_cfg = self.cfg.preprocessing
        if preprocessing_cfg is None or preprocessing_cfg.checkpoint is None:
            raise ValueError("cfg.preprocessing.checkpoint must be configured.")

        cp = preprocessing_cfg.checkpoint
        student_id_col: str = self.cfg.student_id_col

        # sort_cols: str | list[str] in schema; most functions accept list[str] or str.
        sort_cols = cp.sort_cols
        include_cols = cp.include_cols

        # Prefer isinstance() to narrow the union:
        if isinstance(cp, CheckpointNthConfig):
            return checkpoints.nth_student_terms.nth_student_terms(
                df_student_terms,
                n=cp.n,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
                enrollment_year_col=cp.enrollment_year_col,
                valid_enrollment_year=cp.valid_enrollment_year,
            )

        if isinstance(cp, CheckpointFirstConfig):
            return checkpoints.nth_student_terms.first_student_terms(
                df_student_terms,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        if isinstance(cp, CheckpointLastConfig):
            return checkpoints.nth_student_terms.last_student_terms(
                df_student_terms,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        if isinstance(cp, CheckpointFirstAtNumCreditsEarnedConfig):
            return (
                checkpoints.nth_student_terms.first_student_terms_at_num_credits_earned(
                    df_student_terms,
                    min_num_credits=cp.min_num_credits,
                    sort_cols=sort_cols,
                    include_cols=include_cols,
                    student_id_cols=student_id_col,
                )
            )

        if isinstance(cp, CheckpointFirstWithinCohortConfig):
            # Schema guarantees str default, so this is already str
            return checkpoints.nth_student_terms.first_student_terms_within_cohort(
                df_student_terms,
                term_is_pre_cohort_col=cp.term_is_pre_cohort_col,
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        if isinstance(cp, CheckpointLastInEnrollmentYearConfig):
            # enrollment_year is float in schema; function expects int → coerce
            return checkpoints.nth_student_terms.last_student_terms_in_enrollment_year(
                df_student_terms,
                enrollment_year=int(cp.enrollment_year),
                enrollment_year_col=cp.enrollment_year_col,  # str in schema
                sort_cols=sort_cols,
                include_cols=include_cols,
                student_id_cols=student_id_col,
            )

        raise ValueError(f"Unknown checkpoint type: {cp.type_}")

    def run(self):
        """Executes the data preprocessing pipeline."""
        df_student_terms = pd.read_parquet(
            f"{self.args.silver_volume_path}/student_terms.parquet"
        )

        df_ckpt = self.checkpoint_generation(df_student_terms)

        df_ckpt.to_parquet(
            f"{self.args.silver_volume_path}/checkpoint.parquet", index=False
        )



def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)


    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    #again no school has a custom schema for pdp , but add in iff needed
    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     sys.path.append(
    #         f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
    #     )
    #     schemas = importlib.import_module("schemas")
    #     logging.info("Running task with custom schema")
    # except Exception:
    #     logging.info("Running task with default schema")
    task = PDPCheckpointsTask(args)
    task.run()
