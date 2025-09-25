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

from edvise import student_selection
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig
from edvise.dataio.write import write_parquet


# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class InfCohortTask:
    """Handles selection of students based on specified attribute criteria."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def select_inference_cohort(
        self, df_course: pd.DataFrame, df_cohort: pd.DataFrame
    )-> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Selects the specified cohorts from the course and cohort DataFrames.

        Args:
            df_course: The course DataFrame.
            df_cohort: The cohort DataFrame.
            cohorts_list: List of cohorts to select (e.g., ["fall 2023", "spring 2024"]).

        Returns:
            A tuple containing the filtered course and cohort DataFrames.
        
        Raises:
            ValueError: If filtering results in empty DataFrames.
        """
        #change to main config when its updated
        cohorts_list = self.inf_cfg["inference_cohort"]

        #We only have cohort and cohort term split up, so combine and strip to lower to prevent cap issues
        df_course['cohort_selection'] = df_course['cohort_term'].astype(str).str.lower() + " " + df_course['cohort'].astype(str).str.lower()
        df_cohort['cohort_selection'] = df_cohort['cohort_term'].astype(str).str.lower() + " " + df_cohort['cohort'].astype(str).str.lower()
        
        #Subset both datsets to only these cohorts
        df_course_filtered = df_course[df_course['cohort_selection'].isin(cohorts_list)]
        df_cohort_filtered = df_cohort[df_cohort['cohort_selection'].isin(cohorts_list)]
        
        #Log confirmation we are selecting the correct cohorts
        logging.info("Selected cohorts: %s", cohorts_list)
        
        #Throw error if either dataset is empty after filtering
        if df_course_filtered.empty or df_cohort_filtered.empty:
            logging.error("Selected cohorts resulted in empty DataFrames.")
            raise ValueError("Selected cohorts resulted in empty DataFrames.")
        
        logging.info("Cohort selection completed. Course shape: %s, Cohort shape: %s", df_course_filtered.shape, df_cohort_filtered.shape)
        
        return df_course_filtered, df_cohort_filtered        


    def run(self):
        """Execute the student selection task."""
        # --- Load datasets ---
        df_course = pd.read_parquet(
            f"{self.args.silver_volume_path}/df_course_standardized.parquet"
        )
        df_cohort = pd.read_parquet(
            f"{self.args.silver_volume_path}/df_cohort_standardized.parquet"
        )

        # Select students
        df_course_standardized, df_cohort_standardized = self.select_inference_cohort(df_course, df_cohort)

        # --- Write results ---
        write_parquet(
            df_cohort_standardized,
            f"{self.args.silver_volume_path}/df_cohort_standardized.parquet",
        )
        write_parquet(
            df_course_standardized,
            f"{self.args.silver_volume_path}/df_course_standardized.parquet",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student selection based on configured attributes."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     schemas = importlib.import_module("schemas")
    #     logging.info("Using custom schema")
    # except Exception:
    #     logging.info("Using default schema")

    task = InfCohortTask(args)
    task.run()
