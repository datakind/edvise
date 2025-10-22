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
from edvise.shared.logger import resolve_run_path, local_fs_path


# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)


class StudentSelectionTask:
    """Handles selection of students based on specified attribute criteria."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def run(self):
        """Execute the student selection task."""
        # Ensure correct folder: training or inference
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        # Load the student-term data
        st_terms_path = os.path.join(current_run_path, "student_terms.parquet")
        st_terms_path_local = local_fs_path(st_terms_path)
        if not os.path.exists(st_terms_path_local):
            raise FileNotFoundError(
                f"Missing student_terms.parquet at: {st_terms_path} (local: {st_terms_path_local})"
            )
        df_student_terms = pd.read_parquet(st_terms_path_local)

        # Pull selection criteria and ID column(s)
        student_criteria = self.cfg.preprocessing.selection.student_criteria
        student_id_col = self.cfg.student_id_col

        # Select students
        df_selected_students = (
            student_selection.select_students_attributes.select_students_by_attributes(
                df_student_terms,
                student_id_cols=student_id_col,
                **student_criteria,
            )
        )

        # Save to parquet
        out_path = os.path.join(current_run_path, "selected_students.parquet")
        df_selected_students.to_parquet(local_fs_path(out_path), index=True)
        logging.info(f"Saved {len(df_selected_students)} selected students.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Student selection based on configured attributes."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     schemas = importlib.import_module("schemas")
    #     logging.info("Using custom schema")
    # except Exception:
    #     logging.info("Using default schema")

    task = StudentSelectionTask(args)
    task.run()
