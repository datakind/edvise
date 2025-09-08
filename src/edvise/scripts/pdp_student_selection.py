import logging
import argparse
import pandas as pd

from edvise import student_selection
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig


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
        # Load the student-term data
        df_student_terms = pd.read_parquet(
            f"{self.args.silver_volume_path}/student_terms.parquet"
        )

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
        df_selected_students.to_parquet(
            f"{self.args.silver_volume_path}/selected_students.parquet", index=True
        )
        logging.info(f"Saved {len(df_selected_students)} selected students.")


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

    task = StudentSelectionTask(args)
    task.run()
