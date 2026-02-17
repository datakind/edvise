import argparse
import importlib
import logging
import typing as t
import sys
import pandas as pd

from edvise.data_audit import data_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Create callable type
ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Data preprocessing for inference in the SST pipeline."
    )
    parser.add_argument(
        "--course_dataset_validated_path",
        required=False,
        help="Name of the course data file during inference with GCS blobs when connected to webapp",
    )
    parser.add_argument(
        "--cohort_dataset_validated_path",
        required=False,
        help="Name of the cohort data file during inference with GCS blobs when connected to webapp",
    )
    parser.add_argument(
        "--term_filter",
        type=str,
        default=None,
        help='JSON list of term/cohort labels (e.g. ["fall 2024-25"]). Omit or null for config default. Used for cohort and graduation models.',
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--bronze_volume_path", type=str, required=False)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--DB_workspace", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.bronze_volume_path:
        sys.path.append(f"{args.bronze_volume_path}/training_inputs")
    try:
        converter_func = importlib.import_module("dataio")
        cohort_converter_func = converter_func.converter_func_cohort
        LOGGER.info("Running task with custom cohort converter func")
    except Exception as e:
        cohort_converter_func = None
        LOGGER.info("Running task with default cohort converter func")
        LOGGER.warning(f"Failed to load custom converter functions: {e}")
    try:
        converter_func = importlib.import_module("dataio")
        course_converter_func = converter_func.converter_func_course
        LOGGER.info("Running task with custom course converter func")
    except Exception as e:
        course_converter_func = None
        LOGGER.info("Running task default course converter func")
        LOGGER.warning(f"Failed to load custom converter functions: {e}")

    task = data_audit.DataAuditTask(
        args,
        cohort_converter_func=cohort_converter_func,
        course_converter_func=course_converter_func,
    )
    task.run()
    # Ensure all logs are flushed to disk
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
