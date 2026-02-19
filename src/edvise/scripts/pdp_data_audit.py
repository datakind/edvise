import argparse
import importlib
import json
import logging
import sys

from edvise.data_audit.data_audit import DataAuditBackend, DataAuditTask
from edvise.data_audit.schemas import RawPDPCohortDataSchema, RawPDPCourseDataSchema
from edvise.data_audit.standardizer import (
    PDPCohortStandardizer,
    PDPCourseStandardizer,
)
from edvise.utils.data_cleaning import handling_duplicates
from edvise.dataio.read import (
    read_raw_pdp_cohort_data,
    read_raw_pdp_course_data,
)
from edvise.configs.pdp import PDPProjectConfig
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def _pdp_backend() -> DataAuditBackend:
    return DataAuditBackend(
        config_schema=PDPProjectConfig,
        cohort_standardizer=PDPCohortStandardizer(),
        course_standardizer=PDPCourseStandardizer(),
        read_raw_cohort_data=read_raw_pdp_cohort_data,
        read_raw_course_data=read_raw_pdp_course_data,
        raw_cohort_schema=RawPDPCohortDataSchema,
        raw_course_schema=RawPDPCourseDataSchema,
        log_basename="pdp_data_audit.log",
        default_course_converter=partial(handling_duplicates, school_type="pdp"),
    )


def _parse_term_filter_param(value: str | None) -> list[str] | None:
    """
    Parse the --term_filter CLI parameter into a list of term labels or None.

    Treats None, empty string, whitespace, 'null', 'None', and [] as not provided (returns None).
    Valid JSON list of strings is returned (trimmed, empty elements dropped).
    Raises ValueError for invalid JSON or non-list JSON.
    """
    if value is None:
        return None
    s = value.strip()
    if not s or s.lower() in ("null", "none"):
        return None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON for --term_filter: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError(
            '--term_filter must be a JSON list of term labels (e.g. ["fall 2024-25"])'
        )
    result = [item.strip() for item in parsed if item is not None and str(item).strip()]
    return result if result else None


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
        LOGGER.warning("Failed to load custom converter functions: %s", e)
    try:
        converter_func = importlib.import_module("dataio")
        course_converter_func = converter_func.converter_func_course
        LOGGER.info("Running task with custom course converter func")
    except Exception as e:
        course_converter_func = None
        LOGGER.info("Running task default course converter func")
        LOGGER.warning("Failed to load custom converter functions: %s", e)

    task = DataAuditTask(
        args,
        _pdp_backend(),
        course_converter_func=course_converter_func,
        cohort_converter_func=cohort_converter_func,
    )
    task.run()
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
