"""
ES data audit entrypoint. Uses the generalized DataAuditTask with an ES backend.

Requires: ESProjectConfig, RawESCohortDataSchema, RawESCourseDataSchema,
ESCohortStandardizer, ESCourseStandardizer, read_raw_es_cohort_data, read_raw_es_course_data.
"""

import argparse
import importlib
import logging
import sys

from edvise.data_audit.data_audit import DataAuditBackend, DataAuditTask

# ES-specific imports: add these modules/attributes when you implement ES support
from edvise.data_audit.schemas import RawESCohortDataSchema, RawESCourseDataSchema
from edvise.data_audit.standardizer import ESCohortStandardizer, ESCourseStandardizer
from edvise.dataio.read import read_raw_es_cohort_data, read_raw_es_course_data
from edvise.configs.es import ESProjectConfig, InferenceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def _es_backend() -> DataAuditBackend:
    return DataAuditBackend(
        config_schema=ESProjectConfig,
        cohort_standardizer=ESCohortStandardizer(),
        course_standardizer=ESCourseStandardizer(),
        read_raw_cohort_data=read_raw_es_cohort_data,
        read_raw_course_data=read_raw_es_course_data,
        raw_cohort_schema=RawESCohortDataSchema,
        raw_course_schema=RawESCourseDataSchema,
        log_basename="es_data_audit.log",
        default_course_converter=None,  # set if ES has a default duplicate handler
        inference_config_factory=lambda cohort: InferenceConfig(cohort=cohort),
    )


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
        _es_backend(),
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
