"""
ES (Edvise Schema) feature generation entrypoint. Uses FeatureGenerationTask with ES config.
"""

import argparse
import logging
import os
import sys

script_dir = os.getcwd()
repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
src_path = os.path.join(repo_root, "src")
if os.path.isdir(src_path) and src_path not in sys.path:
    sys.path.insert(0, src_path)

from edvise.configs.es import ESProjectConfig
from edvise.feature_generation.cohort_columns import ES_STUDENT_COHORT_COLUMNS
from edvise.feature_generation.course_columns import ES_COURSE_STANDARDIZED_COLUMNS
from edvise.feature_generation.pipeline_columns import ES_FEATURE_PIPELINE_COLUMNS
from edvise.feature_generation.task import FeatureGenerationBackend, FeatureGenerationTask
from edvise.shared.logger import init_file_logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)


def _es_backend() -> FeatureGenerationBackend:
    return FeatureGenerationBackend(
        config_schema=ESProjectConfig,
        log_file_name="es_feature_generation.log",
        student_cohort_columns=ES_STUDENT_COHORT_COLUMNS,
        course_standardized_columns=ES_COURSE_STANDARDIZED_COLUMNS,
        feature_pipeline_columns=ES_FEATURE_PIPELINE_COLUMNS,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature generation in the Edvise pipeline (ES)."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    backend = _es_backend()
    task = FeatureGenerationTask(args, backend)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name=backend.log_file_name,
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()

    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
