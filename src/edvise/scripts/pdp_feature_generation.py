import argparse
import logging
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

from edvise.feature_generation.column_names import (
    PDP_COHORT_INPUT_COLUMNS,
    PDP_COURSE_INPUT_COLUMNS,
)
from edvise.feature_generation.assemble_student_terms import (
    make_student_term_dataset,
    read_standardized_silver_pair,
    student_level_merge_keys,
    warn_cohort_course_key_overlap,
)
from edvise.dataio.read import read_config
from edvise.configs.pdp import PDPProjectConfig
from edvise.shared.logger import (
    resolve_run_path,
    local_fs_path,
    init_file_logging,
)
from edvise.shared.validation import (
    require,
    require_cols,
    require_no_nulls,
)

# Configure logging
LOGGER = logging.getLogger(__name__)


class PDPFeatureGenerationTask:
    """Encapsulates the  feature generation logic for the SST pipeline."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = read_config(self.args.config_file_path, schema=PDPProjectConfig)

    def run(self):
        """Executes the data feature generation."""

        # --- Unpack config ---
        features_cfg = self.cfg.preprocessing.features
        min_passing_grade = features_cfg.min_passing_grade
        min_num_credits_full_time = features_cfg.min_num_credits_full_time
        course_level_pattern = features_cfg.course_level_pattern
        core_terms = features_cfg.core_terms
        key_course_subject_areas = features_cfg.key_course_subject_areas
        key_course_ids = features_cfg.key_course_ids

        # Ensure correct folder: training or inference
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        # Use local path for reading/writing so DBFS is handled correctly
        current_run_path_local = local_fs_path(current_run_path)
        os.makedirs(current_run_path_local, exist_ok=True)

        # --- Load datasets ---
        df_course, df_cohort = read_standardized_silver_pair(current_run_path)

        # --- High-level input sanity ---

        # Must-have columns for this pipeline
        require_cols(
            df_course,
            [
                "institution_id",
                "student_id",
                "academic_year",
                "academic_term",
            ],
            "Course standardized",
        )
        require_cols(df_cohort, ["institution_id", "student_id"], "Cohort standardized")

        # ID fields must not be null
        merge_on = student_level_merge_keys(
            df_cohort,
            df_course,
            cohort_cols=PDP_COHORT_INPUT_COLUMNS,
        )
        require_no_nulls(
            df_course,
            merge_on,
            "Course standardized",
        )
        require_no_nulls(
            df_cohort,
            merge_on,
            "Cohort standardized",
        )
        warn_cohort_course_key_overlap(df_cohort, df_course, merge_on)

        # --- Generate student-term dataset ---
        df_student_terms = make_student_term_dataset(
            df_cohort=df_cohort,
            df_course=df_course,
            merge_on=merge_on,
            min_passing_grade=min_passing_grade,
            min_num_credits_full_time=min_num_credits_full_time,
            course_level_pattern=course_level_pattern,
            core_terms=core_terms,
            key_course_subject_areas=key_course_subject_areas,
            key_course_ids=key_course_ids,
        )

        # --- Check that generated student-term dataset isn't empty ---
        require(len(df_student_terms) > 0, "student_terms.parquet is empty.")

        # --- Write result ---
        out_path = os.path.join(current_run_path, "student_terms.parquet")
        df_student_terms.to_parquet(local_fs_path(out_path), index=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Feature generation in the Edvise pipeline."
    )
    parser.add_argument("--silver_volume_path", type=str, required=True)
    parser.add_argument("--config_file_path", type=str, required=True)
    parser.add_argument("--job_type", type=str, required=True)
    parser.add_argument("--db_run_id", type=str, required=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    # no school use a custom schema for now remove and add back in iff needed
    # try:
    #     sys.path.append(args.custom_schemas_path)
    #     sys.path.append(
    #         f"/Volumes/staging_sst_01/{args.databricks_institution_name}_bronze/bronze_volume/inference_inputs"
    #     )
    #     schemas = importlib.import_module("schemas")
    #     logging.info("Running task with custom schema")
    # except Exception:
    #     logging.info("Running task with default schema")

    task = PDPFeatureGenerationTask(args)
    log_path = init_file_logging(
        args,
        task.cfg,
        logger_name=__name__,
        log_file_name="pdp_feature_generation.log",
    )
    logging.info("Logs will be written to %s", log_path)
    task.run()

    # Ensure all logs hit disk
    for h in logging.getLogger().handlers:
        try:
            h.flush()
        except Exception:
            pass
    logging.shutdown()
