import argparse
import importlib
import logging
import typing as t
import sys
import pandas as pd
import pathlib
import os

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

from edvise.data_audit.schemas import RawPDPCohortDataSchema, RawPDPCourseDataSchema
from edvise.data_audit.standardizer import (
    PDPCohortStandardizer,
    PDPCourseStandardizer,
)
from edvise.utils.databricks import get_spark_session
from edvise.utils.data_cleaning import handling_duplicates

from edvise.dataio.read import (
    read_config,
    read_raw_pdp_cohort_data,
    read_raw_pdp_course_data,
)
from edvise.dataio.write import write_parquet
from edvise.configs.pdp import PDPProjectConfig
from edvise.data_audit.eda import (
    compute_gateway_course_ids_and_cips,
    log_record_drops,
    log_terms,
    log_misjoined_records,
)
from edvise.data_audit.cohort_selection import select_inference_cohort
from edvise.utils.update_config import update_key_courses_and_cips
from edvise.utils.data_cleaning import (
    remove_pre_cohort_courses,
    log_pre_cohort_courses,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Create callable type
ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class PDPDataAuditTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(
        self,
        args: argparse.Namespace,
        course_converter_func: t.Optional[ConverterFunc] = None,
        cohort_converter_func: t.Optional[ConverterFunc] = None,
    ):
        self.args = args
        self.cfg = read_config(
            file_path=self.args.config_file_path, schema=PDPProjectConfig
        )
        self.spark = get_spark_session()
        self.cohort_std = PDPCohortStandardizer()
        self.course_std = PDPCourseStandardizer()
        # Use default converter to handle duplicates if none provided
        self.course_converter_func: ConverterFunc = (
            handling_duplicates
            if course_converter_func is None
            else course_converter_func
        )
        self.cohort_converter_func: t.Optional[ConverterFunc] = cohort_converter_func

    def in_databricks(self) -> bool:
        return bool(
            os.getenv("DATABRICKS_RUNTIME_VERSION") or os.getenv("DB_IS_DRIVER")
        )

    def _path_exists(self, p: str) -> bool:
        if not p:
            return False
        # DBFS scheme
        if p.startswith("dbfs:/"):
            try:
                from databricks.sdk.runtime import dbutils  # lazy import

                dbutils.fs.ls(p)  # will raise if not found
                return True
            except Exception:
                return False
        # Local Posix path (Vols are mounted here)
        return pathlib.Path(p).exists()

    def _pick_existing_path(
        self,
        prefer_path: t.Optional[str],
        fallback_path: str,
        label: str,
        use_fallback_on_dbx: bool = True,
    ) -> str:
        """
        prefer_path: inference-provided path (may be None or empty)
        fallback_path: config path
        label: 'course' or 'cohort'
        use_fallback_on_dbx: only fallback to config automatically when on Databricks
        """
        prefer = (prefer_path or "").strip()
        if prefer and self._path_exists(prefer):
            LOGGER.info("%s: using inference-provided path: %s", label, prefer)
            return prefer

        if prefer and not self._path_exists(prefer):
            LOGGER.warning("%s: inference-provided path not found: %s", label, prefer)

        if (
            use_fallback_on_dbx
            and self.in_databricks()
            and self._path_exists(fallback_path)
        ):
            LOGGER.info(
                "%s: utilizing training-config path on Databricks: %s",
                label,
                fallback_path,
            )
            return fallback_path

        # If we get here, we couldn't find a usable path
        tried = [p for p in [prefer, fallback_path] if p]
        raise FileNotFoundError(
            f"{label}: none of the candidate paths exist. Tried: {tried}. "
            f"Environment: {'Databricks' if self.in_databricks() else 'non-Databricks'}"
        )

    def run(self):
        """Executes the data preprocessing pipeline."""
        # Create a folder to save all the files in
        if self.args.job_type == "training":
            current_run_path = f"{self.args.silver_volume_path}/{self.args.db_run_id}"
            os.makedirs(current_run_path, exist_ok=True)
        elif self.args.job_type == "inference":
            if self.cfg.model.run_id is None:
                raise ValueError("cfg.model.run_id must be set for inference runs.")
            current_run_path = f"{self.args.silver_volume_path}/{self.cfg.model.run_id}"
        else:
            raise ValueError(f"Unsupported job_type: {self.args.job_type}")

        # Determine file paths
        cohort_dataset_raw_path = self._pick_existing_path(
            self.args.cohort_dataset_validated_path,
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_cohort}",
            "cohort",
        )
        course_dataset_raw_path = self._pick_existing_path(
            self.args.course_dataset_validated_path,
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_course}",
            "course",
        )

        # --- Load RAW datasets ---
        LOGGER.info(" Loading raw cohort and course datasets:")

        # Raw cohort data
        df_cohort_raw = read_raw_pdp_cohort_data(
            file_path=cohort_dataset_raw_path,
            schema=None,
            spark_session=self.spark,
        )

        # Raw course data
        dttm_formats = ["ISO8601", "%Y%m%d.0"]
        for fmt in dttm_formats:
            try:
                df_course_raw = read_raw_pdp_course_data(
                    file_path=course_dataset_raw_path,
                    schema=None,
                    dttm_format=fmt,
                    spark_session=self.spark,
                )
                break  # success — exit loop
            except ValueError:
                continue  # try next format
        else:
            raise ValueError(
                " Failed to parse course data with all known datetime formats."
            )

        LOGGER.info(
            " Loaded raw cohort and course data: checking for mismatches in cohort and course files: "
        )
        log_misjoined_records(df_cohort_raw, df_course_raw)

        # Logs cohort year and terms and academic year and terms, grouped and sorted

        LOGGER.info(
            " Listing grouped cohort year and terms and academic year and terms for raw cohort and course data files: "
        )
        log_terms(
            df_course_raw,
            df_cohort_raw,
        )

        # TODO: we may want to add checks here for expected columns, rows, etc. that could break the schemas

        # --- Load COHORT dataset - with schema ---

        # Schema validate cohort data
        LOGGER.info(" Reading and schema validating cohort data:")
        df_cohort_validated = read_raw_pdp_cohort_data(
            file_path=cohort_dataset_raw_path,
            schema=RawPDPCohortDataSchema,
            converter_func=self.cohort_converter_func,
            spark_session=self.spark,
        )

        # Select inference cohort if applicable
        if self.args.job_type == "inference":
            LOGGER.info(" Selecting inference cohort")
            if self.cfg.inference is None or self.cfg.inference.cohort is None:
                raise ValueError("cfg.inference.cohort must be configured.")

            inf_cohort = self.cfg.inference.cohort
            df_cohort_validated = select_inference_cohort(
                df_cohort_validated, cohorts_list=inf_cohort
            )

        # Standardize cohort data
        LOGGER.info(" Standardizing cohort data:")
        df_cohort_standardized = self.cohort_std.standardize(df_cohort_validated)

        LOGGER.info(" Cohort data standardized.")

        # --- Load COURSE dataset - with schema ---

        # Schema validate course data and handle duplicates
        LOGGER.info(
            " Reading and schema validating course data, handling any duplicates:"
        )

        for fmt in dttm_formats:
            try:
                df_course_validated = read_raw_pdp_course_data(
                    file_path=course_dataset_raw_path,
                    schema=RawPDPCourseDataSchema,
                    dttm_format=fmt,
                    converter_func=self.course_converter_func,
                    spark_session=self.spark,
                )
                break  # success — exit loop
            except ValueError:
                continue  # try next format
        else:
            raise ValueError(
                " Failed to parse course data with all known datetime formats."
            )
        LOGGER.info(" Course data read and schema validated, duplicates handled.")

        # TODO: can't tell if this is working?
        try:
            include_pre_cohort = self.cfg.preprocessing.include_pre_cohort_courses
        except AttributeError:
            raise AttributeError(
                "Config error: 'include_pre_cohort_courses' is missing. "
                "Please set it explicitly in the config file under 'preprocessing' based on your school's preference (for default models, this should always be false)."
            )

        if not include_pre_cohort:
            df_course_validated = remove_pre_cohort_courses(
                df_course_validated, self.cfg.student_id_col
            )
        else:
            log_pre_cohort_courses(df_course_validated, self.cfg.student_id_col)

        # Select inference cohort if applicable
        if self.args.job_type == "inference":
            LOGGER.info(" Selecting inference cohort")
            if self.cfg.inference is None or self.cfg.inference.cohort is None:
                raise ValueError("cfg.inference.cohort must be configured.")

            inf_cohort = self.cfg.inference.cohort
            df_course_validated = select_inference_cohort(
                df_course_validated, cohorts_list=inf_cohort
            )

        # Standardize course data
        LOGGER.info(" Standardizing course data:")
        df_course_standardized = self.course_std.standardize(df_course_validated)

        LOGGER.info(" Course data standardized.")

        # Log Math/English gateway courses and add to config
        ids_cips = compute_gateway_course_ids_and_cips(df_course_standardized)

        # Auto-populate only at training time to avoid training-inference skew
        if self.args.job_type == "training":
            LOGGER.info(
                " Auto-populating config with below course IDs and cip codes: change if necessary"
            )
            update_key_courses_and_cips(
                self.args.config_file_path,
                key_course_ids=ids_cips[0],
                key_course_subject_areas=ids_cips[1],
            )

        # Log changes before and after pre-processing
        log_record_drops(
            df_cohort_raw,
            df_cohort_standardized,
            df_course_raw,
            df_course_standardized,
        )

        LOGGER.info(
            " Listing grouped cohort year and terms and academic year and terms for *standardized* cohort and course data files: "
        )

        # Logs cohort year and terms and academic year and terms, grouped and sorted
        log_terms(
            df_course_standardized,
            df_cohort_standardized,
        )

        # --- Write results ---
        write_parquet(
            df_cohort_standardized,
            f"{current_run_path}/df_cohort_standardized.parquet",
        )
        write_parquet(
            df_course_standardized,
            f"{current_run_path}/df_course_standardized.parquet",
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
    # try:
    #     schemas = importlib.import_module("schemas")
    #     LOGGER.info("Running task with custom schema")
    # except Exception as e:
    #     from data_audit import schemas as schemas
    #     LOGGER.info("Running task with default schema")
    #     LOGGER.warning(f"Failed to load custom schema: {e}")

    task = PDPDataAuditTask(
        args,
        cohort_converter_func=cohort_converter_func,
        course_converter_func=course_converter_func,
    )
    task.run()
