import argparse
import logging
import typing as t
import sys
import pandas as pd
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

from edvise.data_audit.schemas import RawEdviseStudentDataSchema, RawEdviseCourseDataSchema
from edvise.data_audit.standardizer import (
    ESCohortStandardizer,
    ESCourseStandardizer,
)
from edvise.utils.databricks import get_spark_session
from edvise.dataio.path_management import pick_existing_path
from edvise.dataio.read import (
    read_config,
    read_raw_es_cohort_data,
    read_raw_es_course_data,
)
from edvise.dataio.write import write_parquet
from edvise.configs.es import ESProjectConfig
from edvise.data_audit.eda import (
    log_grade_distribution,
    log_high_null_columns,
    log_record_drops,
    log_terms,
    log_misjoined_records,
)

from edvise.shared.logger import init_file_logging, resolve_run_path
from edvise.shared.validation import require
from edvise.data_audit.data_audit_cli import (
    apply_bronze_training_inputs_sys_path,
    flush_data_audit_logging,
    parse_data_audit_args,
    run_data_audit_with_training_events,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("py4j").setLevel(logging.WARNING)
LOGGER = logging.getLogger(__name__)

# Create callable type
ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class ESDataAuditTask:
    """Encapsulates the data preprocessing logic for the SST pipeline."""

    def __init__(
        self,
        args: argparse.Namespace,
        course_converter_func: t.Optional[ConverterFunc] = None,
        cohort_converter_func: t.Optional[ConverterFunc] = None,
    ):
        self.args = args
        self.cfg = read_config(
            file_path=self.args.config_file_path, schema=ESProjectConfig
        )
        self.spark = get_spark_session()
        self.cohort_std = ESCohortStandardizer()
        self.course_std = ESCourseStandardizer()
        self.course_converter_func: t.Optional[ConverterFunc] = course_converter_func
        self.cohort_converter_func: t.Optional[ConverterFunc] = cohort_converter_func

    def run(self):
        """Executes the data preprocessing pipeline."""
        # Ensure correct folder: training or inference
        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        os.makedirs(current_run_path, exist_ok=True)

        # Root logger: Databricks-safe console + per-run file (replaces all handlers)
        try:
            init_file_logging(
                self.args,
                self.cfg,
                logger_name=__name__,
                log_file_name="es_data_audit.log",
            )
        except Exception as e:
            LOGGER.exception("Failed to initialize file logging: %s", e)

        # Determine file paths
        cohort_dataset_raw_path = pick_existing_path(
            self.args.cohort_dataset_validated_path,
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_cohort}",
            "cohort",
        )
        course_dataset_raw_path = pick_existing_path(
            self.args.course_dataset_validated_path,
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_course}",
            "course",
        )

        # --- Load RAW datasets w/o schema---
        LOGGER.info(" Loading raw cohort and course datasets:")

        dttm_formats = ["ISO8601", "%Y%m%d.0"]

        for fmt in dttm_formats:
            try:
                df_cohort_raw = read_raw_es_cohort_data(
                    file_path=cohort_dataset_raw_path,
                    schema=None,
                    dttm_format=fmt,
                    spark_session=self.spark,
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                " Failed to parse cohort data with all known datetime formats."
            )

        for fmt in dttm_formats:
            try:
                df_course_raw = read_raw_es_course_data(
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

        # Ensure files are non-empty
        for label, df in [
            ("Raw cohort", df_cohort_raw),
            ("Raw course", df_course_raw),
        ]:
            require(len(df) > 0, f"{label} dataset is empty (0 rows).")

        LOGGER.info(
            " Loaded raw cohort and course data: checking for mismatches in cohort and course files: "
        )
        # Edvise student/course columns (not PDP cohort/cohort_term or enrollment_intensity_first_term)
        log_misjoined_records(
            df_cohort_raw,
            df_course_raw,
            merge_key=self.cfg.student_id_col,
            value_count_columns=["enrollment_type"],
            grouped_count_column_groups=[
                ["entry_year", "entry_term"],
                ["academic_year", "academic_term"],
            ],
        )

        # Logs entry and academic year/term pairs, grouped and sorted (Edvise column names)

        LOGGER.info(
            " Listing grouped entry year/term and academic year/term for raw cohort and course data files: "
        )
        log_terms(df_cohort_raw, "entry_year", "entry_term")
        log_terms(df_course_raw, "academic_year", "academic_term")

        # TODO: we may want to add checks here for expected columns, rows, etc. that could break the schemas

        # --- Load COHORT dataset - with schema ---

        # Schema validate cohort data
        LOGGER.info(" Reading and schema validating cohort data:")
        for fmt in dttm_formats:
            try:
                df_cohort_validated = read_raw_es_cohort_data(
                    file_path=cohort_dataset_raw_path,
                    schema=RawEdviseStudentDataSchema,
                    dttm_format=fmt,
                    converter_func=self.cohort_converter_func,
                    spark_session=self.spark,
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                " Failed to parse cohort data with all known datetime formats (validated pass)."
            )

        # Log high null columns
        log_high_null_columns(df_cohort_validated)
        # Standardize cohort data
        LOGGER.info(" Standardizing cohort data:")
        
        df_cohort_standardized = self.cohort_std.standardize(df_cohort_validated)

        LOGGER.info(" Cohort data standardized.")

        student_id_col = getattr(self.cfg, "student_id_col", None) or "learner_id"

        # --- Load COURSE dataset - with schema ---

        # Schema validate course data and handle duplicates
        LOGGER.info(
            " Reading and schema validating course data, handling any duplicates:"
        )

        for fmt in dttm_formats:
            try:
                df_course_validated = read_raw_es_course_data(
                    file_path=course_dataset_raw_path,
                    schema=RawEdviseCourseDataSchema,
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

        # TODO: Need to merge cohort and course to do this as we dont have cohort 
        # in course data for ES
        # try:
        #     include_pre_cohort = self.cfg.preprocessing.include_pre_cohort_courses
        # except AttributeError:
        #     raise AttributeError(
        #         "Config error: 'include_pre_cohort_courses' is missing. "
        #         "Please set it explicitly in the config file under 'preprocessing' based on your school's preference (for default models, this should always be false)."
        #     )

        # if not include_pre_cohort:
        #     df_course_validated = remove_pre_cohort_courses(
        #         df_course_validated, self.cfg.student_id_col
        #     )
        # else:
        #     log_pre_cohort_courses(df_course_validated, self.cfg.student_id_col)

        # Course exploratory EDA (pre-standardize)
        log_grade_distribution(df_course_validated)
        # Log high null columns
        log_high_null_columns(df_course_validated)
        # Standardize course data
        LOGGER.info(" Standardizing course data:")

        df_course_standardized = self.course_std.standardize(df_course_validated)

        LOGGER.info(" Course data standardized.")

        for label, df in [
            ("Standardized cohort", df_cohort_standardized),
            ("Standardized course", df_course_standardized),
        ]:
            require(
                student_id_col in df.columns,
                f"{label} missing required column: {student_id_col}",
            )
            nulls = int(df[student_id_col].isna().sum())
            require(
                nulls == 0, f"{label} contains {nulls} null {student_id_col} values."
            )

        LOGGER.info(
            f'Validated that cohort and course files both have a {student_id_col} column with no nulls.'
        )

        #TODO: Add gateway or developmental flag handling?

        # Log changes before and after pre-processing
        log_record_drops("Cohort", df_cohort_raw, df_cohort_standardized)
        log_record_drops("Course", df_course_raw, df_course_standardized)

        LOGGER.info(
            " Listing grouped entry year/term and academic year/term for *standardized* cohort and course data files: "
        )

        # Logs entry and academic year/term pairs, grouped and sorted (Edvise column names)
        log_terms(df_cohort_standardized, "entry_year", "entry_term")
        log_terms(df_course_standardized, "academic_year", "academic_term")

        # --- Check that standardized cohort/course files aren't empty ---
        require(len(df_cohort_standardized) > 0, "df_cohort_standardized is empty.")
        require(len(df_course_standardized) > 0, "df_course_standardized is empty.")

        # --- Write results ---
        write_parquet(
            df_cohort_standardized,
            os.path.join(current_run_path, "df_cohort_standardized.parquet"),
        )
        write_parquet(
            df_course_standardized,
            os.path.join(current_run_path, "df_course_standardized.parquet"),
        )


if __name__ == "__main__":
    args = parse_data_audit_args()
    apply_bronze_training_inputs_sys_path(args)
    task = ESDataAuditTask(args)
    run_data_audit_with_training_events(args, task)
    flush_data_audit_logging()
