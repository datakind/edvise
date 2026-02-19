"""
Generalized data audit task. Schema-specific behavior is injected via DataAuditBackend.
"""

import argparse
import json
import logging
import os
import typing as t

import pandas as pd

from edvise.dataio.read import read_config
from edvise.dataio.write import write_parquet
from edvise.dataio.paths import pick_existing_path
from edvise.utils.databricks import get_spark_session
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
from edvise.shared.logger import resolve_run_path, local_fs_path
from edvise.shared.validation import require

LOGGER = logging.getLogger(__name__)

ConverterFunc = t.Callable[[pd.DataFrame], pd.DataFrame]


class DataAuditBackend(t.NamedTuple):
    """Schema-specific pieces for the data audit. Pass from script (PDP or ES)."""

    config_schema: type
    cohort_standardizer: t.Any  # BaseStandardizer instance
    course_standardizer: t.Any  # BaseStandardizer instance
    read_raw_cohort_data: t.Callable[..., pd.DataFrame]
    read_raw_course_data: t.Callable[..., pd.DataFrame]
    raw_cohort_schema: type
    raw_course_schema: type
    log_basename: str
    default_course_converter: t.Optional[ConverterFunc] = None


class DataAuditTask:
    """Encapsulates the data preprocessing logic for the SST pipeline.
    Schema-specific behavior is provided via backend.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        backend: DataAuditBackend,
        course_converter_func: t.Optional[ConverterFunc] = None,
        cohort_converter_func: t.Optional[ConverterFunc] = None,
    ):
        self.args = args
        self._backend = backend
        self.cfg = read_config(
            file_path=self.args.config_file_path,
            schema=backend.config_schema,
        )
        self.spark = get_spark_session()
        self.cohort_std = backend.cohort_standardizer
        self.course_std = backend.course_standardizer
        self.cohort_converter_func = cohort_converter_func
        self.course_converter_func = (
            course_converter_func
            if course_converter_func is not None
            else backend.default_course_converter
        )

    def run(self) -> None:
        """Executes the data audit step."""
        read_cohort = self._backend.read_raw_cohort_data
        read_course = self._backend.read_raw_course_data
        cohort_schema = self._backend.raw_cohort_schema
        course_schema = self._backend.raw_course_schema
        log_basename = self._backend.log_basename

        current_run_path = resolve_run_path(
            self.args, self.cfg, self.args.silver_volume_path
        )
        os.makedirs(current_run_path, exist_ok=True)
        local_run_path = local_fs_path(current_run_path)
        os.makedirs(local_run_path, exist_ok=True)

        log_file_path = os.path.join(local_run_path, log_basename)
        abs_log_file_path = os.path.abspath(log_file_path)
        try:
            root_logger = logging.getLogger()
            for h in list(root_logger.handlers):
                if (
                    isinstance(h, logging.FileHandler)
                    and getattr(h, "baseFilename", None) == abs_log_file_path
                ):
                    root_logger.removeHandler(h)
                    try:
                        h.close()
                    except Exception:
                        pass
            file_handler = logging.FileHandler(abs_log_file_path, mode="w")
            file_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
            )
            root_logger.addHandler(file_handler)
            LOGGER.info(
                "File logging initialized. Logs will be overwritten at: %s",
                log_file_path,
            )
        except Exception as e:
            LOGGER.exception(
                "Failed to initialize file logging at %s: %s", log_file_path, e
            )

        cohort_dataset_raw_path = pick_existing_path(
            getattr(self.args, "cohort_dataset_validated_path", None),
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_cohort}",
            "cohort",
        )
        course_dataset_raw_path = pick_existing_path(
            getattr(self.args, "course_dataset_validated_path", None),
            f"{self.args.bronze_volume_path}/{self.cfg.datasets.raw_course}",
            "course",
        )

        LOGGER.info(" Loading raw cohort and course datasets:")
        df_cohort_raw = read_cohort(
            file_path=cohort_dataset_raw_path,
            schema=None,
            spark_session=self.spark,
        )

        dttm_formats = ["ISO8601", "%Y%m%d.0"]
        for fmt in dttm_formats:
            try:
                df_course_raw = read_course(
                    file_path=course_dataset_raw_path,
                    schema=None,
                    dttm_format=fmt,
                    spark_session=self.spark,
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                " Failed to parse course data with all known datetime formats."
            )

        for label, df in [
            ("Raw cohort", df_cohort_raw),
            ("Raw course", df_course_raw),
        ]:
            require(len(df_cohort_raw) > 0, f"{label} dataset is empty (0 rows).")
            require(len(df_course_raw) > 0, f"{label} dataset is empty (0 rows).")

        LOGGER.info(
            " Loaded raw cohort and course data: checking for mismatches in cohort and course files: "
        )
        log_misjoined_records(df_cohort_raw, df_course_raw)
        LOGGER.info(
            " Listing grouped cohort year and terms and academic year and terms for raw cohort and course data files: "
        )
        log_terms(df_course_raw, df_cohort_raw)

        LOGGER.info(" Reading and schema validating cohort data:")
        df_cohort_validated = read_cohort(
            file_path=cohort_dataset_raw_path,
            schema=cohort_schema,
            converter_func=self.cohort_converter_func,
            spark_session=self.spark,
        )

        LOGGER.info(" Standardizing cohort data:")
        df_cohort_standardized = self.cohort_std.standardize(df_cohort_validated)
        LOGGER.info(" Cohort data standardized.")

        student_id_col = getattr(self.cfg, "student_id_col", None) or "student_id"

        LOGGER.info(
            " Reading and schema validating course data, handling any duplicates:"
        )
        for fmt in dttm_formats:
            try:
                df_course_validated = read_course(
                    file_path=course_dataset_raw_path,
                    schema=course_schema,
                    dttm_format=fmt,
                    converter_func=self.course_converter_func,
                    spark_session=self.spark,
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(
                " Failed to parse course data with all known datetime formats."
            )
        LOGGER.info(" Course data read and schema validated, duplicates handled.")

        try:
            include_pre_cohort = self.cfg.preprocessing.include_pre_cohort_courses
        except AttributeError:
            raise AttributeError(
                "Config error: 'include_pre_cohort_courses' is missing. "
                "Please set it explicitly in the config file under 'preprocessing'."
            )
        if not include_pre_cohort:
            df_course_validated = remove_pre_cohort_courses(
                df_course_validated, self.cfg.student_id_col
            )
        else:
            log_pre_cohort_courses(df_course_validated, self.cfg.student_id_col)

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
                nulls == 0,
                f"{label} contains {nulls} null {student_id_col} values.",
            )
        LOGGER.info(
            " Validated that cohort and course files both have a 'student_id' column with no nulls."
        )

        ids, cips, has_upper_level, lower_ids, lower_cips = (
            compute_gateway_course_ids_and_cips(df_course_standardized)
        )

        if self.args.job_type == "training":
            LOGGER.info(
                "Existing config course IDs and subject areas: %s | %s",
                self.cfg.preprocessing.features.key_course_ids,
                self.cfg.preprocessing.features.key_course_subject_areas,
            )
            if has_upper_level:
                if (
                    lower_ids
                    and lower_cips
                    and len(lower_ids) <= 10
                    and len(lower_cips) <= 10
                ):
                    LOGGER.warning(
                        " Upper-level (>=200) gateway courses detected. Auto-populating config with LOWER-level (<200) "
                        "gateway courses and CIP codes only. Please confirm with the school and adjust if needed."
                    )
                    update_key_courses_and_cips(
                        self.args.config_file_path,
                        key_course_ids=lower_ids,
                        key_course_subject_areas=lower_cips,
                    )
                    existing_ids = set(
                        self.cfg.preprocessing.features.key_course_ids or []
                    )
                    existing_cips = set(
                        self.cfg.preprocessing.features.key_course_subject_areas or []
                    )
                    self.cfg.preprocessing.features.key_course_ids = list(
                        existing_ids.union(lower_ids)
                    )
                    self.cfg.preprocessing.features.key_course_subject_areas = list(
                        existing_cips.union(lower_cips)
                    )
                    LOGGER.info(
                        "New config course IDs and subject areas: %s | %s",
                        self.cfg.preprocessing.features.key_course_ids,
                        self.cfg.preprocessing.features.key_course_subject_areas,
                    )
                else:
                    LOGGER.warning(
                        " Skipping auto-populating of config: upper-level gateways present but no acceptable lower-level set."
                    )
            elif len(ids) <= 25 and len(cips) <= 25:
                LOGGER.info(
                    " Auto-populating config with below course IDs and CIP codes: change if necessary"
                )
                update_key_courses_and_cips(
                    self.args.config_file_path,
                    key_course_ids=ids,
                    key_course_subject_areas=cips,
                )
                existing_ids = set(self.cfg.preprocessing.features.key_course_ids or [])
                existing_cips = set(
                    self.cfg.preprocessing.features.key_course_subject_areas or []
                )
                self.cfg.preprocessing.features.key_course_ids = list(
                    existing_ids.union(ids)
                )
                self.cfg.preprocessing.features.key_course_subject_areas = list(
                    existing_cips.union(cips)
                )
                LOGGER.info(
                    "New config course IDs and subject areas: %s | %s",
                    self.cfg.preprocessing.features.key_course_ids,
                    self.cfg.preprocessing.features.key_course_subject_areas,
                )
            else:
                LOGGER.warning(
                    " Skipping auto-populating of config due to too many IDs that were identified. "
                    "Please check in with school and manually update config."
                )

        log_record_drops(
            df_cohort_raw,
            df_cohort_standardized,
            df_course_raw,
            df_course_standardized,
        )
        LOGGER.info(
            " Listing grouped cohort year and terms and academic year and terms for *standardized* cohort and course data files: "
        )
        log_terms(df_course_standardized, df_cohort_standardized)

        require(len(df_cohort_standardized) > 0, "df_cohort_standardized is empty.")
        require(len(df_course_standardized) > 0, "df_course_standardized is empty.")

        write_parquet(
            df_cohort_standardized,
            os.path.join(current_run_path, "df_cohort_standardized.parquet"),
        )
        write_parquet(
            df_course_standardized,
            os.path.join(current_run_path, "df_course_standardized.parquet"),
        )
