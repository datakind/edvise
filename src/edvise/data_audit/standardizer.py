import pandas as pd
import logging
import typing as t

LOGGER = logging.getLogger(__name__)

from edvise.utils.drop_columns_safely import drop_columns_safely
from edvise.utils.data_cleaning import (
    drop_course_rows_missing_identifiers,
    strip_trailing_decimal_strings,
    replace_na_firstgen_and_pell,
    compute_gateway_course_ids_and_cips,
    handling_duplicates,
    remove_pre_cohort_courses,
)

# TODO think of a better name than standardizer


class BaseStandardizer:
    def add_empty_columns_if_missing(
        self,
        df: pd.DataFrame,
        col_val_dtypes: dict[str, tuple[t.Optional[t.Any], str]],
    ) -> pd.DataFrame:
        return df.assign(
            **{
                col: pd.Series(data=val, index=df.index, dtype=dtype)
                for col, (val, dtype) in col_val_dtypes.items()
                if col not in df.columns
            }
        )


class PDPCohortStandardizer(BaseStandardizer):
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop some columns from raw cohort dataset.

        Args:
            df: As output by :func:`dataio.read_raw_pdp_cohort_data_from_file()` .
        """
        cols_to_drop = [
            # not a viable target variable, but highly correlated with it
            "time_to_credential",
            # not all demographics used for target variable bias checks
            "incarcerated_status",
            "military_status",
            "employment_status",
            "disability_status",
            "naspa_first_generation",
            # redundant
            "attendance_status_term_1",
            # covered indirectly by course dataset fields/features
            "gateway_math_status",
            "gateway_english_status",
            "attempted_gateway_math_year_1",
            "attempted_gateway_english_year_1",
            "completed_gateway_math_year_1",
            "completed_gateway_english_year_1",
            "gateway_math_grade_y_1",
            "gateway_english_grade_y_1",
            "attempted_dev_math_y_1",
            "attempted_dev_english_y_1",
            "completed_dev_math_y_1",
            "completed_dev_english_y_1",
            # let's assume we don't need other institution "demographics"
            "most_recent_bachelors_at_other_institution_state",
            "most_recent_associates_or_certificate_at_other_institution_state",
            "most_recent_last_enrollment_at_other_institution_state",
            "first_bachelors_at_other_institution_state",
            "first_associates_or_certificate_at_other_institution_state",
            "most_recent_bachelors_at_other_institution_carnegie",
            "most_recent_associates_or_certificate_at_other_institution_carnegie",
            "most_recent_last_enrollment_at_other_institution_carnegie",
            "first_bachelors_at_other_institution_carnegie",
            "first_associates_or_certificate_at_other_institution_carnegie",
            "most_recent_bachelors_at_other_institution_locale",
            "most_recent_associates_or_certificate_at_other_institution_locale",
            "most_recent_last_enrollment_at_other_institution_locale",
            "first_bachelors_at_other_institution_locale",
            "first_associates_or_certificate_at_other_institution_locale",
        ]
        col_val_dtypes = {
            "years_to_latest_associates_at_cohort_inst": (None, "Int8"),
            "years_to_latest_certificate_at_cohort_inst": (None, "Int8"),
            "years_to_latest_associates_at_other_inst": (None, "Int8"),
            "years_to_latest_certificate_at_other_inst": (None, "Int8"),
            "first_year_to_associates_at_cohort_inst": (None, "Int8"),
            "first_year_to_certificate_at_cohort_inst": (None, "Int8"),
            "first_year_to_associates_at_other_inst": (None, "Int8"),
            "first_year_to_certificate_at_other_inst": (None, "Int8"),
        }
        df = drop_columns_safely(df, cols_to_drop)
        df = replace_na_firstgen_and_pell(df)
        df = self.add_empty_columns_if_missing(df, col_val_dtypes)

        return df


class PDPCourseStandardizer(BaseStandardizer):
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop some columns and anomalous rows from raw course dataset.

        Args:
            df: As output by :func:`dataio.read_raw_pdp_course_data_from_file()` .
        """
        df = strip_trailing_decimal_strings(df)
        df = drop_course_rows_missing_identifiers(df)
        df = handling_duplicates(df) 
        # I think this will be pre-ingestion

        cols_to_drop = [
            # student demographics found in raw cohort dataset
            "cohort",
            "cohort_term",
            "student_age",
            "race",
            "ethnicity",
            "gender",
            # course name and aspects of core-ness not needed
            "course_name",
            "core_course_type",
            "core_competency_completed",
            "credential_engine_identifier",
            # enrollment record at other insts not needed
            "enrollment_record_at_other_institution_s_state_s",
            "enrollment_record_at_other_institution_s_carnegie_s",
            "enrollment_record_at_other_institution_s_locale_s",
        ]
        df = remove_pre_cohort_courses(df)
        df = drop_columns_safely(df, cols_to_drop)
        df = self.add_empty_columns_if_missing(
            df, {"term_program_of_study": (None, "string")}
        )
        gateway_course_ids_and_cips = compute_gateway_course_ids_and_cips(df)
        LOGGER.info("Math and English Gateway Courses and CIP codes Identified:", gateway_course_ids_and_cips)
        return df
