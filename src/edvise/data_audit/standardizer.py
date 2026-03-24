import pandas as pd
import logging
import typing as t

LOGGER = logging.getLogger(__name__)

from edvise.utils.drop_columns_safely import drop_columns_safely
from edvise.utils.data_cleaning import (
    drop_course_rows_missing_identifiers,
    strip_trailing_decimal_strings,
    replace_na_firstgen_and_pell,
    handling_duplicates,
)
from edvise.data_audit.custom_cleaning import (
    keep_earlier_record,
    drop_readmits,
    assign_numeric_grade,
)
from .eda import (
    log_high_null_columns,
    print_credential_and_enrollment_types_and_intensities,
    print_retention,
    log_grade_distribution,
    check_bias_variables,
    drop_unpopulated_bias_columns,
    find_dupes,
    log_top_majors,
    check_pf_grade_consistency,
    validate_credit_consistency,
)

# TODO think of a better name than standardizer


class BaseStandardizer:
    "This preps data for feature gen by"

    "- dropping useless, redundant, unwanted cols"
    "- ensuring columns exist by adding missing cols (this prevents many if statements in later steps)"
    "- dropping the pdp course rows that are missing for students"

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
        log_high_null_columns(df)
        print_credential_and_enrollment_types_and_intensities(df)
        print_retention(df)
        log_top_majors(df)
        check_bias_variables(df)
        cols_to_drop = [
            # not a viable target variable, but highly correlated with it
            "time_to_credential",
            # not all demographics used for target variable bias checks
            "incarcerated_status",
            "military_status",
            "employment_status",
            "disability_status",
            "naspa_first_generation",
            # redundant; we have course dataset fields/features for these i.e "program_of_study"
            "attendance_status_term_1",
            "program_of_study_year_1",
            "program_of_study_term_1",
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
        df = strip_trailing_decimal_strings(df, cols=["course_number", "course_cip"])
        df = drop_course_rows_missing_identifiers(df)
        log_high_null_columns(df)
        log_grade_distribution(df)
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

        df = drop_columns_safely(df, cols_to_drop)
        df = self.add_empty_columns_if_missing(
            df, {"term_program_of_study": (None, "string")}
        )
        return df


class ESCohortStandardizer(BaseStandardizer):
    """
    Custom Cohort Standardizer. Operates similarly to PDP's cohort standardizer.
    """

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean up and drop some columns from raw cohort dataset.

        Args:
            df: cohort dataframe
        """
        # Log credential types and enrollment types
        print_credential_and_enrollment_types_and_intensities(df)
        # Log high values of NAs
        log_high_null_columns(df)
        # Logs missing bias variables
        check_bias_variables(df)
        # Logs top majors
        log_top_majors(df)

        # Replaces NA fields with "N" in pell and first_gen columns (standardizes to Y/N)
        df = replace_na_firstgen_and_pell(df)

        # Finds and logs duplicates on primary keys; runs drop_readmits, then keep_earlier_record if needed
        primary_keys = ["student_id", "cohort_term"]
        LOGGER.info("Checking for cohort file duplicates on %s...", primary_keys)
        find_dupes(df, primary_keys)
        LOGGER.info("Dropping readmits")
        df = drop_readmits(df)
        LOGGER.info("Dropped readmits: checking again for duplicates...")
        dupes = find_dupes(df, primary_keys)
        if len(dupes) == 0:
            LOGGER.info("No duplicates found after dropping readmits.")
        else:
            df = keep_earlier_record(df)
            LOGGER.info(
                "Duplicates still found; running keep_earlier_record; checking again for duplicates..."
            )
            dupes = find_dupes(df, primary_keys)
            if len(dupes) == 0:
                LOGGER.info("No duplicates found after keep_earlier_record.")
            else:
                LOGGER.warning(
                    "Duplicates still remain after keep_earlier_record func. Investigate further."
                )

        # Drops unused, unpopulated bias columns
        df = drop_unpopulated_bias_columns(df)
        return df


class ESCourseStandardizer(BaseStandardizer):
    """
    Custom Course Standardizer. Operates similarly to PDP's course standardizer.
    """

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop some columns and anomalous rows from raw course dataset.

        Args:
            df: As output by :func:`dataio.read_raw_pdp_course_data_from_file()` .
        """
        df = strip_trailing_decimal_strings(
            df, cols=["course_number", "course_num", "course_cip"]
        )
        # Log high values of NAs
        log_high_null_columns(df)
        log_grade_distribution(df)

        # Finds and logs duplicates on primary keys; runs handling_duplicates
        primary_keys = ["student_id", "term", "course_subject", "course_num"]
        LOGGER.info("Checking for course file duplicates on %s...", primary_keys)
        find_dupes(df, primary_keys)
        df = handling_duplicates(df, schema_type="es", unique_cols=primary_keys)

        # Runs check_pf_grade_consistency func
        check_pf_grade_consistency(df)
        # Runs validate_credit_consistency func
        validate_credit_consistency(df, None, None)
        # Runs assign_numeric_grade func
        df = assign_numeric_grade(df)
        df = self.add_empty_columns_if_missing(df, {})
        return df
