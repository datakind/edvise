import pandas as pd
import logging
import re
import typing as t
from collections.abc import Iterable
from edvise.utils import types
from edvise.dataio.pdp_course_converters import dedupe_by_renumbering_courses

LOGGER = logging.getLogger(__name__)

RE_VARIOUS_PUNCTS = re.compile(r"[!()*+\,\-./:;<=>?[\]^_{|}~]")
RE_QUOTATION_MARKS = re.compile(r"[\'\"\`]")


def unique_elements_in_order(eles: Iterable) -> Iterable:
    """Get unique elements from an iterable, in order of appearance."""
    seen = set()  # type: ignore
    seen_add = seen.add
    for ele in eles:
        if ele not in seen:
            seen_add(ele)
            yield ele


def convert_to_snake_case(col: str) -> str:
    """Convert column name into snake case, without punctuation."""
    col = RE_VARIOUS_PUNCTS.sub(" ", col)
    col = RE_QUOTATION_MARKS.sub("", col)
    # TODO: *pretty sure* this could be cleaner and more performant, but shrug
    words = re.sub(
        r"([A-Z][a-z]+)", r" \1", re.sub(r"([A-Z]+|[0-9]+|\W+)", r" \1", col)
    ).split()
    return "_".join(words).lower()


def convert_intensity_time_limits(
    unit: t.Literal["term", "year"],
    intensity_time_limits: types.IntensityTimeLimitsType,
    *,
    num_terms_in_year: int,
) -> dict[str, float]:
    """
    Convert enrollment intensity-specific time limits into a particular ``unit`` ,
    whether input limits were given in units of years or terms.

    Args:
        unit: The time unit into which inputs are converted, either "term" or "year".
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms (e.g. [4.0, "year"], [12.0, "term"])
            considered "success" for a school in their particular use case.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert between term- and year-based time limits;
            for example: 4 => FALL, WINTER, SPRING, and SUMMER terms.
    """
    if unit == "year":
        intensity_nums = {
            intensity: num if unit == "year" else num / num_terms_in_year
            for intensity, (num, unit) in intensity_time_limits.items()
        }
    else:
        intensity_nums = {
            intensity: num if unit == "term" else num * num_terms_in_year
            for intensity, (num, unit) in intensity_time_limits.items()
        }
    return intensity_nums


def parse_dttm_values(df: pd.DataFrame, *, col: str, fmt: str) -> pd.Series:
    return pd.to_datetime(df[col], format=fmt)


def uppercase_string_values(df: pd.DataFrame, *, col: str) -> pd.Series:
    return df[col].str.upper()


def replace_values_with_null(
    df: pd.DataFrame, *, col: str, to_replace: str | list[str]
) -> pd.Series:
    return df[col].replace(to_replace=to_replace, value=None)


def cast_to_bool_via_int(df: pd.DataFrame, *, col: str) -> pd.Series:
    return (
        df[col]
        .astype("string")
        .map(
            {
                "1": True,
                "0": False,
                "True": True,
                "False": False,
                "true": True,
                "false": False,
            }
        )
        .astype("boolean")
    )


def strip_upper_strings_to_cats(series: pd.Series) -> pd.Series:
    return series.str.strip().str.upper().astype("category")


def drop_course_rows_missing_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows from raw course dataset missing key course identifiers,
    specifically course prefix and number, which supposedly are partial records
    from students' enrollments at *other* institutions -- not wanted here!
    """

    # HACK: get correct student_id_col

    student_id_col = (
        "student_guid"
        if "student_guid" in df.columns
        else "study_id"
        if "study_id" in df.columns
        else "student_id"
    )
    students_before = df[student_id_col].nunique()

    # Identify rows missing either identifier
    id_cols = ["course_prefix", "course_number"]
    present_mask = df[id_cols].notna().all(axis=1)
    drop_mask = ~present_mask
    num_dropped_rows = int(drop_mask.sum())
    pct_dropped_rows = (num_dropped_rows / len(df) * 100.0) if len(df) else 0.0

    # Keep only rows with both identifiers present
    df_cleaned = df.loc[present_mask].reset_index(drop=True)
    students_after = df_cleaned[student_id_col].nunique()
    dropped_students = students_before - students_after

    # Log dropped rows
    if num_dropped_rows > 0:
        LOGGER.warning(
            " Dropped %s rows (%.1f%%) from course dataset due to missing course_prefix or course_number.",
            num_dropped_rows,
            pct_dropped_rows,
        )

        # Additional analysis of dropped records to detect trends
        df_dropped = df.loc[drop_mask].copy()

        # Log grouped cohort & cohort_term
        if "cohort" in df_dropped.columns and "cohort_term" in df_dropped.columns:
            cohort_group_counts = (
                df_dropped.groupby(
                    ["cohort", "cohort_term"], dropna=False, observed=True
                )
                .size()
                .sort_index()
            )
            LOGGER.info(
                " Grouped counts for missing course identifier (dropped) records by cohort and cohort_term to identify potential trends:\n%s",
                cohort_group_counts.to_string(),
            )

    # Log student alignment breakdown if available
    if "enrolled_at_other_institution_s" in df.columns and num_dropped_rows > 0:
        dropped_flag = (
            df.loc[drop_mask, "enrolled_at_other_institution_s"]
            .astype("string")
            .str.upper()
        )
        count_y = int((dropped_flag == "Y").sum())
        count_not_y = int(num_dropped_rows - count_y)
        pct_y = 100.0 * count_y / num_dropped_rows if num_dropped_rows else 0.0
        pct_not_y = 100.0 * count_not_y / num_dropped_rows if num_dropped_rows else 0.0

        LOGGER.warning(
            " Of dropped rows, %s (%.1f%%) had 'Y' in enrolled_at_other_institution_s; %s (%.1f%%) did not.",
            count_y,
            pct_y,
            count_not_y,
            pct_not_y,
        )

        # Additional warning if too many are not clearly transfer records
        if pct_not_y > 10.0:
            LOGGER.warning(
                " drop_course_rows_missing_identifiers: ⚠️ More than 10%% of dropped rows (%d of %d) "
                "were NOT marked as transfer out records based on 'enrolled_at_other_institution_s'. "
                "This is uncommon: please contact data team for further investigation",
                count_not_y,
                num_dropped_rows,
            )

    # Log fully dropped students
    if dropped_students:
        LOGGER.warning(
            " %d students were fully dropped from the course data due to all their records missing identifiers.",
            dropped_students,
        )

    return df_cleaned


def remove_pre_cohort_courses(df_course: pd.DataFrame) -> pd.DataFrame:
    """
    Removes any course records that occur before a student's cohort start term.

    This ensures that any pre-cohort course records are excluded before generating any features
    in our `student_term_df`. These records can otherwise introduce inconsistencies in
    cumulative features. For example, in retention models, we observed mismatches
    between `cumulative_credits_earned` and `number_of_credits_earned` when using the
    first cohort term as the checkpoint because pre-cohort courses were
    still included in the data when generating these features. To avoid this, we drop all records that occurred
    prior to the student's official cohort start term before feature generation.

    Please rememeber to check with your respective schools during the data assessment call how they would like pre-cohort course records to be handled and if this function needs to be called or not.

    Args:
        df_course

    Returns:
        pd.DataFrame: Filtered DataFrame excluding pre-cohort course records.
    """
    # HACK: infer the correct student id col in raw data from the data itself
    student_id_col = (
        "student_guid"
        if "student_guid" in df_course.columns
        else "study_id"
        if "study_id" in df_course.columns
        else "student_id"
    )

    n_before = len(df_course)
    students_before = df_course[student_id_col].nunique()

    # Build mask for "keep" rows (cohort year or later)
    keep_mask = df_course["academic_year"].ge(df_course["cohort"])

    # Split for logging/analysis
    df_dropped = df_course.loc[~keep_mask].copy()
    df_filtered = df_course.loc[keep_mask].assign(
        cohort_id=lambda df: df["cohort"]
        .astype(str)
        .str.cat(df["cohort_term"].astype(str), sep=" "),
        term_id=lambda df: df["academic_year"]
        .astype(str)
        .str.cat(df["academic_term"].astype(str), sep=" "),
    )

    n_after = len(df_filtered)
    students_after = df_filtered[student_id_col].nunique()
    n_removed = n_before - n_after
    dropped_students_count = students_before - students_after
    pct_removed = (n_removed / n_before) * 100 if n_before else 0.0

    # Summary logging
    if n_removed > 0:
        if pct_removed < 0.1:
            LOGGER.info(
                " remove_pre_cohort_courses: %d pre-cohort course records safely removed (<0.1%% of data).",
                n_removed,
            )
        else:
            LOGGER.info(
                " remove_pre_cohort_courses: %d pre-cohort course records safely removed (%.1f%% of data).",
                n_removed,
                pct_removed,
            )

        if dropped_students_count > 0:
            LOGGER.warning(
                " remove_pre_cohort_courses: %d students were fully dropped (i.e., only had pre-cohort records).",
                dropped_students_count,
            )

        # Log grouped academic_year and  academic_term (only if both present)
        if (
            "academic_year" in df_dropped.columns
            and "academic_term" in df_dropped.columns
        ):
            cohort_group_counts = (
                df_dropped.groupby(
                    ["academic_year", "academic_term"], dropna=False, observed=True
                )
                .size()
                .sort_index()
            )
            LOGGER.info(
                " Grouped counts for pre-cohort (dropped) records by academic_year and academic_term to identify potential trends:\n%s",
                cohort_group_counts.to_string(),
            )

    else:
        LOGGER.info("remove_pre_cohort_courses: No pre-cohort course records found.")

    return df_filtered


def replace_na_firstgen_and_pell(df_cohort: pd.DataFrame) -> pd.DataFrame:
    if "pell_status_first_year" in df_cohort.columns:
        LOGGER.info(
            " Before replacing 'pell_status_first_year':\n%s",
            df_cohort["pell_status_first_year"].value_counts(dropna=False),
        )
        na_pell = df_cohort["pell_status_first_year"].isna().sum()
        df_cohort["pell_status_first_year"] = df_cohort[
            "pell_status_first_year"
        ].fillna("N")
        LOGGER.info(
            ' Filled %s NAs in "pell_status_first_year" to "N".',
            int(na_pell),
        )
        LOGGER.info(
            " After replacing 'pell_status_first_year':\n%s",
            df_cohort["pell_status_first_year"].value_counts(dropna=False),
        )
    else:
        LOGGER.warning(
            ' Column "pell_status_first_year" not found; skipping Pell status NA replacement.'
        )

    if "first_gen" in df_cohort.columns:
        LOGGER.info(
            " Before filling 'first_gen':\n%s",
            df_cohort["first_gen"].value_counts(dropna=False),
        )
        na_first = df_cohort["first_gen"].isna().sum()
        df_cohort["first_gen"] = df_cohort["first_gen"].fillna("N")
        LOGGER.info(
            ' Filled %s NAs in "first_gen" with "N".',
            int(na_first),
        )
        LOGGER.info(
            " After filling 'first_gen':\n%s",
            df_cohort["first_gen"].value_counts(dropna=False),
        )
    else:
        LOGGER.warning(
            ' Column "first_gen" not found; skipping first-gen NA replacement.'
        )
    return df_cohort


def strip_trailing_decimal_strings(df_course: pd.DataFrame) -> pd.DataFrame:
    for col in ["course_number", "course_cip"]:
        if col in df_course.columns:
            df_course[col] = df_course[col].astype("string")
            pre_truncated = df_course[col].copy()

            # Only remove literal ".0" at the end of the string
            df_course[col] = df_course[col].str.replace(r"\.0$", "", regex=True)

            truncated = (pre_truncated != df_course[col]).sum(min_count=1)
            LOGGER.info(
                ' Stripped trailing ".0" in %s rows for column "%s".',
                int(truncated or 0),
                col,
            )
        else:
            LOGGER.warning(' Column "%s" not found', col)
    return df_course


def handling_duplicates(df_course: pd.DataFrame) -> pd.DataFrame:
    """
    Dropping duplicate course records and keeping the one with the higher number of credits, except:
    - if duplicate-key rows have DIFFERENT course_names, keep them and
      suffix course_number with -01, -02, ... instead of dropping.
    """
    # HACK: infer the correct student id col in raw data from the data itself
    student_id_col = (
        "student_guid"
        if "student_guid" in df_course.columns
        else "study_id"
        if "study_id" in df_course.columns
        else "student_id"
    )
    unique_cols = [
        student_id_col,
        "academic_year",
        "academic_term",
        "course_prefix",
        "course_number",
        "section_id",
    ]

    # Check for duplicate key rows
    dup_mask = df_course.duplicated(unique_cols, keep=False)

    if dup_mask.any() and "course_name" in df_course.columns:
        # Group and check for variation in course_name
        to_renumber = []
        for _, idx in (
            df_course.loc[dup_mask].groupby(unique_cols, dropna=False).groups.items()
        ):
            idx = list(idx)
            if len(idx) <= 1:
                continue
            names = df_course.loc[idx, "course_name"]
            if names.nunique(dropna=False) > 1:
                to_renumber.extend(idx)

        if to_renumber:
            df_course = dedupe_by_renumbering_courses(df_course)
            return df_course

    # If we reach here, these are true duplicates → drop them
    dupe_rows = df_course.loc[dup_mask, :].sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )
    pct_dup = (len(dupe_rows) / len(df_course)) * 100
    if pct_dup < 0.1:
        LOGGER.warning(
            " %s (<0.1%% of data) true duplicate rows found & dropped",
            len(dupe_rows) // 2,  # integer count of dropped pairs
        )
    else:
        LOGGER.warning(
            " %s (%.1f%% of data) true duplicate rows found & dropped",
            len(dupe_rows) // 2,
            pct_dup,
        )

    df_course = df_course.drop_duplicates(subset=unique_cols, keep="first").sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )

    return df_course
