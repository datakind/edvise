import pandas as pd
import logging
import re
import typing as t
from collections.abc import Iterable
from typing import List

from . import types

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
    num_rows_before = len(df)

    # Identify rows missing either identifier
    id_cols = ["course_prefix", "course_number"]
    present_mask = df[id_cols].notna().all(axis=1)
    drop_mask = ~present_mask
    num_dropped = int(drop_mask.sum())

    if num_dropped > 0:
        # Breakdown by enrolled_at_other_institution_s within the dropped set
        if "enrolled_at_other_institution_s" in df.columns:
            dropped = (
                df.loc[drop_mask, "enrolled_at_other_institution_s"]
                .astype("string")
                .str.upper()
            )
            count_y = int((dropped == "Y").sum())
            count_not_y = num_dropped - count_y
            pct_y = 100.0 * count_y / num_dropped
            pct_not_y = 100.0 * count_not_y / num_dropped

            LOGGER.warning(
                "Dropped %s rows from course dataset due to missing identifiers. "
                "Of these, %s (%.1f%%) had 'Y' in enrolled_at_other_institution_s; "
                "%s (%.1f%%) did not.",
                num_dropped,
                count_y,
                pct_y,
                count_not_y,
                pct_not_y,
            )
        else:
            LOGGER.warning(
                "Dropped %s rows from course dataset due to missing identifiers. "
                "Column 'enrolled_at_other_institution_s' not found; cannot compute alignment breakdown.",
                num_dropped,
            )

    # Keep only rows with both identifiers present
    df = df.loc[present_mask].reset_index(drop=True)
    num_rows_after = len(df)
    return df


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
    df_course = df_course.groupby(student_id_col, group_keys=False).apply(
        lambda df_course: df_course[
            (df_course["academic_year"] > df_course["cohort"])
            | (
                (df_course["academic_year"] == df_course["cohort"])
                & (df_course["academic_term"] >= df_course["cohort_term"])
            )
        ]
    )


def replace_na_firstgen_and_pell(df_cohort: pd.DataFrame) -> pd.DataFrame:
    if "pell_status_first_year" in df_cohort.columns:
        LOGGER.info("Before replacing 'pell_status_first_year':\n%s", 
                    df_cohort["pell_status_first_year"].value_counts(dropna=False))
        na_pell = (df_cohort["pell_status_first_year"] == "UK").sum()
        df_cohort["pell_status_first_year"] = df_cohort[
            "pell_status_first_year"
        ].replace("UK", "N")
        LOGGER.info('Filled %s NAs in "pell_status_first_year" to "N".', int(na_pell))
        df_cohort["pell_status_first_year"] = df_cohort[
            "pell_status_first_year"
        ].fillna("N")
        LOGGER.info("After replacing 'pell_status_first_year':\n%s", 
                    df_cohort["pell_status_first_year"].value_counts(dropna=False))
    else:
        LOGGER.warning(
            'Column "pell_status_first_year" not found; skipping Pell status NA replacement.'
        )
    if "first_gen" in df_cohort.columns:
        LOGGER.info("Before filling 'first_gen':\n%s", 
                    df_cohort["first_gen"].value_counts(dropna=False))
        na_first = df_cohort["first_gen"].isna().sum()
        df_cohort["first_gen"] = df_cohort["first_gen"].fillna("N")
        LOGGER.info('Filled %s NAs in "first_gen" with "N".', int(na_first))
        LOGGER.info("After filling 'first_gen':\n%s", 
                    df_cohort["first_gen"].value_counts(dropna=False))
    else:
        LOGGER.warning(
            'Column "first_gen" not found; skipping first-gen NA replacement.'
        )
    return df_cohort


def strip_trailing_decimal_strings(df_course: pd.DataFrame) -> pd.DataFrame:
    for col in ["course_number", "course_cip"]:
        if col in df_course.columns:
            df_course[col] = df_course[col].astype("string")
            pre_truncated = df_course[col].copy()
            df_course[col] = df_course[col].str.rstrip(".0")
            truncated = (pre_truncated != df_course[col]).sum(min_count=1)
            LOGGER.info(
                'Stripped trailing ".0" in %s rows for column "%s".',
                int(truncated or 0),
                col,
            )
        else:
            LOGGER.warning('Column "%s" not found', col)
    return df_course


def handling_duplicates(df_course: pd.DataFrame) -> pd.DataFrame:
    """
    Dropping duplicate course records, except:
    - if duplicate-key rows all share the SAME course_name, keep them and
      suffix course_number with -01, -02, ... instead of dropping.
    """
    unique_cols = [
        "study_id",
        "academic_year",
        "academic_term",
        "course_prefix",
        "course_number",
        "section_id",
    ]

    dup_mask = df_course.duplicated(unique_cols, keep=False)
    if (
        dup_mask.any()
        and "course_name" in df_course.columns
        and "course_number" in df_course.columns
    ):
        same_name_idx = []
        for _, idx in (
            df_course.loc[dup_mask].groupby(unique_cols, dropna=False).groups.items()
        ):
            idx = list(idx)
            if len(idx) <= 1:
                continue
            names = df_course.loc[idx, "course_name"]
            if names.nunique(dropna=False) == 1:
                same_name_idx.extend(idx)

        if same_name_idx:
            deduped_course_numbers = (
                df_course.loc[dup_mask, :]
                .sort_values(
                    by=unique_cols + ["number_of_credits_attempted"],
                    ascending=False,
                    ignore_index=False,
                )
                .assign(
                    grp_num=lambda d: d.groupby(unique_cols)["course_number"].transform(
                        "cumcount"
                    )
                    + 1,
                    course_number=lambda d: d["course_number"]
                    .astype("string")
                    .str.cat(d["grp_num"].astype(int).map("{:02d}".format), sep="-"),
                )
                .loc[:, ["course_number"]]
            )
            to_apply = deduped_course_numbers.reindex(same_name_idx).dropna(how="all")
            df_course.loc[to_apply.index, "course_number"] = to_apply["course_number"]

    dupe_rows = df.loc[df.duplicated(unique_cols, keep=False), :].sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )
    LOGGER.warning("%s duplicate rows found & dropped", int(len(dupe_rows) / 2))

    df = df.drop_duplicates(subset=unique_cols, keep="first").sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )
    return df


def compute_gateway_course_ids_and_cips(df_course: pd.DataFrame) -> List[str]:
    """
    Build a list of course IDs and CIP codes for Math/English gateway courses.
    Filter: math_or_english_gateway in {"M", "E"}
    ID format: "<course_prefix><course_number>" (both coerced to strings, trimmed)
    CIP codes taken from 'course_cip' column
    """
    if not {"math_or_english_gateway", "course_prefix", "course_number"}.issubset(
        df_course.columns
    ):
        LOGGER.warning("Cannot compute key_course_ids: required columns missing.")
        return []

    mask = df_course["math_or_english_gateway"].astype("string").isin({"M", "E"})
    ids = df_course.loc[mask, "course_prefix"].fillna("") + df_course.loc[
        mask, "course_number"
    ].fillna("")
    
    cips = (
        df_course.loc[mask, "course_cip"]
        .astype(str)
        .fillna("")
        .str.strip()
    )

    # edit this to auto populate the config
    cips = cips[cips.ne("") & cips.str.lower().ne("nan")].drop_duplicates()
    ids = ids[ids.str.strip().ne("") & ids.str.lower().ne("nan")].drop_duplicates()
    
    LOGGER.info(f"Identified {len(ids)} unique gateway course IDs: {ids.tolist()}")
    LOGGER.info(f"Identified {len(cips)} unique CIP codes: {cips.tolist()}")

    return [ids.tolist(), cips.tolist()]
