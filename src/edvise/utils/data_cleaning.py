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


def drop_course_rows_missing_identifiers(df_course: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows from raw course dataset missing key course identifiers,
    specifically course prefix and number, which supposedly are partial records
    from students' enrollments at *other* institutions -- not wanted here!
    """
    # HACK: infer the correct student id col in raw data from the data itself
    student_id_col = (
        "student_guid"
        if "student_guid" in df_course.columns
        else "study_id"
        if "study_id" in df_course.columns
        else "student_id"
    )
    students_before = df_course[student_id_col].nunique()

    # Identify rows missing either identifier
    id_cols = ["course_prefix", "course_number"]
    present_mask = df_course[id_cols].notna().all(axis=1)
    drop_mask = ~present_mask
    num_dropped_rows = int(drop_mask.sum())
    pct_dropped_rows = (
        (num_dropped_rows / len(df_course) * 100.0) if len(df_course) else 0.0
    )

    # Keep only rows with both identifiers present
    df_cleaned = df_course.loc[present_mask].reset_index(drop=True)
    students_after = df_cleaned[student_id_col].nunique()
    dropped_students = students_before - students_after

    # Log dropped rows
    if num_dropped_rows > 0:
        LOGGER.warning(
            " ⚠️ Dropped %s rows (%.1f%%) from course dataset due to missing course_prefix or course_number.",
            num_dropped_rows,
            pct_dropped_rows,
        )

    # Warn if any full academic term was completely removed
    if {"academic_year", "academic_term"}.issubset(df_course.columns):
        original_terms = (
            df_course.loc[:, ["academic_year", "academic_term"]]
            .drop_duplicates()
            .assign(_present=True)
        )
        cleaned_terms = (
            df_cleaned.loc[:, ["academic_year", "academic_term"]]
            .drop_duplicates()
            .assign(_present=True)
        )

        merged_terms = original_terms.merge(
            cleaned_terms,
            on=["academic_year", "academic_term"],
            how="left",
            suffixes=("", "_cleaned"),
            indicator=True,
        )

        dropped_terms = merged_terms.loc[
            merged_terms["_merge"] == "left_only", ["academic_year", "academic_term"]
        ]

        if not dropped_terms.empty:
            TERM_ORDER = {"Spring": 1, "Summer": 2, "Fall": 3, "Winter": 4}

            def parse_year(year_str: str) -> int:
                """
                Extracts the first year as an integer from formats like:
                '2022', '2022-23', or '2022-2023'
                """
                # Grab the first 4 digits
                import re
                match = re.search(r"\d{4}", year_str)
                return int(match.group()) if match else 0

            term_list = sorted(
                [
                    f"{r.academic_term} {r.academic_year}"
                    for r in dropped_terms.itertuples()
                ],
                key=lambda s: (
                    parse_year(s.split()[-1]),                   # handle '2022-23'
                    TERM_ORDER.get(s.split()[0], 99),            # order terms
                ),
            )

            LOGGER.warning(
                " ⚠️ ENTIRE academic term(s) dropped because *all* rows were missing course identifiers: %s",
                ", ".join(term_list),
            )

    # Log transfer-out alignment breakdowns if available
    if "enrolled_at_other_institution_s" in df_course.columns and num_dropped_rows > 0:
        # Normalize the flag just once on the full frame, then slice with drop_mask
        norm_flag = (
            df_course["enrolled_at_other_institution_s"].astype("string").str.upper()
        )

        # Build mutually exclusive masks for the *dropped* rows
        dropped_transfer_mask = drop_mask & (norm_flag == "Y")
        dropped_non_transfer_mask = drop_mask & (
            norm_flag != "Y"
        )  # includes N/blank/NA

        count_y = int(dropped_transfer_mask.sum())
        count_not_y = int(dropped_non_transfer_mask.sum())
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
                " ⚠️ drop_course_rows_missing_identifiers: More than 10%% of dropped rows (%d of %d) "
                "were NOT marked as transfer out records based on 'enrolled_at_other_institution_s'. "
                "This is uncommon: please contact data team for further investigation",
                count_not_y,
                num_dropped_rows,
            )

        # If we have cohort/academic fields, log grouped counts for BOTH segments
        required_cols = {"cohort", "cohort_term", "academic_year", "academic_term"}
        if required_cols.issubset(df_course.columns):

            def _group_and_log(mask: pd.Series, segment_label: str) -> None:
                if not mask.any():
                    return
                df_seg = df_course.loc[mask]

                academic_group_counts = (
                    df_seg.groupby(
                        ["academic_year", "academic_term"], dropna=False, observed=True
                    )
                    .size()
                    .reset_index(name="count")
                    .sort_values(
                        by=["academic_year", "academic_term"], kind="mergesort"
                    )
                )
                LOGGER.info(
                    "Grouped counts by academic year and academic term for %s rows with missing course identifiers:\n%s",
                    segment_label,
                    academic_group_counts.to_string(index=False),
                )

                cohort_group_counts = (
                    df_seg.groupby(
                        ["cohort", "cohort_term"], dropna=False, observed=True
                    )
                    .size()
                    .reset_index(name="count")
                    .sort_values(by=["cohort", "cohort_term"], kind="mergesort")
                )
                LOGGER.info(
                    "Grouped counts by cohort year and cohort term for %s rows with missing course identifiers:\n%s",
                    segment_label,
                    cohort_group_counts.to_string(index=False),
                )

            # Log for NOT-marked-as-transfer (existing behavior)
            _group_and_log(
                dropped_non_transfer_mask, "NOT-marked-as-transfer-out ('N')"
            )

            # NEW: Log for rows MARKED as transfer-outs
            _group_and_log(dropped_transfer_mask, "MARKED-as-transfer-out ('Y')")

    return df_cleaned


def remove_pre_cohort_courses(
    df_course: pd.DataFrame, student_id_col: str
) -> pd.DataFrame:
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

    n_before = len(df_course)
    students_before = df_course[student_id_col].nunique()

    # Build mask for "keep" rows (cohort year or later)
    keep_mask = df_course["academic_year"].ge(df_course["cohort"])

    # Split for logging/analysis
    df_dropped = df_course.loc[~keep_mask].copy()
    df_filtered = df_course.loc[keep_mask]

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
                "  ⚠️ remove_pre_cohort_courses: %d students were fully dropped (i.e., only had pre-cohort records).",
                dropped_students_count,
            )

        # Log grouped cohort and academic year/term counts for dropped pre-cohort records
        required_cols = {"academic_year", "academic_term", "cohort", "cohort_term"}
        if required_cols.issubset(df_dropped.columns):
            # --- Grouped by academic year and term ---
            academic_group_counts = (
                df_dropped.groupby(
                    ["academic_year", "academic_term"], dropna=False, observed=True
                )
                .size()
                .reset_index(name="count")
                .sort_values(by=["academic_year", "academic_term"])
            )
            LOGGER.info(
                "Pre-cohort records grouped by academic year and term:\n%s",
                academic_group_counts.to_string(index=False),
            )

            # --- Grouped by cohort year and term ---
            cohort_group_counts = (
                df_dropped.groupby(
                    ["cohort", "cohort_term"], dropna=False, observed=True
                )
                .size()
                .reset_index(name="count")
                .sort_values(by=["cohort", "cohort_term"])
            )
            LOGGER.info(
                "Pre-cohort records grouped by cohort year and term:\n%s",
                cohort_group_counts.to_string(index=False),
            )

        else:
            missing = required_cols - df_dropped.columns.to_series().index.to_set()
            LOGGER.warning(
                " ⚠️ Could not log full pre-cohort groupings. Missing columns: %s",
                ", ".join(missing),
            )

    return df_filtered


def log_pre_cohort_courses(df_course: pd.DataFrame, student_id_col: str) -> None:
    """
    Logs any course records that occur before a student's cohort start term.

    This is a read-only helper: it does not modify or return the DataFrame.
    It can be used to review how many records would be dropped by
    `remove_pre_cohort_courses` without actually filtering them.
    This is for schools that choose to keep these courses.

    Args:
        df_course (pd.DataFrame): The course-level DataFrame.
        student_id_col (str): Column name for student IDs.

    Returns:
        None
    """
    n_total = len(df_course)
    students_total = df_course[student_id_col].nunique()

    # Identify pre-cohort records
    pre_mask = df_course["academic_year"].lt(df_course["cohort"])
    df_pre = df_course.loc[pre_mask].copy()

    n_pre = len(df_pre)
    students_pre = df_pre[student_id_col].nunique()
    pct_pre = (n_pre / n_total) * 100 if n_total else 0.0

    if n_pre == 0:
        LOGGER.info("log_pre_cohort_courses: No pre-cohort course records found.")
        return

    LOGGER.info(
        "log_pre_cohort_courses: %d pre-cohort course records found (%.1f%% of data) and will be kept "
        "across %d students.",
        n_pre,
        pct_pre,
        students_pre,
    )

    # Students with only pre-cohort records
    pre_only_students = df_pre[student_id_col].unique()
    students_with_only_pre = [
        sid
        for sid in pre_only_students
        if (df_course[student_id_col] == sid).sum()
        == (df_pre[student_id_col] == sid).sum()
    ]
    if students_with_only_pre:
        LOGGER.warning(
            " ⚠️ log_pre_cohort_courses: %d students have only pre-cohort records.",
            len(students_with_only_pre),
        )

    # Log grouped cohort and academic year/term counts for dropped pre-cohort records
    required_cols = {"academic_year", "academic_term", "cohort", "cohort_term"}
    if required_cols.issubset(df_pre.columns):
        # --- Grouped by academic year and term ---
        academic_group_counts = (
            df_pre.groupby(
                ["academic_year", "academic_term"], dropna=False, observed=True
            )
            .size()
            .reset_index(name="count")
            .sort_values(by=["academic_year", "academic_term"])
        )
        LOGGER.info(
            "Pre-cohort records grouped by academic year and term:\n%s",
            academic_group_counts.to_string(index=False),
        )

        # --- Grouped by cohort year and term ---
        cohort_group_counts = (
            df_pre.groupby(["cohort", "cohort_term"], dropna=False, observed=True)
            .size()
            .reset_index(name="count")
            .sort_values(by=["cohort", "cohort_term"])
        )
        LOGGER.info(
            "Pre-cohort records grouped by cohort year and term:\n%s",
            cohort_group_counts.to_string(index=False),
        )

    else:
        missing = required_cols - df_pre.columns.to_series().index.to_set()
        LOGGER.warning(
            " ⚠️ Could not log full pre-cohort groupings. Missing columns: %s",
            ", ".join(missing),
        )


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
            ' ⚠️ Column "pell_status_first_year" not found; skipping Pell status NA replacement.'
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
            ' ⚠️ Column "first_gen" not found; skipping first-gen NA replacement.'
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
            LOGGER.warning(' ⚠️ Column "%s" not found', col)
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
            # TODO: check if this works; capture rows about to be renumbered
            dup_info_before = df_course.loc[
                to_renumber, ["course_prefix", "course_name", "course_number"]
            ]
            LOGGER.info(f"Renumbering these duplicates (before):\n{dup_info_before}")

            df_course = dedupe_by_renumbering_courses(df_course)

            # TODO: check if this works; log the same rows again to see the updated course numbers
            dup_info_after = df_course.loc[
                to_renumber, ["course_prefix", "course_name", "course_number"]
            ]
            LOGGER.info(f"Renumbering these duplicates (after):\n{dup_info_after}")

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
            " ⚠️ %s (<0.1%% of data) true duplicate rows found & dropped",
            len(dupe_rows) // 2,  # integer count of dropped pairs
        )
    else:
        LOGGER.warning(
            "  ⚠️ %s (%.1f%% of data) true duplicate rows found & dropped",
            len(dupe_rows) // 2,
            pct_dup,
        )

    df_course = df_course.drop_duplicates(subset=unique_cols, keep="first").sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )

    return df_course
