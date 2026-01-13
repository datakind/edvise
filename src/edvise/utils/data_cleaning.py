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
                    parse_year(s.split()[-1]),  # handle '2022-23'
                    TERM_ORDER.get(s.split()[0], 99),  # order terms
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
                " remove_pre_cohort_courses: %d pre-cohort course records safely removed (<0.1 percent of data).",
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
    else:
        LOGGER.info("remove_pre_cohort_courses: No pre-cohort course records found.")

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


import pandas as pd


def handling_duplicates(df: pd.DataFrame, school_type: str) -> pd.DataFrame:
    """
    Combined duplicate handling with a school_type switch.

    PDP mode: keep logic as close as possible to original `handling_duplicates`:
      - infer student_id_col
      - unique_cols = [student_id_col, "academic_year", "academic_term",
                      "course_prefix", "course_number", "section_id"]
      - if duplicate-key rows have DIFFERENT course_name -> renumber via dedupe_by_renumbering_courses()
      - else -> drop true dupes with warning logging similar to original

    Edvise schema mode: based on `handle_duplicates`, but with "handling_duplicates"-style logic:
      - unique_cols = ["student_id", "term", "course_subject", "course_num"]
      - drop exact duplicates first
      - for remaining key-dupes:
          * if differ by course_type OR course_name -> KEEP ALL and renumber course_num
            like original handle_duplicates: FIRST stays unchanged, others get -01, -02, ...
          * else -> drop true dupes keeping max credits (if available)
      - build course_id = course_subject + course_num
      - include handle_duplicates-style summary logging + breakdown
    """

    df = df.copy()
    school_type = (school_type or "").strip().lower()
    if school_type not in {"pdp", "schema"}:
        raise ValueError("school_type must be either 'pdp' or 'schema', short for edvise schema.")

    # ---------------------------------------------------------------------
    # PDP MODE
    # ---------------------------------------------------------------------
    if school_type == "pdp":
        student_id_col = (
            "student_guid"
            if "student_guid" in df.columns
            else "study_id"
            if "study_id" in df.columns
            else "student_id"
        )
        LOGGER.info("handle_duplicates: PDP mode triggered")
        unique_cols = [
            student_id_col,
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]

        dup_mask = df.duplicated(unique_cols, keep=False)

        if dup_mask.any() and "course_name" in df.columns:
            to_renumber = []
            for _, idx in (
                df.loc[dup_mask].groupby(unique_cols, dropna=False).groups.items()
            ):
                idx = list(idx)
                if len(idx) <= 1:
                    continue
                names = df.loc[idx, "course_name"]
                if names.nunique(dropna=False) > 1:
                    to_renumber.extend(idx)

            if to_renumber:
                dup_info_before = df.loc[
                    to_renumber, ["course_prefix", "course_name", "course_number"]
                ]
                LOGGER.info(
                    f"Renumbering these duplicates (before):\n{dup_info_before}"
                )

                df = dedupe_by_renumbering_courses(df)

                dup_info_after = df.loc[
                    to_renumber, ["course_prefix", "course_name", "course_number"]
                ]
                LOGGER.info(f"Renumbering these duplicates (after):\n{dup_info_after}")

                return df

        # true duplicates -> drop (original behavior)
        dupe_rows = df.loc[dup_mask, :].sort_values(
            by=unique_cols + ["number_of_credits_attempted"],
            ascending=False,
            ignore_index=True,
        )
        pct_dup = (len(dupe_rows) / len(df)) * 100 if len(df) else 0.0
        if pct_dup < 0.1:
            LOGGER.warning(
                " ⚠️ %s (<0.1 percent of data) true duplicate rows found & dropped",
                len(dupe_rows) // 2,
            )
        else:
            LOGGER.warning(
                "  ⚠️ %s (%.1f%% of data) true duplicate rows found & dropped",
                len(dupe_rows) // 2,
                pct_dup,
            )

        df = df.drop_duplicates(subset=unique_cols, keep="first").sort_values(
            by=unique_cols + ["number_of_credits_attempted"],
            ascending=False,
            ignore_index=True,
        )
        return df

    # ---------------------------------------------------------------------
    # EDVISE SCHEMA MODE
    # ---------------------------------------------------------------------
    LOGGER.info("handle_duplicates: edvise schema mode triggered")
    unique_cols = ["student_id", "term", "course_subject", "course_num"]
    total_before = len(df)

    # Key-based duplicates BEFORE removing exact dupes
    initial_dupes_mask = df.duplicated(unique_cols, keep=False)
    initial_dup_rows = int(initial_dupes_mask.sum())
    initial_dup_pct = (initial_dup_rows / total_before * 100) if total_before else 0.0

    # Drop exact duplicates (fully identical rows)
    before_drop = len(df)
    df = df.drop_duplicates(keep="first")
    after_drop = len(df)
    true_dupes_dropped = before_drop - after_drop

    # Remaining duplicates (key-based)
    dupes_mask = df.duplicated(unique_cols, keep=False)
    duplicate_rows = df.loc[dupes_mask]
    records_in_dupe_groups = len(duplicate_rows)

    has_course_type = "course_type" in df.columns
    has_course_name = "course_name" in df.columns

    # Credits column
    if "course_credits" in df.columns:
        credits_col = "course_credits"
    elif "number_of_credits_attempted" in df.columns:
        credits_col = "number_of_credits_attempted"
    else:
        credits_col = None

    # Lab/Lecture combo stats among renumbered groups
    LAB_LABELS = {"Lab"}
    LEC_LABELS = {"Lecture"}

    def is_lab_lecture_combo(s: pd.Series) -> bool:
        types = set(s.dropna())
        return bool(types & LAB_LABELS) and bool(types & LEC_LABELS)

    # Group breakdown logging (handle_duplicates-style)
    LOGGER.info("Duplicate Course Groups (course_type / course_name breakdown)")
    if records_in_dupe_groups == 0:
        LOGGER.info("No duplicate course groups remain.")
    else:
        for key_vals, group in duplicate_rows.groupby(
            unique_cols, observed=True, dropna=False
        ):
            sid, term, subj, num = key_vals
            parts = []
            if has_course_type:
                type_counts = group["course_type"].fillna("UNKNOWN").value_counts()
                parts.append(
                    "type: " + ", ".join(f"{c}×{t}" for t, c in type_counts.items())
                )
            if has_course_name:
                name_counts = group["course_name"].fillna("UNKNOWN").value_counts()
                parts.append(
                    "name: " + ", ".join(f"{c}×{n}" for n, c in name_counts.items())
                )
            extra = (" | " + " | ".join(parts)) if parts else ""
            LOGGER.info(f"  {sid} {term} {subj} {num}{extra}")

    # Decide: renumber vs drop
    renumber_groups = 0
    drop_groups = 0
    renumber_work_idx = []  # rows we will renumber
    drop_idx = []  # rows we will drop
    lab_lecture_rows = 0

    if records_in_dupe_groups > 0:
        for _, grp in duplicate_rows.groupby(unique_cols, observed=True, dropna=False):
            type_varies = (
                has_course_type and grp["course_type"].nunique(dropna=False) > 1
            )
            name_varies = (
                has_course_name and grp["course_name"].nunique(dropna=False) > 1
            )
            must_renumber = type_varies or name_varies

            if must_renumber:
                renumber_groups += 1
                renumber_work_idx.extend(list(grp.index))
                if has_course_type and is_lab_lecture_combo(grp["course_type"]):
                    lab_lecture_rows += len(grp)
            else:
                drop_groups += 1
                # Keep best row (highest credits if possible), drop the rest
                if credits_col is not None:
                    grp_sorted = grp.sort_values(
                        by=[credits_col], ascending=False, kind="mergesort"
                    )
                else:
                    grp_sorted = grp
                keep_one = grp_sorted.index[0]
                drop_idx.extend([i for i in grp_sorted.index if i != keep_one])

    # Drop true duplicates rows
    dropped_rows = len(drop_idx)
    if dropped_rows > 0:
        pct_dropped = (dropped_rows / len(df)) * 100 if len(df) else 0.0
        if pct_dropped < 0.1:
            LOGGER.warning(
                "⚠️ Dropping %s rows (<0.1%% of data) from true-duplicate groups (keeping best row per key)",
                dropped_rows,
            )
        else:
            LOGGER.warning(
                "⚠️ Dropping %s rows (%.2f%% of data) from true-duplicate groups (keeping best row per key)",
                dropped_rows,
                pct_dropped,
            )
        df = df.drop(index=drop_idx)

    # Renumber duplicates where course_type OR course_name differs
    if renumber_work_idx:
        renumber_work_idx = [i for i in renumber_work_idx if i in df.index]
        if renumber_work_idx:
            # Log before snapshot (cap to avoid huge logs)
            cols_to_show = ["course_subject", "course_num"]
            if has_course_type:
                cols_to_show.append("course_type")
            if has_course_name:
                cols_to_show.append("course_name")
            if credits_col is not None:
                cols_to_show.append(credits_col)

            LOGGER.info(
                "Renumbering duplicates (before) [showing up to 50 rows]:\n%s",
                df.loc[renumber_work_idx, cols_to_show]
                .sort_values(["course_subject", "course_num"], kind="mergesort")
                .head(50),
            )

            # Ensure course_num can hold strings
            df["course_num"] = df["course_num"].astype("string")

            work = df.loc[renumber_work_idx].copy()

            # Deterministic ordering (like handle_duplicates uses credits desc)
            sort_cols = unique_cols.copy()
            if credits_col is not None:
                sort_cols += [credits_col]
                ascending = [True] * len(unique_cols) + [False]
            else:
                ascending = [True] * len(unique_cols)

            work = work.sort_values(by=sort_cols, ascending=ascending, kind="mergesort")

            # IMPORTANT: mimic original handle_duplicates:
            # - first record in each dup group keeps original course_num
            # - subsequent records get -01, -02, ...
            dup_index = work.groupby(
                unique_cols, observed=True, dropna=False
            ).cumcount()
            suffix_index = dup_index.where(dup_index > 0)  # NaN/0 for first

            new_vals = work["course_num"].astype("string")
            mask = suffix_index.notna()
            # For dup_index==1 -> -01, dup_index==2 -> -02, ...
            new_vals.loc[mask] = (
                new_vals.loc[mask]
                + "-"
                + suffix_index.loc[mask].astype("int").map(lambda x: f"{x:02d}")
            )

            df.loc[work.index, "course_num"] = new_vals.astype("string")

            LOGGER.info(
                "Renumbering duplicates (after) [showing up to 50 rows]:\n%s",
                df.loc[work.index, cols_to_show]
                .sort_values(["course_subject", "course_num"], kind="mergesort")
                .head(50),
            )

    # Build course_id (edvise schema only)
    df["course_id"] = (
        df["course_subject"].astype("string").str.strip()
        + df["course_num"].astype("string").str.strip()
    )

    # Summary (handle_duplicates-style)
    final_dupe_rows = int(df.duplicated(unique_cols, keep=False).sum())
    renumbered_rows = len(set(renumber_work_idx)) if renumber_work_idx else 0
    lab_lecture_pct = (
        (lab_lecture_rows / renumbered_rows * 100) if renumbered_rows else 0.0
    )

    LOGGER.info("COURSE RECORD DUPLICATE SUMMARY (edvise schema)")
    LOGGER.info(f"Total course records before:      {total_before}")
    LOGGER.info(
        f"Duplicate records found:          {initial_dup_rows} ({initial_dup_pct:.2f}%)"
    )
    LOGGER.info(f"True duplicates dropped:          {true_dupes_dropped}")
    LOGGER.info(f"Records renumbered:               {renumbered_rows}")
    if has_course_type:
        LOGGER.info(
            f"Lab/lecture duplicate rows:       {lab_lecture_rows} ({lab_lecture_pct:.2f}%)"
        )
    LOGGER.info(f"Duplicate groups renumbered:      {renumber_groups}")
    LOGGER.info(f"Duplicate groups dropped:         {drop_groups}")
    LOGGER.info(f"Rows dropped from true dupes:     {dropped_rows}")
    LOGGER.info(f"Total course records after:       {len(df)}")
    LOGGER.info(f"Remaining key-duplicate rows:     {final_dupe_rows}")
    LOGGER.info("")

    return df
