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


def _infer_student_id_col(df: pd.DataFrame) -> str:
    """Infer the student ID column name from available columns."""
    if "student_guid" in df.columns:
        return "student_guid"
    elif "study_id" in df.columns:
        return "study_id"
    else:
        return "student_id"


def _is_lab_lecture_combo(s: pd.Series) -> bool:
    """Check if a series contains both Lab and Lecture course types (case-insensitive)."""
    LAB_LABELS = {"lab"}
    LEC_LABELS = {"lecture"}
    types = set(s.dropna().astype(str).str.lower())
    return bool(types & LAB_LABELS) and bool(types & LEC_LABELS)


def _find_pdp_rows_to_renumber(
    df: pd.DataFrame, dup_mask: pd.Series, unique_cols: list[str]
) -> list[int]:
    """Identify PDP duplicate rows that need renumbering (different course_name)."""
    to_renumber = []
    for _, idx in df.loc[dup_mask].groupby(unique_cols, dropna=False).groups.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue
        names = df.loc[idx, "course_name"]
        if names.nunique(dropna=False) > 1:
            to_renumber.extend(idx)
    return to_renumber


def _log_pdp_duplicate_drop(df: pd.DataFrame, dup_mask: pd.Series) -> None:
    """Log information about true duplicate rows being dropped in PDP mode."""
    dupe_rows = df.loc[dup_mask, :]
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


def _handle_pdp_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Handle duplicates for PDP mode."""
    student_id_col = _infer_student_id_col(df)
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
        to_renumber = _find_pdp_rows_to_renumber(df, dup_mask, unique_cols)

        if to_renumber:
            dup_info_before = df.loc[
                to_renumber, ["course_prefix", "course_name", "course_number"]
            ]
            LOGGER.info(f"Renumbering these duplicates (before):\n{dup_info_before}")

            df = dedupe_by_renumbering_courses(df)

            dup_info_after = df.loc[
                to_renumber, ["course_prefix", "course_name", "course_number"]
            ]
            LOGGER.info(f"Renumbering these duplicates (after):\n{dup_info_after}")

            return df

    # true duplicates -> drop (original behavior)
    _log_pdp_duplicate_drop(df, dup_mask)

    df = df.drop_duplicates(subset=unique_cols, keep="first").sort_values(
        by=unique_cols + ["number_of_credits_attempted"],
        ascending=False,
        ignore_index=True,
    )
    return df


def _log_duplicate_groups(
    duplicate_rows: pd.DataFrame,
    unique_cols: list[str] = [
        "student_id",
        "academic_term",
        "course_prefix",
        "course_number",
    ],
    course_type_col: str | None = "course_classification",
    course_name_col: str | None = "course_name",
) -> None:
    """Log detailed breakdown of duplicate course groups."""
    LOGGER.info("Duplicate Course Groups (course_type / course_name breakdown)")
    if len(duplicate_rows) == 0:
        LOGGER.info("No duplicate course groups remain.")
        return

    for key_vals, group in duplicate_rows.groupby(
        unique_cols, observed=True, dropna=False
    ):
        sid, term, subj, num = key_vals
        parts = []
        if course_type_col is not None:
            type_counts = group[course_type_col].fillna("UNKNOWN").value_counts()
            parts.append(
                "type: " + ", ".join(f"{c}×{t}" for t, c in type_counts.items())
            )
        if course_name_col is not None:
            name_counts = group[course_name_col].fillna("UNKNOWN").value_counts()
            parts.append(
                "name: " + ", ".join(f"{c}×{n}" for n, c in name_counts.items())
            )
        extra = (" | " + " | ".join(parts)) if parts else ""
        LOGGER.info(f"  {sid} {term} {subj} {num}{extra}")


def _classify_duplicate_groups(
    duplicate_rows: pd.DataFrame,
    unique_cols: list[str] = [
        "student_id",
        "academic_term",
        "course_prefix",
        "course_number",
    ],
    course_type_col: str | None = "course_classification",
    course_name_col: str | None = "course_name",
    credits_col: str | None = "course_credits_attempted",
    grade_col: str | None = "grade",
) -> tuple[list[int], list[int], int, int, int]:
    """
    Classify duplicate groups into renumber vs drop categories.

    Returns:
        Tuple of (renumber_idx, drop_idx, renumber_groups, drop_groups, lab_lecture_rows)
    """
    renumber_groups = 0
    drop_groups = 0
    renumber_work_idx = []
    drop_idx = []
    lab_lecture_rows = 0

    for _, grp in duplicate_rows.groupby(unique_cols, observed=True, dropna=False):
        type_varies = (
            grp[course_type_col].nunique(dropna=False) > 1
            if course_type_col is not None
            else False
        )
        name_varies = (
            grp[course_name_col].nunique(dropna=False) > 1
            if course_name_col is not None
            else False
        )
        must_renumber = type_varies or name_varies

        if must_renumber:
            renumber_groups += 1
            renumber_work_idx.extend(list(grp.index))
            if course_type_col is not None and _is_lab_lecture_combo(
                grp[course_type_col]
            ):
                lab_lecture_rows += len(grp)
        else:
            drop_groups += 1

            grp_sorted = grp

            # Build sort keys (descending)
            sort_cols: list[str] = []
            ascending: list[bool] = []

            if credits_col is not None and credits_col in grp_sorted.columns:
                sort_cols.append(credits_col)
                ascending.append(False)

            # Grade as a numeric score (descending)
            if grade_col is not None and grade_col in grp_sorted.columns:
                score_col = "__grade_score__"
                grp_sorted = grp_sorted.assign(
                    **{score_col: grp_sorted[grade_col].map(_grade_to_score)}
                )
                sort_cols.append(score_col)
                ascending.append(False)

            if sort_cols:
                grp_sorted = grp_sorted.sort_values(
                    by=sort_cols,
                    ascending=ascending,
                    kind="mergesort",  # stable: preserves original order on ties
                )

            keep_one = grp_sorted.index[0]
            drop_idx.extend([i for i in grp_sorted.index if i != keep_one])

    return (
        renumber_work_idx,
        drop_idx,
        renumber_groups,
        drop_groups,
        lab_lecture_rows,
    )


def _drop_true_duplicate_rows(df: pd.DataFrame, drop_idx: list[int]) -> pd.DataFrame:
    """Drop true duplicate rows and log the operation."""
    dropped_rows = len(drop_idx)
    if dropped_rows > 0:
        pct_dropped = (dropped_rows / len(df)) * 100 if len(df) else 0.0
        if pct_dropped < 0.1:
            LOGGER.warning(
                "⚠️ Dropping %s rows (<0.1%% of data) from duplicate-key groups (keeping best row per key)",
                dropped_rows,
            )
        else:
            LOGGER.warning(
                "⚠️ Dropping %s rows (%.2f%% of data) from duplicate-key groups (keeping best row per key)",
                dropped_rows,
                pct_dropped,
            )
        df = df.drop(index=drop_idx)
    return df


def _renumber_duplicates(
    df: pd.DataFrame,
    renumber_work_idx: list[int],
    unique_cols: list[str] | None = None,
    credits_col: str | None = "course_credits_attempted",
    course_type_col: str | None = "course_classification",
    course_name_col: str | None = "course_name",
) -> pd.DataFrame:
    """Renumber duplicate courses for schema mode."""
    if unique_cols is None:
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]

    if not renumber_work_idx:
        return df

    renumber_work_idx = [i for i in renumber_work_idx if i in df.index]
    if not renumber_work_idx:
        return df

    cols_to_show = ["course_prefix", "course_number"]
    if course_type_col is not None:
        cols_to_show.append(course_type_col)
    if course_name_col is not None:
        cols_to_show.append(course_name_col)
    if credits_col is not None:
        cols_to_show.append(credits_col)

    LOGGER.info(
        "Renumbering duplicates (before) [showing up to 50 rows]:\n%s",
        df.loc[renumber_work_idx, cols_to_show]
        .sort_values(["course_prefix", "course_number"], kind="mergesort")
        .head(50),
    )

    # Work only on rows we intend to renumber
    work = df.loc[renumber_work_idx].copy()

    # Optional: if you want credits to influence -1/-2 ordering and schema credits
    # aren't already in number_of_credits_attempted
    if credits_col is not None and "number_of_credits_attempted" not in work.columns:
        work["number_of_credits_attempted"] = work[credits_col]

    work = dedupe_by_renumbering_courses(
        work,
        unique_cols=unique_cols,
    )

    # Update only affected rows
    df.loc[renumber_work_idx, "course_number"] = work["course_number"].astype("string")

    LOGGER.info(
        "Renumbering duplicates (after) [showing up to 50 rows]:\n%s",
        df.loc[renumber_work_idx, cols_to_show]
        .sort_values(["course_prefix", "course_number"], kind="mergesort")
        .head(50),
    )

    return df


def _log_schema_summary(
    total_before: int,
    initial_dup_rows: int,
    initial_dup_pct: float,
    exact_dupes_dropped: int,
    keeper_dropped_rows: int,
    renumbered_rows: int,
    lab_lecture_rows: int,
    lab_lecture_pct: float,
    renumber_groups: int,
    final_dupe_rows: int,
    total_after: int,
    course_type_col: str | None,
    course_name_col: str | None,
    keeper_rule: str,
) -> None:
    LOGGER.info("COURSE RECORD DUPLICATE SUMMARY (edvise schema)")

    LOGGER.info(
        "Before cleanup: %s records, %s duplicate-key rows (%.2f%%)",
        total_before,
        initial_dup_rows,
        initial_dup_pct,
    )

    total_removed = total_before - total_after
    LOGGER.info(
        "Rows removed: %s total (exact-identical=%s, keeper-drop=%s) | Rows renumbered: %s",
        total_removed,
        exact_dupes_dropped,
        keeper_dropped_rows,
        renumbered_rows,
    )

    LOGGER.info("Keeper rule for drop-groups: %s", keeper_rule)

    if course_type_col is not None:
        LOGGER.info(
            "Lab/lecture duplicates within renumbered rows: %s (%.2f%%)",
            lab_lecture_rows,
            lab_lecture_pct,
        )

    LOGGER.info("Duplicate groups renumbered: %s", renumber_groups)

    LOGGER.info(
        "After cleanup: %s records | Remaining key-duplicates: %s",
        total_after,
        final_dupe_rows,
    )

    LOGGER.info("")


def _handle_schema_duplicates(
    df: pd.DataFrame,
    unique_cols: list[str] | None = None,
    credits_col: str | None = "course_credits_attempted",
    course_type_col: str | None = "course_classification",
    course_name_col: str | None = "course_name",
    grade_col: str | None = "grade",
) -> pd.DataFrame:
    """Handle duplicates for Edvise schema mode."""
    LOGGER.info("handle_duplicates: edvise schema mode triggered")

    # Set defaults for unique_cols
    if unique_cols is None:
        unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]

    # Validate course_type_col
    if course_type_col not in df.columns:
        LOGGER.warning(
            f"Column '{course_type_col}' not found in dataframe. Will skip course_type operations."
        )
        course_type_col = None

    # Validate course_name_col
    if course_name_col not in df.columns:
        LOGGER.warning(
            f"Column '{course_name_col}' not found in dataframe. Will skip course_name operations."
        )
        course_name_col = None

    # Validate credits_col
    if credits_col not in df.columns:
        LOGGER.warning(
            f"Column '{credits_col}' not found in dataframe. Will skip credits-based operations."
        )
        credits_col = None

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
    # Classify duplicates: renumber vs drop
    (
        renumber_work_idx,
        drop_idx,
        renumber_groups,
        drop_groups,
        lab_lecture_rows,
    ) = _classify_duplicate_groups(
        duplicate_rows,
        unique_cols,
        course_type_col,
        course_name_col,
        credits_col,
        grade_col,
    )

    # Drop rows from duplicate-key groups (keeper logic)
    dropped_rows = len(drop_idx)
    df = _drop_true_duplicate_rows(df, drop_idx)

    # Renumber duplicates
    df = _renumber_duplicates(
        df,
        renumber_work_idx,
        unique_cols,
        credits_col,
        course_type_col,
        course_name_col,
    )

    # Build course_id (always in schema mode)
    df["course_id"] = (
        df["course_prefix"].astype("string").str.strip()
        + df["course_number"].astype("string").str.strip()
    )

    # Calculate summary statistics
    total_after = len(df)
    final_dupe_rows = int(df.duplicated(unique_cols, keep=False).sum())
    renumbered_rows = len(set(renumber_work_idx)) if renumber_work_idx else 0
    lab_lecture_pct = (
        (lab_lecture_rows / renumbered_rows * 100) if renumbered_rows else 0.0
    )

    keeper_rule = (
        "keep highest credits; if tied, keep highest grade; if tied, keep first row"
    )

    # Log summary (NOW everything exists)
    _log_schema_summary(
        total_before,
        initial_dup_rows,
        initial_dup_pct,
        true_dupes_dropped,  # exact-identical
        dropped_rows,  # keeper-drop
        renumbered_rows,
        lab_lecture_rows,
        lab_lecture_pct,
        renumber_groups,
        final_dupe_rows,
        total_after,
        course_type_col,
        course_name_col,
        keeper_rule,
    )

    return df


def handling_duplicates(
    df: pd.DataFrame,
    school_type: str,
    unique_cols: list[str] | None = [
        "student_id",
        "academic_term",
        "course_prefix",
        "course_number",
    ],
    credits_col: str | None = "course_credits_attempted",
    course_type_col: str | None = "course_classification",
    course_name_col: str | None = "course_name",
) -> pd.DataFrame:
    """
    Combined duplicate handling with a school_type switch.

    PDP mode: keep logic as close as possible to original `handling_duplicates`:
      - infer student_id_col
      - unique_cols = [student_id_col, "academic_year", "academic_term",
                      "course_prefix", "course_number", "section_id"]
      - if duplicate-key rows have DIFFERENT course_name -> renumber via dedupe_by_renumbering_courses()
      - else -> drop true dupes with warning logging similar to original

    Edvise schema mode: based on `handle_duplicates`, but with "handling_duplicates"-style logic:
      - unique_cols = ["student_id", "academic_term", "course_prefix", "course_number"]
      - drop exact duplicates first
      - for remaining key-dupes:
          * if differ by course_type OR course_name -> KEEP ALL and renumber course_number
            like original handle_duplicates: FIRST stays unchanged, others get -01, -02, ...
          * else -> drop true dupes keeping max credits (if available)
      - build course_id = course_prefix + course_number
      - include handle_duplicates-style summary logging + breakdown
    """
    df = df.copy()
    school_type = (school_type or "").strip().lower()
    if school_type not in {"pdp", "schema"}:
        raise ValueError(
            "school_type must be either 'pdp' or 'schema', short for edvise schema."
        )

    if school_type == "pdp":
        return _handle_pdp_duplicates(df)
    else:
        return _handle_schema_duplicates(
            df, unique_cols, credits_col, course_type_col, course_name_col
        )


def _grade_to_score(val: object) -> float:
    """
    Convert a grade value to a numeric score for sorting.
    Handles numeric grades (e.g., 92, "3.7") and letter grades (A, A-, B+ ...).
    Unknown/blank -> -inf so it loses ties.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return float("-inf")

    s = str(val).strip().upper()
    if not s:
        return float("-inf")

    # Numeric grade?
    try:
        return float(s)
    except ValueError:
        pass

    # Letter grades
    # You can tune this mapping to your schema.
    base = {
        "A": 4.0,
        "B": 3.0,
        "C": 2.0,
        "D": 1.0,
        "F": 0.0,
    }

    m = re.match(r"^([A-F])([+-])?$", s)
    if not m:
        return float("-inf")

    letter, sign = m.group(1), m.group(2)
    score = base[letter]
    if sign == "+":
        score += 0.3
    elif sign == "-":
        score -= 0.3
    return score
