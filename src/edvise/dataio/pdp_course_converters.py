import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def dedupe_by_renumbering_courses(
    df: pd.DataFrame,
    *,
    unique_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Deduplicate rows in raw course data ``df`` by renumbering courses, such that
    the data passes data schema uniqueness requirements.

    Args:
        df: Raw course dataset
        unique_cols: Optional list of columns defining duplicate keys.
            If None, defaults to PDP raw course uniqueness logic.

    Warning:
        This logic assumes that all rows are actually valid, and that the school's
        course numbering is wonky (e.g., lab/lecture sharing course numbers).

        Don't use this function if there are actual duplicate records in the data!
    """
    # HACK: infer the correct student id col in raw data from the data itself
    student_id_col = (
        "student_guid"
        if "student_guid" in df.columns
        else "study_id"
        if "study_id" in df.columns
        else "student_id"
    )

    # Default PDP behavior (unchanged)
    if unique_cols is None:
        unique_cols = [
            student_id_col,
            "academic_year",
            "academic_term",
            "course_prefix",
            "course_number",
            "section_id",
        ]

    # Decide whether we can sort by credits
    has_credits = "number_of_credits_attempted" in df.columns

    sort_cols = unique_cols + (["number_of_credits_attempted"] if has_credits else [])
    ascending = [False] * len(sort_cols)

    deduped_course_numbers = (
        df.loc[df.duplicated(unique_cols, keep=False), :]
        .sort_values(
            by=sort_cols,
            ascending=ascending,
            ignore_index=False,
        )
        .assign(
            grp_num=lambda d: (
                d.groupby(unique_cols)["course_number"].transform("cumcount") + 1
            ),
            course_number=lambda d: d["course_number"].astype("string").str.cat(
                d["grp_num"].astype("string"), sep="-"
            ),
        )
        .loc[:, ["course_number"]]
    )

    LOGGER.warning(
        "%s duplicate course records found (%.1f%%); course numbers modified to avoid duplicates",
        len(deduped_course_numbers),
        (len(deduped_course_numbers) / len(df)) * 100,
    )

    df.update(deduped_course_numbers, overwrite=True)
    return df
