import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)


def dedupe_by_renumbering_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows in raw course data ``df`` by renumbering courses,
    avoiding re-using existing suffixes like 291-1 if they already exist.

    Warning:
        Only use when duplicates are due to course-number collisions,
        not when you have truly duplicate rows.
    """
    # infer student id column
    student_id_col = (
        "student_guid"
        if "student_guid" in df.columns
        else "study_id"
        if "study_id" in df.columns
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

    # subset of rows that are duplicated
    dupes = df.loc[df.duplicated(unique_cols, keep=False), :].copy()

    def renumber_group(grp: pd.DataFrame) -> pd.Series:
        """
        For each group of rows with same unique_cols, append incremental
        suffixes to 'course_number' starting from the max existing suffix + 1.
        """
        # Extract base numbers and any existing numeric suffix
        suffix_re = re.compile(r"^(.*?)-(\d+)$")

        # figure out current max suffix per base
        base_max = {}
        for cn in grp["course_number"]:
            m = suffix_re.match(cn)
            if m:
                base, suf = m.group(1), int(m.group(2))
                base_max[base] = max(base_max.get(base, 0), suf)
            else:
                base_max[cn] = max(base_max.get(cn, 0), 0)

        # Sort group so higher credits come first (the "primary")
        grp_sorted = grp.sort_values("number_of_credits_attempted", ascending=False)

        # Build new numbers only for the duplicates inside this group
        seen = {}
        new_numbers = []
        for _, row in grp_sorted.iterrows():
            cn = row["course_number"]
            m = suffix_re.match(cn)
            base = m.group(1) if m else cn
            if cn not in seen:
                # first occurrence: keep as is
                new_numbers.append(cn)
            else:
                # duplicate: increment from existing max
                base_max[base] += 1
                new_numbers.append(f"{base}-{base_max[base]}")
            seen[cn] = True

        # Return in the same order as grp
        return pd.Series(new_numbers, index=grp_sorted.index).reindex(grp.index)

    # Apply group-wise renumbering
    dupes["course_number"] = dupes.groupby(
        [c for c in unique_cols if c != "course_number"], group_keys=False
    ).apply(renumber_group)

    LOGGER.warning(
        "%s duplicate course records found; course numbers modified to avoid duplicates",
        len(dupes),
    )

    # Update the original dataframe
    df.update(dupes[["course_number"]], overwrite=True)
    return df
