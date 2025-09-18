import logging

import pandas as pd
import re

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
        Renumber duplicate course_numbers within this group,
        appending -1, -2… while respecting any existing suffixes.
        """
        suffix_re = re.compile(r"^(.*?)-(\d+)$")

        # find current max suffix for each base
        base_max = {}
        for cn in grp["course_number"]:
            m = suffix_re.match(cn)
            if m:
                base, suf = m.group(1), int(m.group(2))
            else:
                base, suf = cn, 0
            base_max[base] = max(base_max.get(base, 0), suf)

        # Sort for “primary” selection but keep original order reference
        grp_sorted = grp.sort_values("number_of_credits_attempted", ascending=False)

        seen = {base: set() for base in base_max}
        new_numbers_by_index = {}

        for idx, row in grp_sorted.iterrows():
            cn = row["course_number"]
            m = suffix_re.match(cn)
            base = m.group(1) if m else cn

            # If this *exact* number hasn’t been used yet for this base, keep it
            if cn not in seen[base]:
                new_numbers_by_index[idx] = cn
            else:
                base_max[base] += 1
                new_numbers_by_index[idx] = f"{base}-{base_max[base]}"
            seen[base].add(new_numbers_by_index[idx])

        # now return in the original order, using the mapping we built
        return pd.Series(new_numbers_by_index).loc[grp.index]

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
