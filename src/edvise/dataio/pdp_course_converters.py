import logging

import pandas as pd
import re

LOGGER = logging.getLogger(__name__)


def dedupe_by_renumbering_courses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate rows in raw course data ``df`` by renumbering courses so that
    the combination of student/year/term/prefix/section/course_number is unique.

    Rules
    -----
    • Only renumber when duplicates are due to course-number collisions.
    • If a course_number already has a suffix like 291-1 it is left untouched.
    • When extra plain numbers exist (e.g. multiple '101's), keep the highest-
      credit row as plain and suffix the rest: 101-1, 101-2, … starting from the
      highest suffix already present.

    Warning
    -------
    Do not use if the data contains true duplicate rows (same everything).
    """
    if df.empty:
        return df

    # ---- identify the student id column ----
    student_id_col = (
        "student_guid" if "student_guid" in df.columns
        else "study_id"   if "study_id"   in df.columns
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

    # work only on rows that are true duplicates on these keys
    dupes = df.loc[df.duplicated(unique_cols, keep=False)].copy()
    if dupes.empty:
        return df

    def renumber_group(grp: pd.DataFrame) -> pd.Series:
        """
        Within one duplicate group:
          • leave already-suffixed numbers untouched,
          • give extra plain numbers new suffixes without re-using any.
        """
        suffix_re = re.compile(r"^(.*?)-(\d+)$")

        # 1️⃣ find current highest suffix for each base
        base_max = {}
        for cn in grp["course_number"]:
            m = suffix_re.match(cn)
            if m:
                base, suf = m.group(1), int(m.group(2))
            else:
                base, suf = cn, 0
            base_max[base] = max(base_max.get(base, 0), suf)

        # 2️⃣ rows whose course_number has NO suffix
        is_plain = ~grp["course_number"].str.contains(r"-\d+$", regex=True)

        # 3️⃣ copy column and renumber extra plain rows
        new_numbers = grp["course_number"].copy()

        # group only the plain rows by their base value
        for base, sub in grp[is_plain].groupby(
            grp.loc[is_plain, "course_number"]
        ):
            if len(sub) <= 1:
                continue  # only one plain row → nothing to renumber

            # keep the highest-credit plain row unsuffixed
            sub_sorted = sub.sort_values(
                "number_of_credits_attempted",
                ascending=False,
                na_position="last",
            )
            keep_idx = sub_sorted.index[0]
            rest = sub_sorted.index[1:]

            next_suffix = base_max[base] + 1
            for ix in rest:
                new_numbers.loc[ix] = f"{base}-{next_suffix}"
                next_suffix += 1

        return new_numbers

    # ---- apply to each duplicate group (excluding course_number in the key) ----
    dupes["course_number"] = dupes.groupby(
        [c for c in unique_cols if c != "course_number"],
        group_keys=False
    ).apply(renumber_group)

    LOGGER.warning(
        "%s duplicate course records found; course numbers modified to avoid duplicates",
        len(dupes),
    )

    # ---- update the original DataFrame in place ----
    df = df.copy()
    df.update(dupes[["course_number"]], overwrite=True)
    return df
