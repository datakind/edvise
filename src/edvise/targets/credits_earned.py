import typing as t

import numpy as np
import pandas as pd

from .. import utils
from .. import checkpoints
from . import shared


def compute_target(
    df: pd.DataFrame,
    *,
    min_num_credits: float,
    checkpoint: pd.DataFrame | t.Callable[[pd.DataFrame], pd.DataFrame],
    intensity_time_limits: utils.types.IntensityTimeLimitsType,
    num_terms_in_year: int = 4,
    max_term_rank: int | t.Literal["infer"] = "infer",
    student_id_cols: str | list[str] = "student_id",
    enrollment_intensity_col: str = "student_term_enrollment_intensity",
    num_credits_col: str = "cumsum_num_credits_earned",
    term_rank_col: str = "term_rank",
) -> pd.Series:
    """
    Compute *insufficient* credits earned target for each distinct student in ``df`` ,
    for which intensity-specific time limits determine if credits earned is "on-time".

    Target = True: Student did NOT earn min_num_credits within the allowed time → at-risk student
    Target = False: Student DID earn enough credits on time → successful student
    Target = NA: Not enough data to determine (student's "on-time window" extends beyond available data)

    Args:
        df: Student-term dataset.
        min_num_credits: Minimum number of credits earned within specified time limits
            to be considered a *success* => target=False.
        checkpoint: "Checkpoint" from which time limits to target term are determined,
            typically either the first enrolled term or the first term above an intermediate
            number of credits earned; may be given as a data frame with one row per student,
            or as a callable that takes ``df`` as input and returns all checkpoint terms.
        intensity_time_limits: Mapping of enrollment intensity value (e.g. "FULL-TIME")
            to the maximum number of years or terms considered to be "on-time" for
            the target number of credits earned (e.g. [4.0, "year"], [12.0, "term"]),
            where the numeric values are for the time between "checkpoint" and "target"
            terms. Passing special "*" as the only key applies the corresponding time limits
            to all students, regardless of intensity.
        num_terms_in_year: Number of academic terms in one academic year,
            used to convert from year-based time limits to term-based time limits;
            default value assumes FALL, WINTER, SPRING, and SUMMER terms.
        max_term_rank: Maximum term rank value in the full dataset ``df`` , either inferred
            from ``df[term_rank_col]`` itself or as a manually specified value which
            may be different from the actual max value in ``df`` , depending on use case.
        student_id_cols: Columns that uniquely identify students, used for grouping rows.
        enrollment_intensity_col: Column whose values give students' "enrollment intensity"
            (usually either "FULL-TIME" or "PART-TIME"), for which the most common
            value per student is used when comparing against intensity-specific time limits.
        num_credits_col
        term_rank_col: Column whose values give the absolute integer ranking of a given
            term within the full dataset ``df`` .
    """
    # 1. :
    # Get unique students
    student_id_cols = utils.types.to_list(student_id_cols)
    df_distinct_students = df[student_id_cols].drop_duplicates(ignore_index=True)

    # 2. Find checkpoint term
    # Determine "checkpoint" (starting point) term - usually first enrollment or a milestone like earning 30 credits
    df_ckpt = (
        checkpoint.copy(deep=True)
        if isinstance(checkpoint, pd.DataFrame)
        else checkpoint(df)
    )
    if df_ckpt.groupby(by=student_id_cols).size().gt(1).any():
        raise ValueError("checkpoint df must include exactly 1 row per student")

    # 3. Find target term
    # For each student, find the first term they reached min_num_credits (e.g., 120 credits for a degree)
    df_tgt = checkpoints.nth_student_terms.first_student_terms_at_num_credits_earned(
        df,
        min_num_credits=min_num_credits,
        student_id_cols=student_id_cols,
        sort_cols=term_rank_col,
        num_credits_col=num_credits_col,
        include_cols=[enrollment_intensity_col],
    )

    # 4. Combine checkpoint and target data to be able to calculate time elapsed
    df_at = pd.merge(
        df_ckpt,
        df_tgt,
        on=student_id_cols,
        how="left",
        suffixes=("_ckpt", "_tgt"),
    )

    # 5. Convert from year limits to term limits for time limits provided in years
    intensity_num_terms = utils.data_cleaning.convert_intensity_time_limits(
        "term", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )

    # 6. Apply intensity-specific time limits and select at-risk student IDs
    # Check if student took more terms than allowed (e.g., full-time students get 12 terms, part-time get 16)
    # Or if they never reached the credit threshold (target is na)
    tr_col = term_rank_col  
    targets = [
        (
            (
                df_at[f"{enrollment_intensity_col}_ckpt"].eq(intensity)
                | (intensity == "*")
            )
            & (

                (df_at[f"{tr_col}_tgt"] - df_at[f"{tr_col}_ckpt"]).gt(num_terms)
                | df_at[f"{tr_col}_tgt"].isna()
            )
        )
        for intensity, num_terms in intensity_num_terms.items()
    ]
    target = np.logical_or.reduce(targets)


    # create a dataframe of just the at-risk students and add a column 'target' that is true boolean value
    df_target_true = (
        df_at.loc[target, student_id_cols]  
        .assign(target=True)        
        .astype({"target": "boolean"})      
    )

    # 7. Eligibility:
    intensity_num_terms_minus_1 = {
        intensity: max(num_terms - 1, 0)
        for intensity, num_terms in intensity_num_terms.items()
    }
    intensity_time_limits_for_eligibility = t.cast(
        utils.types.IntensityTimeLimitsType,
        {
            intensity: (float(num_terms), "term")
            for intensity, num_terms in intensity_num_terms_minus_1.items()
        },
    )

    # Filter to labelable students:
    # Only label students whose "deadline term" falls within the available data
    # Prevents mislabeling: if a student's data ends before their deadline, we can't know if they'll succeed
    df_labelable_students = shared.get_students_with_max_target_term_in_dataset(
        df,
        checkpoint=df_ckpt,
        intensity_time_limits=intensity_time_limits_for_eligibility,
        max_term_rank=max_term_rank,
        num_terms_in_year=num_terms_in_year,
        student_id_cols=student_id_cols,
        enrollment_intensity_col=enrollment_intensity_col,
        term_rank_col=term_rank_col,
    )
    # 8. Assign labels:
    # Labelable students who took too long or never graduated → True
    # Labelable students who made it on time → False
    # Students with insufficient data → NA (dropped)
    df_labeled = (
        # match positive labels to label-able students
        pd.merge(df_labelable_students, df_target_true, on=student_id_cols, how="left")
        # assign False to all label-able students not already assigned True
        .fillna({"target": False})
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    df_all_student_targets = (
        # assign null target to all students
        df_distinct_students.assign(target=pd.Series(pd.NA, dtype="boolean"))
        # structure so student-ids as index, target as only column
        .set_index(student_id_cols)
    )
    # update null targets in-place with bool targets on matching student-id indexes
    df_all_student_targets.update(df_labeled)
    # #drop if target is uncalculable (null)
    df_all_student_targets["target"] = (
        df_all_student_targets["target"].astype("boolean").dropna()
    )
    # return as a series with target as values and student ids as index
    return df_all_student_targets.loc[:, "target"].dropna()
