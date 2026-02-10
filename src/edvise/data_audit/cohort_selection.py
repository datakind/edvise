import logging

import pandas as pd


def select_inference_cohort(
    df: pd.DataFrame,
    cohorts_list: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort",
) -> pd.DataFrame:
    """
    Selects the specified cohorts from DataFrames.

    Args:
        df: The DataFrame.
        cohorts_list: List of cohorts to select (e.g., ["fall 2023-24", "spring 2024-25"]).
        cohort_term_column: Column name for cohort term (e.g. FALL, SPRING). Default "cohort_term".
        cohort_column: Column name for cohort year/label. Default "cohort".

    Returns:
        The filtered DataFrame.

    Raises:
        ValueError: If cohorts_list has no non-empty labels, or if filtering results in empty DataFrames.
    """

    # Normalize cohort labels to lowercase so matching is case-insensitive (data column is built lowercased).
    cohorts_list_normalized = [
        label.strip().lower() for label in cohorts_list if label and str(label).strip()
    ]
    if not cohorts_list_normalized:
        raise ValueError("cohorts_list had no non-empty cohort labels.")

    # We only have cohort and cohort term split up, so combine and strip to lower to prevent cap issues
    df["cohort_selection"] = (
        df[cohort_term_column].astype(str).str.lower()
        + " "
        + df[cohort_column].astype(str).str.lower()
    )

    # Subset both datsets to only these cohorts
    df_filtered = df[df["cohort_selection"].isin(cohorts_list_normalized)].copy()

    logging.info(
        "Selected cohorts for inference: %s\nCohort counts in filtered data:\n%s",
        cohorts_list,
        df_filtered["cohort_selection"].value_counts().to_string(),
    )

    # Throw error if either dataset is empty after filtering
    if df_filtered.empty:
        logging.error(
            "Selected cohorts resulted in empty DataFrames; requested cohorts_list=%s",
            cohorts_list,
        )
        raise ValueError(
            f"Selected cohorts resulted in empty DataFrames; requested cohorts_list={cohorts_list!r}."
        )

    df_filtered.drop(columns="cohort_selection", inplace=True)

    return df_filtered
