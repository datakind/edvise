import pandas as pd
import logging


def select_inference_cohort(
    df: pd.DataFrame, 
    cohorts_list: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort"
)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects the specified cohorts from DataFrames.

    Args:
        df: The DataFrame.
        cohorts_list: List of cohorts to select (e.g., ["fall 2023-24", "spring 2024-25"]).

    Returns:
        A tuple containing the filtered course and cohort DataFrames.
    
    Raises:
        ValueError: If filtering results in empty DataFrames.
    """

    #We only have cohort and cohort term split up, so combine and strip to lower to prevent cap issues
    df['cohort_selection'] = df[cohort_term_column].astype(str).str.lower() + " " + df[cohort_column].astype(str).str.lower()

    #Subset both datsets to only these cohorts
    df_filtered = df[df['cohort_selection'].isin(cohorts_list)].copy()

    logging.info(
    "Selected cohorts for inference: %s\nCohort counts in filtered data:\n%s",
    cohorts_list,
    df_filtered['cohort_selection'].value_counts().to_string()
    )

    #Throw error if either dataset is empty after filtering
    if df_filtered.empty:
        logging.error("Selected cohorts resulted in empty DataFrames.")
        raise ValueError("Selected cohorts resulted in empty DataFrames.")
    
    df_filtered.drop(columns="cohort_selection", inplace=True)
    
    return df_filtered