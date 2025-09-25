import pandas as pd
import logging


def select_inference_cohort(
    self, 
    df_course: pd.DataFrame, 
    df_cohort: pd.DataFrame, 
    cohorts_list: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort"
)-> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Selects the specified cohorts from the course and cohort DataFrames.

    Args:
        df_course: The course DataFrame.
        df_cohort: The cohort DataFrame.
        cohorts_list: List of cohorts to select (e.g., ["fall 2023-24", "spring 2024-25"]).

    Returns:
        A tuple containing the filtered course and cohort DataFrames.
    
    Raises:
        ValueError: If filtering results in empty DataFrames.
    """

    #We only have cohort and cohort term split up, so combine and strip to lower to prevent cap issues
    df_course['cohort_selection'] = df_course[cohort_term_column].astype(str).str.lower() + " " + df_course[cohort_column].astype(str).str.lower()
    df_cohort['cohort_selection'] = df_cohort[cohort_term_column].astype(str).str.lower() + " " + df_cohort[cohort_column].astype(str).str.lower()
    cohorts_list = [c.lower().strip() for c in cohorts_list]

    #Subset both datsets to only these cohorts
    df_course_filtered = df_course[df_course['cohort_selection'].isin(cohorts_list)].copy()
    df_cohort_filtered = df_cohort[df_cohort['cohort_selection'].isin(cohorts_list)].copy()
       
    #Throw error if either dataset is empty after filtering
    if df_course_filtered.empty or df_cohort_filtered.empty:
        logging.error("Selected cohorts resulted in empty DataFrames.")
        raise ValueError("Selected cohorts resulted in empty DataFrames.")
    
    df_course_filtered.drop(columns="cohort_selection", inplace=True)
    df_cohort_filtered.drop(columns="cohort_selection", inplace=True)
    
    return df_course_filtered, df_cohort_filtered    