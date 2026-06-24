import json
import logging
import typing as t

import pandas as pd


def parse_term_filter_param(value: t.Optional[str]) -> t.Optional[list[str]]:
    """Parse ``--term_filter`` job param (same semantics as PDP ``pdp_inf_prep``).

    Treat ``None``, ``''``, ``'null'`` as not provided (use config). Invalid JSON
    raises ``ValueError``. After parsing, an empty list means not provided.
    """
    if value is None:
        return None
    s = value.strip()
    if s in ("", "null", "None"):
        return None
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON for term_filter param: %s", value)
        raise ValueError(f"Invalid JSON for --term_filter: {e}") from e
    if not isinstance(parsed, list):
        raise ValueError("--term_filter must be a JSON list of strings")
    labels = [str(item).strip() for item in parsed if str(item).strip()]
    if not labels:
        return None
    return labels


def _normalize_label_list(labels: list[str]) -> list[str]:
    return [
        label.strip().lower()
        for label in labels
        if label and str(label).strip()
    ]


def _joined_labels(
    df: pd.DataFrame,
    first_column: str,
    second_column: str,
) -> pd.Series:
    return (
        df[first_column].astype(str).str.lower()
        + " "
        + df[second_column].astype(str).str.lower()
    )


def _filter_by_joined_columns(
    df: pd.DataFrame,
    selection_list: list[str],
    first_column: str,
    second_column: str,
    selection_type: str,
) -> pd.DataFrame:
    """
    Base function to filter rows by combining two columns and matching against a list.

    Args:
        df: The DataFrame.
        selection_list: List of values to filter (e.g., ["fall 2023-24", "spring 2024-25"]).
        first_column: First column name to combine (e.g., "cohort_term" or "academic_term").
        second_column: Second column name to combine (e.g., "cohort" or "academic_year").
        selection_type: Type of selection for logging/errors (e.g., "cohorts" or "terms").

    Returns:
        The filtered DataFrame.

    Raises:
        ValueError: If selection_list has no non-empty labels, or if filtering results in empty DataFrame.
    """
    # Normalize labels to lowercase so matching is case-insensitive
    selection_list_normalized = _normalize_label_list(selection_list)
    if not selection_list_normalized:
        raise ValueError(
            f"{selection_type}_list had no non-empty {selection_type} labels."
        )

    # Combine columns and normalize to lowercase
    temp_column = f"{selection_type}_selection"
    df[temp_column] = _joined_labels(df, first_column, second_column)

    # Filter to only the specified values
    df_filtered = df[df[temp_column].isin(selection_list_normalized)].copy()

    logging.info(
        "Filtered %s for inference: %s\n%s counts in filtered data:\n%s",
        selection_type,
        selection_list,
        selection_type.capitalize(),
        df_filtered[temp_column].value_counts().to_string(),
    )

    # Throw error if dataset is empty after filtering
    if df_filtered.empty:
        logging.error(
            "Filtered %s resulted in empty DataFrame; requested %s_list=%s",
            selection_type,
            selection_type,
            selection_list,
        )
        raise ValueError(
            f"Filtered {selection_type} resulted in empty DataFrame; requested {selection_type}_list={selection_list!r}."
        )

    df_filtered.drop(columns=temp_column, inplace=True)

    return df_filtered


def filter_inference_cohort(
    df: pd.DataFrame,
    cohorts_list: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort",
) -> pd.DataFrame:
    """
    Filters the specified cohorts from DataFrame.

    Args:
        df: The DataFrame.
        cohorts_list: List of cohorts to filter (e.g., ["fall 2023-24", "spring 2024-25"]).
        cohort_term_column: Column name for cohort term (e.g. FALL, SPRING). Default "cohort_term".
        cohort_column: Column name for cohort year/label. Default "cohort".

    Returns:
        The filtered DataFrame.

    Raises:
        ValueError: If cohorts_list has no non-empty labels, or if filtering results in empty DataFrame.
    """
    return _filter_by_joined_columns(
        df=df,
        selection_list=cohorts_list,
        first_column=cohort_term_column,
        second_column=cohort_column,
        selection_type="cohorts",
    )


def exclude_training_cohort_students(
    df: pd.DataFrame,
    training_cohorts: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort",
) -> pd.DataFrame:
    """Exclude students whose entry cohort was used in model training."""
    training_labels = _normalize_label_list(training_cohorts)
    if not training_labels:
        return df

    if cohort_term_column not in df.columns or cohort_column not in df.columns:
        logging.warning(
            "Cannot exclude training cohort students: missing columns %r and/or %r.",
            cohort_term_column,
            cohort_column,
        )
        return df

    labels = _joined_labels(df, cohort_term_column, cohort_column)
    exclude_mask = labels.isin(training_labels)
    n_excluded = int(exclude_mask.sum())
    if n_excluded:
        logging.info(
            "Excluded %d stop-out students from training cohorts %s.\n%s",
            n_excluded,
            training_cohorts,
            labels[exclude_mask].value_counts().to_string(),
        )

    df_filtered = df.loc[~exclude_mask]
    if df_filtered.empty and not df.empty:
        raise ValueError(
            "Excluding training cohort students resulted in empty DataFrame; "
            f"training_cohorts={training_cohorts!r}."
        )
    return df_filtered


def filter_inference_term(
    df: pd.DataFrame,
    term_list: list[str],
    academic_term_col: str = "academic_term",
    academic_year_col: str = "academic_year",
) -> pd.DataFrame:
    """
    Filters the specified terms from DataFrame.

    Args:
        df: The DataFrame.
        term_list: List of terms to filter (e.g., ["fall 2023-24", "spring 2024-25"]).
        academic_term_col: Column name for term (e.g. FALL, SPRING). Default "academic_term".
        academic_year_col: Column name for year/label. Default "academic_year".

    Returns:
        The filtered DataFrame.

    Raises:
        ValueError: If term_list has no non-empty labels, or if filtering results in empty DataFrame.
    """
    return _filter_by_joined_columns(
        df=df,
        selection_list=term_list,
        first_column=academic_term_col,
        second_column=academic_year_col,
        selection_type="terms",
    )
