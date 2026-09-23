import json
import logging
import typing as t

import pandas as pd

from edvise.shared.utils import cohort_pair_columns
from edvise.utils.data_cleaning import convert_intensity_time_limits
from edvise.utils.types import IntensityTimeLimitsType

_SEASON_ORDER = {
    2: {"fall": 0, "spring": 1},
    3: {"fall": 0, "winter": 1, "spring": 2},
    4: {"fall": 0, "winter": 1, "spring": 2, "summer": 3},
}


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
    return [label.strip().lower() for label in labels if label and str(label).strip()]


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
    *,
    as_of_term: str | None = None,
    intensity_time_limits: IntensityTimeLimitsType | None = None,
    num_terms_in_year: int = 2,
    enrollment_intensity_col: str | None = None,
) -> pd.DataFrame:
    """Exclude students whose entry cohort was used in model training.

    A cohort is recorded once any student in it is labelable, which is
    usually the full-time limit. Part-time classmates in that cohort often
    are not labelable yet and were never in the training set. When
    ``intensity_time_limits`` are provided, those students stay; anyone
    already past their own limit is still excluded.
    """
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
    if as_of_term and intensity_time_limits:
        still_open = _still_inside_intensity_window(
            df,
            as_of_term=as_of_term,
            intensity_time_limits=intensity_time_limits,
            num_terms_in_year=num_terms_in_year,
            cohort_term_column=cohort_term_column,
            cohort_column=cohort_column,
            enrollment_intensity_col=enrollment_intensity_col,
        )
        n_kept = int((exclude_mask & still_open).sum())
        if n_kept:
            logging.info(
                "Kept %d training-cohort students still inside their intensity "
                "window as of %s. Their cohort is on the training list because "
                "other intensities were already labelable.",
                n_kept,
                as_of_term,
            )
        exclude_mask = exclude_mask & ~still_open

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


def parse_term_label(label: str) -> tuple[str, str]:
    """Split ``fall 2025-26`` into ``("fall", "2025-26")``."""
    parts = str(label).strip().split()
    if len(parts) < 2:
        raise ValueError(f"Term label must look like 'fall 2025-26', got {label!r}")
    return parts[0], parts[1]


def _canonical_season(season: str, num_terms_in_year: int) -> str:
    key = str(season).strip().lower()
    if num_terms_in_year == 2:
        if key == "winter":
            return "fall"
        if key == "summer":
            return "spring"
    return key


def academic_term_index(season: str, academic_year: str, num_terms_in_year: int) -> int:
    """Integer index so later academic terms compare greater."""
    order = _SEASON_ORDER.get(num_terms_in_year, _SEASON_ORDER[4])
    season_key = _canonical_season(season, num_terms_in_year)
    if season_key not in order:
        raise ValueError(
            f"Unknown academic term season {season!r} for "
            f"num_terms_in_year={num_terms_in_year}"
        )
    year_start = int(str(academic_year).strip().split("-")[0])
    return year_start * num_terms_in_year + order[season_key]


def latest_as_of_term(term_list: list[str], num_terms_in_year: int) -> str:
    """Return the latest label in ``term_list`` on the school's term calendar."""
    labels = _normalize_label_list(term_list)
    if not labels:
        raise ValueError("term_list had no non-empty term labels.")
    return max(
        labels,
        key=lambda lab: academic_term_index(*parse_term_label(lab), num_terms_in_year),
    )


def _term_index_series(
    seasons: pd.Series, years: pd.Series, num_terms_in_year: int
) -> pd.Series:
    order = _SEASON_ORDER.get(num_terms_in_year, _SEASON_ORDER[4])
    season_key = seasons.astype("string").str.strip().str.lower()
    if num_terms_in_year == 2:
        season_key = season_key.replace({"winter": "fall", "summer": "spring"})
    year_start = (
        years.astype("string").str.strip().str.split("-").str[0].astype("Int32")
    )
    return year_start * num_terms_in_year + season_key.map(order).astype("Int32")


def _intensity_year_limit(
    intensity: pd.Series,
    intensity_num_years: dict[str, float],
) -> pd.Series:
    if "*" in intensity_num_years:
        return pd.Series(
            intensity_num_years["*"], index=intensity.index, dtype="float64"
        )
    mapped = (
        intensity.astype("string")
        .str.strip()
        .str.upper()
        .map({str(k).strip().upper(): float(v) for k, v in intensity_num_years.items()})
    )
    return mapped.astype("float64")


def resolve_enrollment_intensity_col(df: pd.DataFrame) -> str:
    """Prefer checkpoint intensity, then first-term intensity."""
    for col in (
        "student_term_enrollment_intensity",
        "enrollment_intensity_first_term",
    ):
        if col in df.columns:
            return col
    raise ValueError(
        "Cannot apply open-window inference filter; missing enrollment "
        "intensity column (student_term_enrollment_intensity or "
        "enrollment_intensity_first_term)."
    )


def _still_inside_intensity_window(
    df: pd.DataFrame,
    *,
    as_of_term: str,
    intensity_time_limits: IntensityTimeLimitsType,
    num_terms_in_year: int,
    cohort_term_column: str,
    cohort_column: str,
    enrollment_intensity_col: str | None,
) -> pd.Series:
    """True where elapsed time is strictly less than the student's own limit.

    Unknown intensity or start term is False so those training-cohort rows
    stay excluded.
    """
    try:
        if enrollment_intensity_col is None:
            enrollment_intensity_col = resolve_enrollment_intensity_col(df)
    except ValueError:
        logging.warning(
            "Intensity limits are configured but no enrollment intensity "
            "column was found; excluding the full training cohort."
        )
        return pd.Series(False, index=df.index)

    if enrollment_intensity_col not in df.columns:
        logging.warning(
            "Intensity limits are configured but %r is missing; excluding "
            "the full training cohort.",
            enrollment_intensity_col,
        )
        return pd.Series(False, index=df.index)

    as_of_idx = academic_term_index(*parse_term_label(as_of_term), num_terms_in_year)
    start_idx = _term_index_series(
        df[cohort_term_column], df[cohort_column], num_terms_in_year
    )
    elapsed_years = (as_of_idx - start_idx + 1) / float(num_terms_in_year)
    intensity_num_years = convert_intensity_time_limits(
        "year", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )
    year_limit = _intensity_year_limit(
        df[enrollment_intensity_col], intensity_num_years
    )
    still_open = elapsed_years.lt(year_limit) & start_idx.notna() & year_limit.notna()
    return still_open.fillna(False)


def _resolve_intensity_time_limits(
    preprocessing: object | None,
) -> IntensityTimeLimitsType | None:
    target = getattr(preprocessing, "target", None)
    selection = getattr(preprocessing, "selection", None)
    return getattr(target, "intensity_time_limits", None) or getattr(
        selection, "intensity_time_limits", None
    )


def resolve_inference_terms_from_param(
    cfg: t.Any,
    *,
    schema_type: str,
    term_filter: str | None,
    job_type: str = "inference",
) -> None:
    """Set ``cfg.inference.term`` from ``--term_filter`` when the job param is provided."""
    if job_type != "inference":
        return

    from edvise.configs.es import InferenceConfig as ESInferenceConfig
    from edvise.configs.pdp import InferenceConfig as PDPInferenceConfig
    from edvise.shared.schema_type import is_edvise_schema

    param = parse_term_filter_param(term_filter)
    if param is not None:
        inference_config_cls = (
            ESInferenceConfig if is_edvise_schema(schema_type) else PDPInferenceConfig
        )
        if cfg.inference is None:
            cfg.inference = inference_config_cls(cohort=param)
        else:
            cfg.inference.term = param
        logging.info("Inference cohort source: job param; term_filter=%s", param)
    else:
        logging.info(
            "Inference cohort source: config; cohort=%s",
            cfg.inference.term if cfg.inference else None,
        )


def select_inference_students(
    df: pd.DataFrame,
    *,
    inf_terms: list[str],
    preprocessing: object | None = None,
    cohort_pair: tuple[str, str] | None = None,
    training_cohorts: list[str] | None = None,
) -> pd.DataFrame:
    """Filter merged checkpoint rows to the inference scoring population.

    Training-cohort exclusion always runs. When the frozen config has
    intensity time limits, students in those cohorts whose own window is
    still open are kept. That is the usual part-time case: the cohort was
    listed because full-time classmates were already labelable.
    """
    logging.info(
        "Selecting students for inference who met the checkpoint in term(s) of interest"
    )
    df_filtered = filter_inference_term(df, term_list=inf_terms)
    if not training_cohorts:
        return df_filtered
    cohort_pair = cohort_pair or cohort_pair_columns(df_filtered)
    if cohort_pair is None:
        logging.warning(
            "Training cohorts configured but cohort columns not found; "
            "skipping stop-out exclusion."
        )
        return df_filtered

    cohort_year_column, cohort_term_column = cohort_pair
    limits = _resolve_intensity_time_limits(preprocessing)
    window_kwargs: dict[str, t.Any] = {}
    if limits:
        target = getattr(preprocessing, "target", None)
        num_terms_in_year = int(getattr(target, "num_terms_in_year", None) or 2)
        as_of_term = latest_as_of_term(inf_terms, num_terms_in_year)
        logging.info(
            "Training-cohort exclusion keeps students still inside their "
            "intensity window as of %s.",
            as_of_term,
        )
        window_kwargs = {
            "as_of_term": as_of_term,
            "intensity_time_limits": limits,
            "num_terms_in_year": num_terms_in_year,
        }
    return exclude_training_cohort_students(
        df_filtered,
        training_cohorts=training_cohorts,
        cohort_term_column=cohort_term_column,
        cohort_column=cohort_year_column,
        **window_kwargs,
    )


def log_inference_selection_breakdown(
    df: pd.DataFrame,
    cohort_pair: tuple[str, str] | None,
) -> None:
    if cohort_pair is not None:
        cohort_column, cohort_term_column = cohort_pair
        logging.info(
            "Cohort & Cohort Term breakdowns (counts):\n%s",
            df[[cohort_column, cohort_term_column]]
            .value_counts(dropna=False)
            .sort_index()
            .to_string(),
        )
    if {"academic_year", "academic_term"}.issubset(df.columns):
        logging.info(
            "Term breakdowns (counts):\n%s",
            df[["academic_year", "academic_term"]]
            .value_counts(dropna=False)
            .sort_index()
            .to_string(),
        )


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
