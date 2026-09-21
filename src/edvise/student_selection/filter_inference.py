import json
import logging
import typing as t

import pandas as pd

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


def filter_inference_open_window(
    df: pd.DataFrame,
    *,
    as_of_term: str,
    intensity_time_limits: IntensityTimeLimitsType,
    num_terms_in_year: int,
    years_to_degree_col: str | None = None,
    exclude_graduates: bool = True,
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort",
    checkpoint_term_col: str = "academic_term",
    checkpoint_year_col: str = "academic_year",
    enrollment_intensity_col: str | None = None,
) -> pd.DataFrame:
    """Keep students whose on-time window is not yet complete.

    A student is kept when, as of ``as_of_term``:

    - their checkpoint term is on or before that term (already reached the
      checkpoint, e.g. 30 credits)
    - elapsed core terms since first cohort enrollment are **strictly less
      than** the intensity-specific limit (e.g. 3.0 years full-time, 4.5
      part-time). Students enrolled long enough to be labelable for training
      are excluded so they cannot leak into inference.
    - if ``exclude_graduates``, ``years_to_degree_col`` is null
    """
    if enrollment_intensity_col is None:
        enrollment_intensity_col = resolve_enrollment_intensity_col(df)

    required = {
        cohort_term_column,
        cohort_column,
        checkpoint_term_col,
        checkpoint_year_col,
        enrollment_intensity_col,
    }
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(
            f"Cannot apply open-window inference filter; missing columns {missing}."
        )

    as_of_idx = academic_term_index(*parse_term_label(as_of_term), num_terms_in_year)
    start_idx = _term_index_series(
        df[cohort_term_column], df[cohort_column], num_terms_in_year
    )
    ckpt_idx = _term_index_series(
        df[checkpoint_term_col], df[checkpoint_year_col], num_terms_in_year
    )
    elapsed_years = (as_of_idx - start_idx + 1) / float(num_terms_in_year)
    intensity_num_years = convert_intensity_time_limits(
        "year", intensity_time_limits, num_terms_in_year=num_terms_in_year
    )
    year_limit = _intensity_year_limit(
        df[enrollment_intensity_col], intensity_num_years
    )

    reached_checkpoint = ckpt_idx.le(as_of_idx)
    still_open = elapsed_years.lt(year_limit)
    keep = reached_checkpoint & still_open & start_idx.notna() & year_limit.notna()

    n_before = len(df)
    n_not_yet_ckpt = int((~reached_checkpoint.fillna(False)).sum())
    n_labelable = int(
        (reached_checkpoint.fillna(False) & ~still_open.fillna(False)).sum()
    )
    n_unknown = int(
        (keep.isna() | (reached_checkpoint & still_open & year_limit.isna())).sum()
    )

    if exclude_graduates:
        if not years_to_degree_col or years_to_degree_col not in df.columns:
            logging.warning(
                "exclude_graduates is set but %r is missing; skipping "
                "graduate exclusion.",
                years_to_degree_col,
            )
        else:
            graduated = df[years_to_degree_col].notna()
            n_graduated = int((keep.fillna(False) & graduated).sum())
            keep = keep & ~graduated
            logging.info("Excluded %d students who already graduated.", n_graduated)

    df_filtered = df.loc[keep.fillna(False)].copy()
    logging.info(
        "Open-window inference as of %s: kept %d/%d. "
        "Not yet at checkpoint=%d; already labelable (enrolled "
        ">= intensity limit)=%d; unknown intensity/start=%d.",
        as_of_term,
        len(df_filtered),
        n_before,
        n_not_yet_ckpt,
        n_labelable,
        n_unknown,
    )
    if df_filtered.empty and not df.empty:
        raise ValueError(
            "Open-window inference filter resulted in empty DataFrame; "
            f"as_of_term={as_of_term!r}."
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
