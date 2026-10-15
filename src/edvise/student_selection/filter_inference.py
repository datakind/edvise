import json
import logging
import typing as t

import pandas as pd

from edvise.feature_generation.constants import (
    DEFAULT_CORE_TERMS,
    DEFAULT_SEASON_ORDER_MAP,
)
from edvise.shared.utils import cohort_pair_columns
from edvise.utils.data_cleaning import convert_intensity_time_limits
from edvise.utils.types import IntensityTimeLimitsType

_ACADEMIC_TERM_COLS = ("academic_term", "academic_year")
_TERM_INTENSITY_COL = "student_term_enrollment_intensity"
_FIRST_TERM_INTENSITY_COL = "enrollment_intensity_first_term"


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


class _InferenceConfigHolder(t.Protocol):
    """PDP, Edvise, and legacy project configs all expose ``inference``."""

    inference: t.Any


def resolve_inference_terms_from_param(
    cfg: _InferenceConfigHolder,
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
    if set(_ACADEMIC_TERM_COLS).issubset(df.columns):
        logging.info(
            "Term breakdowns (counts):\n%s",
            df[list(_ACADEMIC_TERM_COLS)]
            .value_counts(dropna=False)
            .sort_index()
            .to_string(),
        )


def apply_inference_training_cohort_filter(
    df: pd.DataFrame,
    *,
    training_cohorts: list[str] | None,
    inf_terms: list[str],
    preprocessing: object | None,
    student_terms: pd.DataFrame | None,
    student_id_col: str,
) -> pd.DataFrame:
    """Term-filtered inference rows minus training cohort stop-outs (graduation-aware)."""
    if not training_cohorts:
        return df
    cohort_pair = cohort_pair_columns(df)
    if cohort_pair is None:
        logging.warning(
            "Training cohorts configured but cohort columns not found; "
            "skipping stop-out exclusion."
        )
        return df
    cohort_column, cohort_term_column = cohort_pair
    return exclude_training_cohort_students(
        df,
        training_cohorts=training_cohorts,
        cohort_column=cohort_column,
        cohort_term_column=cohort_term_column,
        student_terms=student_terms,
        student_id_col=student_id_col,
        **graduation_open_window(preprocessing, inf_terms),
    )


def exclude_training_cohort_students(
    df: pd.DataFrame,
    training_cohorts: list[str],
    cohort_term_column: str = "cohort_term",
    cohort_column: str = "cohort",
    *,
    as_of_term: str | None = None,
    intensity_time_limits: IntensityTimeLimitsType | None = None,
    num_terms_in_year: int = 4,
    student_terms: pd.DataFrame | None = None,
    student_id_col: str = "student_id",
) -> pd.DataFrame:
    """Exclude students whose entry cohort was used in model training.

    Graduation passes intensity limits so full-time and part-time students
    still inside their own window are kept. Other callers are unchanged.
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
    if as_of_term and intensity_time_limits and bool(exclude_mask.any()):
        # Only training-cohort rows can be kept; skip the rest of the frame.
        still_open = _still_inside_intensity_window(
            df.loc[exclude_mask],
            as_of_term=as_of_term,
            intensity_time_limits=intensity_time_limits,
            num_terms_in_year=num_terms_in_year,
            cohort_term_column=cohort_term_column,
            cohort_column=cohort_column,
            student_terms=student_terms,
            student_id_col=student_id_col,
        )
        n_kept = int(still_open.sum())
        if n_kept:
            logging.info(
                "Kept %d graduation students in training cohorts whose own "
                "full-time or part-time window is still open as of %s.",
                n_kept,
                as_of_term,
            )
        exclude_mask = exclude_mask.copy()
        exclude_mask.loc[still_open.index] = ~still_open.to_numpy()

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


def _calendar_year(season: str, academic_year_start: int) -> int:
    """Fall and winter of year N open that academic year. Spring and summer close it."""
    if season in ("fall", "winter"):
        return academic_year_start
    return academic_year_start + 1


def _season_position(num_terms_in_year: int) -> dict[str, int]:
    """0-based rank of each season inside one academic year.

    ``DEFAULT_SEASON_ORDER_MAP`` numbers a calendar year from spring. Pairing it
    with calendar year puts spring 2024-25 (Spring 2025) before fall 2025-26.
    Two-term years keep the core terms. Three-term years drop the trailing summer.
    """
    ordered = sorted(
        DEFAULT_SEASON_ORDER_MAP,
        key=lambda season: (
            _calendar_year(season, 0),
            DEFAULT_SEASON_ORDER_MAP[season],
        ),
    )
    if num_terms_in_year == 2:
        core = {term.lower() for term in DEFAULT_CORE_TERMS}
        ordered = [season for season in ordered if season in core]
    elif num_terms_in_year == 3:
        ordered = ordered[:3]
    return {season: index for index, season in enumerate(ordered)}


def _term_index(
    seasons: pd.Series, years: pd.Series, num_terms_in_year: int
) -> pd.Series:
    """Later terms compare greater. Unknown seasons are null."""
    position = _season_position(num_terms_in_year)
    season_key = seasons.astype("string").str.strip().str.lower()
    year_start = pd.to_numeric(
        years.astype("string").str.strip().str.split("-").str[0], errors="coerce"
    )
    return year_start * num_terms_in_year + season_key.map(position)


def _latest_term(term_list: list[str], num_terms_in_year: int) -> str:
    labels = _normalize_label_list(term_list)
    parts = [label.split(maxsplit=1) for label in labels]
    if not parts or any(len(part) < 2 for part in parts):
        raise ValueError(
            f"Inference terms must look like 'fall 2025-26', got {term_list!r}."
        )
    indexes = _term_index(
        pd.Series([part[0] for part in parts]),
        pd.Series([part[1] for part in parts]),
        num_terms_in_year,
    )
    if indexes.isna().any():
        raise ValueError(f"Could not order inference terms {term_list!r}.")
    return labels[int(indexes.to_numpy().argmax())]


def _intensity_by_term_label(
    keys: pd.DataFrame,
    lookup: pd.DataFrame | None,
    *,
    student_id_col: str,
    key_term_col: str,
    key_year_col: str,
    intensity_col: str = _TERM_INTENSITY_COL,
    lookup_term_col: str = _ACADEMIC_TERM_COLS[0],
    lookup_year_col: str = _ACADEMIC_TERM_COLS[1],
) -> pd.Series:
    """Map each key row to ``intensity_col`` on the lookup row with the same term label."""
    empty = pd.Series(pd.NA, index=keys.index)
    if lookup is None:
        return empty
    if not {
        student_id_col,
        key_term_col,
        key_year_col,
    }.issubset(keys.columns) or not {
        student_id_col,
        lookup_term_col,
        lookup_year_col,
        intensity_col,
    }.issubset(lookup.columns):
        return empty

    label = _joined_labels(keys, key_term_col, key_year_col)
    st = lookup[
        [student_id_col, lookup_term_col, lookup_year_col, intensity_col]
    ].copy()
    st["_label"] = _joined_labels(st, lookup_term_col, lookup_year_col)
    merged = (
        keys[[student_id_col]]
        .assign(_label=label.to_numpy())
        .merge(
            st.drop_duplicates([student_id_col, "_label"]),
            on=[student_id_col, "_label"],
            how="left",
        )
    )
    return merged[intensity_col].reset_index(drop=True)


def _enrollment_intensity_for_open_window(
    df: pd.DataFrame,
    *,
    student_terms: pd.DataFrame | None,
    student_id_col: str,
    cohort_term_column: str,
    cohort_column: str,
) -> pd.Series:
    """Intensity at cohort entry (not the checkpoint term when they differ)."""
    if _FIRST_TERM_INTENSITY_COL in df.columns:
        return df[_FIRST_TERM_INTENSITY_COL]

    from_terms = _intensity_by_term_label(
        df,
        student_terms,
        student_id_col=student_id_col,
        key_term_col=cohort_term_column,
        key_year_col=cohort_column,
    )
    if from_terms.notna().any():
        logging.info(
            "Using %s at cohort entry (%s not on frame).",
            _TERM_INTENSITY_COL,
            _FIRST_TERM_INTENSITY_COL,
        )
        return from_terms

    if _TERM_INTENSITY_COL not in df.columns:
        return pd.Series(pd.NA, index=df.index)

    at_entry = _joined_labels(df, cohort_term_column, cohort_column).eq(
        _joined_labels(df, *_ACADEMIC_TERM_COLS)
    )
    if not at_entry.any():
        return pd.Series(pd.NA, index=df.index)
    logging.info(
        "Using %s on checkpoint rows that coincide with cohort entry.",
        _TERM_INTENSITY_COL,
    )
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out.loc[at_entry] = df.loc[at_entry, _TERM_INTENSITY_COL]
    return out


def _year_limits_by_intensity(
    intensity: pd.Series,
    intensity_time_limits: IntensityTimeLimitsType,
    num_terms_in_year: int,
) -> pd.Series:
    limits = {
        str(k).strip().upper(): float(v)
        for k, v in convert_intensity_time_limits(
            "year", intensity_time_limits, num_terms_in_year=num_terms_in_year
        ).items()
    }
    return intensity.astype("string").str.strip().str.upper().map(limits)


def _still_inside_intensity_window(
    df: pd.DataFrame,
    *,
    as_of_term: str,
    intensity_time_limits: IntensityTimeLimitsType,
    num_terms_in_year: int,
    cohort_term_column: str,
    cohort_column: str,
    student_terms: pd.DataFrame | None = None,
    student_id_col: str = "student_id",
) -> pd.Series:
    """True where elapsed years are under the student's own limit.

    Year limits come from ``convert_intensity_time_limits``, the same helper
    graduation targets use. Intensity is read at enrollment (first term),
    not at checkpoint time. Missing intensity or start term stays excluded.
    """
    enrollment_intensity = _enrollment_intensity_for_open_window(
        df,
        student_terms=student_terms,
        student_id_col=student_id_col,
        cohort_term_column=cohort_term_column,
        cohort_column=cohort_column,
    )
    if enrollment_intensity.isna().all():
        logging.warning(
            "No enrollment-time intensity; excluding the full training cohort."
        )
        return pd.Series(False, index=df.index)

    season, year = as_of_term.strip().lower().split(maxsplit=1)
    as_of_idx = _term_index(
        pd.Series([season]), pd.Series([year]), num_terms_in_year
    ).iloc[0]
    start_idx = _term_index(
        df[cohort_term_column], df[cohort_column], num_terms_in_year
    )
    elapsed_years = (as_of_idx - start_idx + 1) / float(num_terms_in_year)
    year_limit = _year_limits_by_intensity(
        enrollment_intensity, intensity_time_limits, num_terms_in_year
    )
    return (
        elapsed_years.lt(year_limit) & start_idx.notna() & year_limit.notna()
    ).fillna(False)


def graduation_open_window(
    preprocessing: object | None,
    inf_terms: list[str],
) -> dict[str, t.Any]:
    """Exclusion kwargs for graduation. Empty for every other target."""
    target = getattr(preprocessing, "target", None)
    limits = getattr(target, "intensity_time_limits", None)
    if getattr(target, "type_", None) != "graduation" or not limits:
        return {}
    num_terms_in_year = int(getattr(target, "num_terms_in_year", None) or 4)
    return {
        "as_of_term": _latest_term(inf_terms, num_terms_in_year),
        "intensity_time_limits": limits,
        "num_terms_in_year": num_terms_in_year,
    }


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
