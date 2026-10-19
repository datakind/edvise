import json
import logging
import typing as t

import pandas as pd

from edvise.shared.utils import cohort_pair_columns
from edvise.targets import shared as target_shared
from edvise.utils.academic_terms import (
    DEFAULT_FIRST_TERM_OF_YEAR,
    cohort_term_sort_ranks,
)
from edvise.utils.types import IntensityTimeLimitsType

_ACADEMIC_TERM_COLS = ("academic_term", "academic_year")
_TERM_INTENSITY_COL = "student_term_enrollment_intensity"
_TERM_RANK_COL = "term_rank"


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
    student_terms: pd.DataFrame | None,
    student_id_col: str,
    open_window: dict[str, t.Any] | None = None,
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
        **(open_window or {}),
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
    first_term_of_year: str = DEFAULT_FIRST_TERM_OF_YEAR,
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
            intensity_time_limits=intensity_time_limits,
            num_terms_in_year=num_terms_in_year,
            first_term_of_year=first_term_of_year,
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
        exclude_mask.loc[still_open.index] = ~still_open

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


def _first_term_of_year(preprocessing: object | None) -> str:
    features = getattr(preprocessing, "features", None)
    raw = getattr(features, "first_term_of_year", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip().upper()
    return DEFAULT_FIRST_TERM_OF_YEAR


def _latest_term(term_list: list[str], first_term_of_year: str) -> str:
    """Latest inference label using the institution academic-year term order."""
    labels = _normalize_label_list(term_list)
    season_ranks = cohort_term_sort_ranks(first_term_of_year)

    def sort_key(label: str) -> tuple[int, int]:
        season, year = label.split(maxsplit=1)
        rank = season_ranks.get(season)
        if rank is None:
            raise ValueError(
                f"Inference terms must look like 'fall 2025-26', got {term_list!r}."
            )
        return int(year.split("-")[0]), rank

    return max(labels, key=sort_key)


def _value_at_cohort_entry(
    keys: pd.DataFrame,
    lookup: pd.DataFrame | None,
    *,
    student_id_col: str,
    cohort_term_column: str,
    cohort_column: str,
    value_col: str,
    lookup_term_col: str = _ACADEMIC_TERM_COLS[0],
    lookup_year_col: str = _ACADEMIC_TERM_COLS[1],
) -> pd.Series:
    """Map each row to ``value_col`` on the lookup row at cohort entry."""
    empty = pd.Series(pd.NA, index=keys.index)
    if lookup is None:
        return empty
    if not {
        student_id_col,
        cohort_term_column,
        cohort_column,
    }.issubset(keys.columns) or not {
        student_id_col,
        lookup_term_col,
        lookup_year_col,
        value_col,
    }.issubset(lookup.columns):
        return empty

    cohort_label = _joined_labels(keys, cohort_term_column, cohort_column)
    st = lookup[[student_id_col, lookup_term_col, lookup_year_col, value_col]].copy()
    st["_label"] = _joined_labels(st, lookup_term_col, lookup_year_col)
    merged = (
        keys[[student_id_col]]
        .assign(_label=cohort_label.to_numpy())
        .merge(
            st.drop_duplicates([student_id_col, "_label"]),
            on=[student_id_col, "_label"],
            how="left",
        )
    )
    return pd.Series(merged[value_col].to_numpy(), index=keys.index)


def _entry_checkpoint_for_open_window(
    df: pd.DataFrame,
    *,
    student_terms: pd.DataFrame | None,
    student_id_col: str,
    cohort_term_column: str,
    cohort_column: str,
) -> pd.DataFrame | None:
    """Cohort-entry checkpoint rows (same grain training labelability uses)."""
    intensity = _value_at_cohort_entry(
        df,
        student_terms,
        student_id_col=student_id_col,
        cohort_term_column=cohort_term_column,
        cohort_column=cohort_column,
        value_col=_TERM_INTENSITY_COL,
    )
    term_rank = _value_at_cohort_entry(
        df,
        student_terms,
        student_id_col=student_id_col,
        cohort_term_column=cohort_term_column,
        cohort_column=cohort_column,
        value_col=_TERM_RANK_COL,
    )

    if student_terms is None and _TERM_INTENSITY_COL in df.columns:
        at_entry = _joined_labels(df, cohort_term_column, cohort_column).eq(
            _joined_labels(df, *_ACADEMIC_TERM_COLS)
        )
        if at_entry.any():
            intensity = intensity.where(intensity.notna(), df[_TERM_INTENSITY_COL])
            if _TERM_RANK_COL in df.columns:
                term_rank = term_rank.where(term_rank.notna(), df[_TERM_RANK_COL])

    if intensity.isna().all() or term_rank.isna().all():
        return None

    entry = pd.DataFrame(
        {
            _TERM_INTENSITY_COL: intensity,
            _TERM_RANK_COL: pd.to_numeric(term_rank, errors="coerce"),
            _ACADEMIC_TERM_COLS[0]: df[cohort_term_column],
        },
        index=df.index,
    )
    if student_id_col in df.columns:
        entry[student_id_col] = df[student_id_col]
    return entry


def _still_inside_intensity_window(
    df: pd.DataFrame,
    *,
    intensity_time_limits: IntensityTimeLimitsType,
    num_terms_in_year: int,
    first_term_of_year: str,
    cohort_term_column: str,
    cohort_column: str,
    student_terms: pd.DataFrame | None = None,
    student_id_col: str = "student_id",
) -> pd.Series:
    """True when the inference term is strictly before training's label boundary."""
    entry_ckpt = _entry_checkpoint_for_open_window(
        df,
        student_terms=student_terms,
        student_id_col=student_id_col,
        cohort_term_column=cohort_term_column,
        cohort_column=cohort_column,
    )
    if entry_ckpt is None or _TERM_RANK_COL not in df.columns:
        logging.warning(
            "Missing cohort-entry checkpoint or %s; excluding training cohort rows.",
            _TERM_RANK_COL,
        )
        return pd.Series(False, index=df.index)

    intensity_num_terms = target_shared.intensity_term_budgets(
        intensity_time_limits,
        num_terms_in_year=num_terms_in_year,
        first_term_of_year=first_term_of_year,
    )
    ranked = target_shared.assign_student_max_term_rank(
        entry_ckpt,
        intensity_num_terms=intensity_num_terms,
        num_terms_in_year=num_terms_in_year,
        first_term_of_year=first_term_of_year,
    )
    as_of_rank = pd.to_numeric(df[_TERM_RANK_COL], errors="coerce")
    return as_of_rank.lt(ranked["student_max_term_rank"]).fillna(False)


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
    first_term_of_year = _first_term_of_year(preprocessing)
    return {
        "as_of_term": _latest_term(inf_terms, first_term_of_year),
        "intensity_time_limits": limits,
        "num_terms_in_year": num_terms_in_year,
        "first_term_of_year": first_term_of_year,
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
