"""Utilities for adding term order and labels to a DataFrame in our GenAI IdentityAgent."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from edvise.utils.data_cleaning import convert_to_snake_case

from .schemas import TermOrderConfig

logger = logging.getLogger(__name__)


def _normalize_term_config_column_names(tc: dict) -> dict:
    """
    Align ``term_col`` / ``year_col`` / ``season_col`` with :func:`~edvise.data_audit.custom_cleaning.normalize_columns`.

    IdentityAgent may emit uppercase or mixed-case names; ``clean_dataset`` always uses
    :func:`~edvise.utils.data_cleaning.convert_to_snake_case` on headers before term order runs.
    """
    out = dict(tc)
    for key in ("term_col", "year_col", "season_col"):
        v = out.get(key)
        if v is not None:
            out[key] = convert_to_snake_case(v)
    return out


# Canonical seasons that begin an academic year.
# FALL and WINTER of year N belong to academic year N → N+1.
# SPRING and SUMMER of year N belong to academic year N-1 → N.
_ACADEMIC_YEAR_START_SEASONS = {"FALL", "WINTER"}


def _norm_token(t: str | None) -> str | None:
    if t is None:
        return None
    return " ".join(t.lower().split())


def _resolve_season_token(t_norm: str | None, norm_keys: list[str]) -> str | None:
    """Map a normalized token to a season_map raw key (lowercase)."""
    if t_norm is None:
        return None
    for key in norm_keys:
        if t_norm == key:
            return key
    for key in norm_keys:
        if t_norm.startswith(key):
            return key
    for key in norm_keys:
        if t_norm.endswith(key):
            return key
    return None


def _season_map_lookups(season_map: list) -> tuple[dict[str, int], list[str]]:
    raw_to_rank = {item["raw"].lower(): i + 1 for i, item in enumerate(season_map)}
    norm_keys = sorted(raw_to_rank.keys(), key=len, reverse=True)
    return raw_to_rank, norm_keys


def _finalize_season_year_order(
    out: pd.DataFrame,
    season_norm: pd.Series,
    raw_to_rank: dict[str, int],
) -> pd.DataFrame:
    out = out.copy()
    out["_season"] = season_norm.astype("string")

    found = set(season_norm.dropna().unique())
    valid = set(raw_to_rank.keys())
    unexpected = found - valid
    if unexpected:
        logger.warning(
            "Unexpected season tokens: %s. Filtering to valid: %s",
            unexpected,
            valid,
        )
        mask = season_norm.isin(valid)
        out = out[mask]
        season_norm = season_norm[mask]

    season_rank = season_norm.map(raw_to_rank).astype("Int64")
    out["_term_order"] = (out["_year"] * 100 + season_rank).astype("Int64")
    return out


def add_edvise_term_order(
    df: pd.DataFrame,
    term_config: dict,
    year_extractor: callable | None = None,
    season_extractor: callable | None = None,
) -> pd.DataFrame:
    """
    Adds _year, _season, _term_order, and standard term label columns to a DataFrame
    from a term_config dictionary.

    Runs :func:`add_edvise_term_labels` at the end so callers always get canonical
    season and academic-year strings alongside the sort key.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the term column(s).
    term_config : dict
        Term config emitted by IdentityAgent. Either:

        - ``term_col``: single column encoding both year and season; or
        - ``year_col`` and ``season_col``: separate columns (mutually exclusive with ``term_col``).

        Also ``season_map``, ``term_extraction`` (``standard`` | ``custom``).
    year_extractor : callable | None
        Required when term_config["term_extraction"] == "custom" (combined ``term_col`` only).
    season_extractor : callable | None
        Required when term_config["term_extraction"] == "custom" (combined ``term_col`` only).

    Returns
    -------
    pd.DataFrame with added columns:
        _year                     : Int64  — extracted calendar year
        _season                   : string — raw season token (e.g. "FA", "9", "Spring")
        _term_order               : Int64  — chronological sort key (year * 100 + season_rank)
        _edvise_term_season       : string — canonical season (FALL, SPRING, SUMMER, WINTER)
        _edvise_term_academic_year: string — e.g. "2017-18"
    """
    season_map = term_config["season_map"]
    term_extraction = term_config["term_extraction"]
    term_col = term_config.get("term_col")
    year_col = term_config.get("year_col")
    season_col = term_config.get("season_col")

    has_single = term_col is not None
    has_split = year_col is not None and season_col is not None
    has_partial_split = (year_col is None) != (season_col is None)

    if has_partial_split:
        raise ValueError(
            "year_col and season_col must be provided together in term_config, not individually."
        )
    if not has_single and not has_split:
        raise ValueError(
            "term_config must include term_col or both year_col and season_col."
        )
    if has_single and has_split:
        raise ValueError(
            "term_col is mutually exclusive with year_col and season_col in term_config."
        )

    if term_extraction == "custom":
        if has_split:
            raise ValueError(
                "term_extraction 'custom' is not supported when year_col and season_col are set."
            )
        if year_extractor is None or season_extractor is None:
            raise ValueError(
                "term_extraction is 'custom' but year_extractor and/or season_extractor not provided. "
                "Resolve extractors from institution hook file via resolve_term_extractors()."
            )
    elif has_split and (year_extractor is not None or season_extractor is not None):
        raise ValueError(
            "year_extractor and season_extractor are only used with term_extraction 'custom' "
            "and a combined term_col."
        )

    raw_to_rank, norm_keys = _season_map_lookups(season_map)
    out = df.copy()

    if has_split:
        for c in (year_col, season_col):
            if c not in out.columns:
                raise KeyError(f"DataFrame must contain column '{c}'")
        out["_year"] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        s_season = out[season_col].astype("string").str.strip()

        def _cell_to_season_norm(val: object) -> str | None:
            if val is None or (isinstance(val, str) and val.strip() == ""):
                return None
            try:
                if pd.isna(val):
                    return None
            except (ValueError, TypeError):
                return None
            return _resolve_season_token(_norm_token(str(val).strip()), norm_keys)

        season_norm = s_season.map(_cell_to_season_norm)
        ordered = _finalize_season_year_order(out, season_norm, raw_to_rank)
        return add_edvise_term_labels(ordered, term_config)

    # --- Combined term_col path ---
    if term_col not in out.columns:
        raise KeyError(f"DataFrame must contain column '{term_col}'")

    s = out[term_col].astype("string").str.strip()

    if year_extractor is not None:
        year_str = s.apply(lambda t: str(year_extractor(t)) if pd.notna(t) else None)
    else:
        year_str = s.str.extract(r"(\d{4})", expand=False)

    out["_year"] = pd.to_numeric(year_str, errors="coerce").astype("Int64")

    if season_extractor is not None:
        # Extractor returns the raw season fragment; normalize to season_map keys (lowercase).
        season_norm = s.apply(
            lambda t: _norm_token(season_extractor(t)) if pd.notna(t) else None
        )
    else:

        def _extract_season(term: str | None) -> str | None:
            return _resolve_season_token(_norm_token(term), norm_keys)

        season_norm = s.apply(_extract_season)

    ordered = _finalize_season_year_order(out, season_norm, raw_to_rank)
    return add_edvise_term_labels(ordered, term_config)


def add_edvise_term_labels(
    df: pd.DataFrame,
    term_config: dict,
) -> pd.DataFrame:
    """
    Adds Edvise standard term label columns to a DataFrame.
    Expects ``_year`` and ``_season`` (e.g. from :func:`add_edvise_term_order`, which
    invokes this function automatically).

    Calendar year is ``_year`` only; there is no separate ``_edvise_term_year`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, must contain _year and _season columns.
    term_config : dict
        Term config emitted by IdentityAgent. Expected keys:
            season_map : list[{"raw": str, "canonical": str}]

    Returns
    -------
    pd.DataFrame with added columns:
        _edvise_term_season         : string — e.g. "FALL"
        _edvise_term_academic_year  : string — e.g. "2017-18"
    """
    for col in ("_year", "_season"):
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found. "
                "Run add_edvise_term_order before add_edvise_term_labels."
            )

    season_map = term_config["season_map"]
    raw_to_canonical = {
        item["raw"].lower(): item["canonical"].upper() for item in season_map
    }

    out = df.copy()

    # _edvise_term_season — map raw token to canonical label
    out["_edvise_term_season"] = (
        out["_season"]
        .astype("string")
        .str.lower()
        .map(raw_to_canonical)
        .astype("string")
    )

    # Warn on unmapped seasons
    unmapped = out.loc[
        out["_season"].notna() & out["_edvise_term_season"].isna(), "_season"
    ].unique()
    if len(unmapped) > 0:
        logger.warning("Unmapped season tokens in standardization: %s", unmapped)

    # _edvise_term_academic_year
    # FALL/WINTER of year N -> "N-(N+1 2-digit)"
    # SPRING/SUMMER of year N -> "(N-1)-(N 2-digit)"
    def _academic_year(row: pd.Series) -> str | pd.NA:
        year = row["_year"]
        season = row["_edvise_term_season"]
        if pd.isna(year) or pd.isna(season):
            return pd.NA
        year = int(year)
        if season in _ACADEMIC_YEAR_START_SEASONS:
            return f"{year}-{str(year + 1)[-2:]}"
        return f"{year - 1}-{str(year)[-2:]}"

    out["_edvise_term_academic_year"] = out.apply(_academic_year, axis=1).astype(
        "string"
    )

    return out


def term_order_column_for_clean_dataset(config: TermOrderConfig) -> str:
    """
    Return the column name to set on ``CleanSpec.term_column`` when using
    :func:`term_order_fn_from_term_order_config` with :func:`~edvise.data_audit.custom_cleaning.clean_dataset`.

    Names are passed through :func:`~edvise.utils.data_cleaning.convert_to_snake_case` so they match
    headers after ``clean_dataset`` step (1) normalization.

    ``clean_dataset`` only checks a single column name before invoking ``term_order_fn``.
    For split year/season configs, this returns normalized ``year_col`` so that check passes; the frame
    must still contain normalized ``season_col`` from the same config.
    """
    if config.term_col is not None:
        return convert_to_snake_case(config.term_col)
    if config.year_col is not None:
        return convert_to_snake_case(config.year_col)
    raise ValueError(
        "TermOrderConfig must set term_col or year_col for clean_dataset integration."
    )


def term_order_fn_from_term_order_config(
    config: TermOrderConfig,
) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    """
    Build a ``(df, term_column) -> df`` hook compatible with
    :class:`~edvise.data_audit.custom_cleaning.TermOrderFn` / ``clean_dataset``.

    Pass the result as ``term_order_fn`` and set ``term_column`` to
    :func:`term_order_column_for_clean_dataset`\\(config\\) so the optional term-order step runs.

    Custom extraction (``term_extraction == \"custom\"``) is not supported here; resolve
    extractors from ``hook_spec`` and call :func:`add_edvise_term_order` directly instead.
    """
    if config.term_extraction == "custom":
        raise ValueError(
            "term_order_fn_from_term_order_config does not support term_extraction 'custom'; "
            "call add_edvise_term_order with year_extractor and season_extractor from hook_spec."
        )
    expected_column = term_order_column_for_clean_dataset(config)
    tc = _normalize_term_config_column_names(config.model_dump(mode="json"))

    def _fn(df: pd.DataFrame, term_column: str) -> pd.DataFrame:
        if term_column != expected_column:
            raise ValueError(
                f"term_column {term_column!r} must be {expected_column!r} for this TermOrderConfig "
                "(use term_order_column_for_clean_dataset(config) when building CleanSpec)."
            )
        return add_edvise_term_order(
            df, tc, year_extractor=None, season_extractor=None
        )

    return _fn


def apply_term_order_from_config(
    df: pd.DataFrame, config: TermOrderConfig
) -> pd.DataFrame:
    """
    Apply a validated :class:`~edvise.genai.identity_agent.term_normalization.schemas.TermOrderConfig`
    by calling :func:`add_edvise_term_order` with ``config`` serialized as the JSON-compatible dict
    IdentityAgent emits (including :func:`add_edvise_term_labels`).

    Custom extraction (``term_extraction == \"custom\"``) requires ``year_extractor`` and
    ``season_extractor``; resolve those from ``hook_spec`` and call :func:`add_edvise_term_order`
    directly instead of this helper.
    """
    if config.term_extraction == "custom":
        raise ValueError(
            "term_config.term_extraction is 'custom' — provide year_extractor and season_extractor "
            "from hook_spec to add_edvise_term_order (or preprocess the term column)."
        )
    tc = _normalize_term_config_column_names(config.model_dump(mode="json"))
    return add_edvise_term_order(df, tc, year_extractor=None, season_extractor=None)
