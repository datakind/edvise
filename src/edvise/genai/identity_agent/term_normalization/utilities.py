"""Utilities for adding term order and labels to a DataFrame in our GenAI IdentityAgent."""

from __future__ import annotations

import logging

import pandas as pd

from .schemas import TermOrderConfig

logger = logging.getLogger(__name__)

# Canonical seasons that begin an academic year.
# FALL and WINTER of year N belong to academic year N → N+1.
# SPRING and SUMMER of year N belong to academic year N-1 → N.
_ACADEMIC_YEAR_START_SEASONS = {"FALL", "WINTER"}


def add_edvise_term_order(
    df: pd.DataFrame,
    term_config: dict,
    year_extractor: callable | None = None,
    season_extractor: callable | None = None,
) -> pd.DataFrame:
    """
    Adds _year, _season, and term_order columns to a DataFrame
    from a term_config dictionary.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the term column.
    term_config : dict
        Term config emitted by IdentityAgent. Expected keys:
            term_col        : str — name of raw term column
            season_map      : list[{"raw": str, "canonical": str}] — chronologically ordered
            term_extraction : "standard" | "custom"
    year_extractor : callable | None
        Required when term_config["term_extraction"] == "custom".
        Signature: (str) -> int. Extracts 4-digit year from raw term string.
        Resolved from institution hook file via resolve_term_extractors().
    season_extractor : callable | None
        Required when term_config["term_extraction"] == "custom".
        Signature: (str) -> str. Extracts raw season token from term string.
        Resolved from institution hook file via resolve_term_extractors().

    Returns
    -------
    pd.DataFrame with added columns:
        _year        : Int64  — extracted year
        _season      : string — raw season token (e.g. "FA", "9", "Spring")
        _term_order  : Int64  — chronological sort key (year * 100 + season_rank)
    """
    term_col = term_config["term_col"]
    season_map = term_config["season_map"]
    term_extraction = term_config["term_extraction"]

    if term_col not in df.columns:
        raise KeyError(f"DataFrame must contain column '{term_col}'")

    if term_extraction == "custom" and (
        year_extractor is None or season_extractor is None
    ):
        raise ValueError(
            "term_extraction is 'custom' but year_extractor and/or season_extractor not provided. "
            "Resolve extractors from institution hook file via resolve_term_extractors()."
        )

    # Build lookup from season_map — rank is 1-indexed chronological position
    raw_to_rank = {item["raw"].lower(): i + 1 for i, item in enumerate(season_map)}
    norm_keys = sorted(raw_to_rank.keys(), key=len, reverse=True)

    out = df.copy()
    s = out[term_col].astype("string").str.strip()

    # --- Year extraction ---
    if year_extractor is not None:
        year_str = s.apply(lambda t: str(year_extractor(t)) if pd.notna(t) else None)
    else:
        year_str = s.str.extract(r"(\d{4})", expand=False)

    out["_year"] = pd.to_numeric(year_str, errors="coerce").astype("Int64")

    # --- Season extraction ---
    def _norm_token(t: str | None) -> str | None:
        if t is None:
            return None
        return " ".join(t.lower().split())

    if season_extractor is not None:
        season_norm = s.apply(
            lambda t: _norm_token(season_extractor(t)) if pd.notna(t) else None
        )
    else:

        def _extract_season(term: str | None) -> str | None:
            if term is None:
                return None
            t_norm = _norm_token(term)
            if t_norm is None:
                return None
            # Prefix match (e.g. "Fall 2019")
            for key in norm_keys:
                if t_norm.startswith(key):
                    return key
            # Suffix match (e.g. "2016FA", "2015S1")
            for key in norm_keys:
                if t_norm.endswith(key):
                    return key
            return None

        season_norm = s.apply(_extract_season)

    out["_season"] = season_norm.astype("string")

    # Warn on unexpected season tokens
    found = set(season_norm.dropna().unique())
    valid = set(raw_to_rank.keys())
    unexpected = found - valid
    if unexpected:
        logger.warning(
            f"Unexpected season tokens: {unexpected}. Filtering to valid: {valid}"
        )
        mask = season_norm.isin(valid)
        out = out[mask]
        season_norm = season_norm[mask]

    # --- term_order ---
    season_rank = season_norm.map(raw_to_rank).astype("Int64")
    out["_term_order"] = (out["_year"] * 100 + season_rank).astype("Int64")

    return out


def add_edvise_term_labels(
    df: pd.DataFrame,
    term_config: dict,
) -> pd.DataFrame:
    """
    Adds Edvise standard term columns to a DataFrame.
    Expects _year and _season columns produced by add_edvise_term_order.

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
        _edvise_term_year           : Int64  — e.g. 2017
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

    # _edvise_term_year — direct passthrough
    out["_edvise_term_year"] = out["_year"]

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
        logger.warning(f"Unmapped season tokens in standardization: {unmapped}")

    # _edvise_term_academic_year
    # FALL/WINTER of year N -> "N-(N+1 2-digit)"
    # SPRING/SUMMER of year N -> "(N-1)-(N 2-digit)"
    def _academic_year(row: pd.Series) -> str | pd.NA:
        year = row["_edvise_term_year"]
        season = row["_edvise_term_season"]
        if pd.isna(year) or pd.isna(season):
            return pd.NA
        year = int(year)
        if season in _ACADEMIC_YEAR_START_SEASONS:
            return f"{year}-{str(year + 1)[-2:]}"
        else:
            return f"{year - 1}-{str(year)[-2:]}"

    out["_edvise_term_academic_year"] = out.apply(_academic_year, axis=1).astype(
        "string"
    )

    return out


def apply_term_order_from_config(
    df: pd.DataFrame, config: TermOrderConfig
) -> pd.DataFrame:
    """
    Apply a validated :class:`~edvise.genai.identity_agent.term_normalization.schemas.TermOrderConfig`
    by calling :func:`add_edvise_term_order` with ``config`` serialized as the JSON-compatible dict
    IdentityAgent emits.

    Custom extraction (``term_extraction == \"custom\"``) requires ``year_extractor`` and
    ``season_extractor``; resolve those from ``hook_spec`` and call :func:`add_edvise_term_order`
    directly instead of this helper.
    """
    if config.term_extraction == "custom":
        raise ValueError(
            "term_config.term_extraction is 'custom' — provide year_extractor and season_extractor "
            "from hook_spec to add_edvise_term_order (or preprocess the term column)."
        )
    tc = config.model_dump(mode="json")
    return add_edvise_term_order(df, tc, year_extractor=None, season_extractor=None)
