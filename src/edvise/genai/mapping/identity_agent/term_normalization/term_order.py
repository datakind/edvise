"""Apply term order and Edvise term labels to a DataFrame (IdentityAgent term_config / TermOrderConfig)."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pandas.api.types as ptypes

from edvise.utils.data_cleaning import convert_to_snake_case

from edvise.genai.mapping.shared.hitl.hook_spec.schemas import HookSpec
from edvise.genai.mapping.shared.schema_contract.schemas import TermNormalizationSummary

from .schemas import TermOrderConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EdviseTermColumnSet:
    """Column names materialized by :func:`add_edvise_term_order` for one normalization stream."""

    year: str
    season: str
    term_order: str
    term_grain: str
    edvise_season: str
    edvise_academic_year: str


def edvise_term_column_set(output_prefix: str | None) -> EdviseTermColumnSet:
    """
    Map optional materialized prefix to Edvise term work column names.

    IdentityAgent emits entry-only configs (unprefixed ``_edvise_term_*``). A non-null prefix is
    still honored when present on legacy term_config dicts so older cleaned frames remain readable.
    """
    if not output_prefix:
        return EdviseTermColumnSet(
            year="_year",
            season="_season",
            term_order="_term_order",
            term_grain="_term_grain",
            edvise_season="_edvise_term_season",
            edvise_academic_year="_edvise_term_academic_year",
        )
    p = output_prefix
    return EdviseTermColumnSet(
        year=f"{p}_year",
        season=f"{p}_season",
        term_order=f"{p}_term_order",
        term_grain=f"{p}_term_grain",
        edvise_season=f"{p}_edvise_term_season",
        edvise_academic_year=f"{p}_edvise_term_academic_year",
    )


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
    try:
        if pd.isna(t):
            return None
    except (ValueError, TypeError):
        pass
    s = t if isinstance(t, str) else str(t).strip()
    if s == "":
        return None
    return " ".join(s.lower().split())


def _calendar_year_from_column(series: pd.Series) -> pd.Series:
    """Extract calendar year from a split ``year_col`` (Int64, string year, or datetime)."""
    if ptypes.is_datetime64_any_dtype(series):
        return series.dt.year.astype("Int64")
    return pd.to_numeric(series, errors="coerce").astype("Int64")


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


def _is_excluded_by_term_prefix(
    series: pd.Series, exclude_tokens: list[str]
) -> pd.Series:
    """True where the row should be dropped — raw string starts with any exclude prefix (case-insensitive)."""
    if not exclude_tokens:
        return pd.Series(False, index=series.index)
    cleaned = series.astype("string").fillna("").str.strip().str.lower()
    excluded = pd.Series(False, index=series.index)
    for tok in exclude_tokens:
        tl = tok.strip().lower()
        if tl:
            excluded |= cleaned.str.startswith(tl)
    return excluded


def _season_map_lookups(season_map: list) -> tuple[dict[str, int], list[str]]:
    raw_to_rank = {item["raw"].lower(): i + 1 for i, item in enumerate(season_map)}
    norm_keys = sorted(raw_to_rank.keys(), key=len, reverse=True)
    return raw_to_rank, norm_keys


def _add_term_grain(df: pd.DataFrame, cols: EdviseTermColumnSet) -> pd.DataFrame:
    """
    Stable source-term key from calendar year, raw season token, and sort key.

    Used downstream for PDP ``source_term_key`` / SMA mapping so enrollments stay
    distinct when ``_edvise_term_*`` canonical labels collapse multiple source terms.
    """
    out = df.copy()
    out[cols.term_grain] = (
        out[cols.year].astype("string")
        + "|"
        + out[cols.season].astype("string")
        + "|"
        + out[cols.term_order].astype("string")
    )
    return out


def _calendar_year_from_semantics(
    year: pd.Series,
    season_norm: pd.Series,
    raw_to_canonical: dict[str, str],
    year_semantics: str | None,
) -> pd.Series:
    """
    Adjust an extracted term year to a calendar year based on ``year_semantics``.

    ``None`` / ``"calendar_literal"`` leaves the year unchanged (extracted year is already
    the calendar year). For ``"academic_year_prefix"`` the extracted year is the academic-year
    start: FALL/WINTER keep it, while SPRING/SUMMER roll forward one calendar year. This rule is
    independent of how the season is encoded (letter suffix, numeric period code, spelled) — that
    is resolved upstream by ``season_map`` / hooks. Rows with an unmapped season are left unchanged.

    Downstream (``add_edvise_term_labels``) always treats ``_year`` as the calendar year, so
    this conversion keeps academic-year labels and term ordering consistent across encodings.
    """
    if year_semantics in (None, "calendar_literal"):
        return year
    canonical = season_norm.astype("string").str.lower().map(raw_to_canonical)
    rolls_forward = canonical.notna() & ~canonical.isin(_ACADEMIC_YEAR_START_SEASONS)
    return year.where(~rolls_forward, other=year + 1)


def _finalize_season_year_order(
    out: pd.DataFrame,
    season_norm: pd.Series,
    raw_to_rank: dict[str, int],
    cols: EdviseTermColumnSet,
    *,
    raw_to_canonical: dict[str, str] | None = None,
    year_semantics: str | None = None,
) -> pd.DataFrame:
    out = out.copy()
    out[cols.season] = season_norm.astype("string")

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

    out[cols.year] = _calendar_year_from_semantics(
        out[cols.year], season_norm, raw_to_canonical or {}, year_semantics
    )

    season_rank = season_norm.map(raw_to_rank).astype("Int64")
    out[cols.term_order] = (out[cols.year] * 100 + season_rank).astype("Int64")
    return out


def add_edvise_term_order(
    df: pd.DataFrame,
    term_config: dict,
    year_extractor: Callable[..., Any] | None = None,
    season_extractor: Callable[..., Any] | None = None,
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

        Also ``season_map``, ``term_extraction`` (``standard`` | ``hook_required``), and optional
        ``year_semantics`` (``calendar_literal`` default | ``academic_year_prefix``) controlling
        whether the extracted year is treated as the calendar year or an academic-year start
        (SPRING/SUMMER roll forward one calendar year).
    year_extractor : callable | None
        Required when term_config["term_extraction"] == "hook_required" (combined ``term_col`` only).
    season_extractor : callable | None
        Required when term_config["term_extraction"] == "hook_required" (combined ``term_col`` only).

    Returns
    -------
    pd.DataFrame with added columns (names depend on optional ``output_prefix`` in dict — see
    :func:`edvise_term_column_set`).
    """
    cols = edvise_term_column_set(term_config.get("output_prefix"))
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

    if term_extraction == "hook_required":
        if has_split:
            raise ValueError(
                "term_extraction 'hook_required' is not supported when year_col and season_col are set."
            )
        if year_extractor is None or season_extractor is None:
            raise ValueError(
                "term_extraction is 'hook_required' but year_extractor and/or season_extractor not provided. "
                "Use load_term_extractors_from_hook_spec(hook_spec, modules_root=...) or pass callables."
            )
    elif has_split and (year_extractor is not None or season_extractor is not None):
        raise ValueError(
            "year_extractor and season_extractor are only used with term_extraction 'hook_required' "
            "and a combined term_col."
        )

    raw_to_rank, norm_keys = _season_map_lookups(season_map)
    raw_to_canonical = {
        item["raw"].lower(): item["canonical"].upper() for item in season_map
    }
    year_semantics = term_config.get("year_semantics")
    out = df.copy()

    exclude_tokens = [
        str(t).strip()
        for t in (term_config.get("exclude_tokens") or [])
        if str(t).strip()
    ]
    if exclude_tokens:
        if has_split:
            excl = _is_excluded_by_term_prefix(
                out[season_col], exclude_tokens
            ) | _is_excluded_by_term_prefix(out[year_col], exclude_tokens)
            out = out.loc[~excl].copy()
        else:
            excl = _is_excluded_by_term_prefix(out[term_col], exclude_tokens)
            out = out.loc[~excl].copy()

    if has_split:
        for c in (year_col, season_col):
            if c not in out.columns:
                raise KeyError(f"DataFrame must contain column '{c}'")
        out[cols.year] = _calendar_year_from_column(out[year_col])
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
        ordered = _finalize_season_year_order(
            out,
            season_norm,
            raw_to_rank,
            cols,
            raw_to_canonical=raw_to_canonical,
            year_semantics=year_semantics,
        )
        ordered = _add_term_grain(ordered, cols)
        return add_edvise_term_labels(ordered, term_config, columns=cols)

    # --- Combined term_col path ---
    if term_col not in out.columns:
        raise KeyError(f"DataFrame must contain column '{term_col}'")

    s = out[term_col].astype("string").str.strip()

    if year_extractor is not None:
        year_str = s.apply(lambda t: str(year_extractor(t)) if pd.notna(t) else None)
    else:
        year_str = s.str.extract(r"(\d{4})", expand=False)

    out[cols.year] = pd.to_numeric(year_str, errors="coerce").astype("Int64")

    if season_extractor is not None:
        # Extractor returns the raw season fragment; normalize to season_map keys (lowercase).
        season_norm = s.apply(
            lambda t: _norm_token(season_extractor(t)) if pd.notna(t) else None
        )
    else:

        def _extract_season(term: str | None) -> str | None:
            return _resolve_season_token(_norm_token(term), norm_keys)

        season_norm = s.apply(_extract_season)

    ordered = _finalize_season_year_order(
        out,
        season_norm,
        raw_to_rank,
        cols,
        raw_to_canonical=raw_to_canonical,
        year_semantics=year_semantics,
    )
    ordered = _add_term_grain(ordered, cols)
    return add_edvise_term_labels(ordered, term_config, columns=cols)


def add_edvise_term_labels(
    df: pd.DataFrame,
    term_config: dict,
    *,
    columns: EdviseTermColumnSet | None = None,
) -> pd.DataFrame:
    """
    Adds Edvise standard term label columns to a DataFrame.
    Expects year and season columns (e.g. from :func:`add_edvise_term_order`, which
    invokes this function automatically).

    Calendar year is the work ``year`` column only; there is no separate ``_edvise_term_year``.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame, must contain year and season work columns.
    term_config : dict
        Term config emitted by IdentityAgent. Expected keys:
            season_map : list[{"raw": str, "canonical": str}]
    columns : EdviseTermColumnSet | None
        Column names for this stream; defaults to entry (legacy ``_year`` / ``_season``).
    """
    cols = columns or edvise_term_column_set(None)
    for col in (cols.year, cols.season):
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

    # edvise_season — map raw token to canonical label
    out[cols.edvise_season] = (
        out[cols.season]
        .astype("string")
        .str.lower()
        .map(raw_to_canonical)
        .astype("string")
    )

    # Warn on unmapped seasons
    unmapped = out.loc[
        out[cols.season].notna() & out[cols.edvise_season].isna(), cols.season
    ].unique()
    if len(unmapped) > 0:
        logger.warning("Unmapped season tokens in standardization: %s", unmapped)

    # edvise_academic_year
    # FALL/WINTER of year N -> "N-(N+1 2-digit)"
    # SPRING/SUMMER of year N -> "(N-1)-(N 2-digit)"
    def _academic_year(row: pd.Series) -> str | pd.NA:
        year = row[cols.year]
        season = row[cols.edvise_season]
        if pd.isna(year) or pd.isna(season):
            return pd.NA
        year = int(year)
        if season in _ACADEMIC_YEAR_START_SEASONS:
            return f"{year}-{str(year + 1)[-2:]}"
        return f"{year - 1}-{str(year)[-2:]}"

    out[cols.edvise_academic_year] = out.apply(_academic_year, axis=1).astype("string")

    return out


def _function_names_for_year_season_resolution(
    functions: list[Any],
) -> tuple[list[str], list[str]]:
    """
    Classify ``HookSpec.functions``-shaped rows into year-role and season-role names.

    Prefer canonical prefixes ``year_extractor`` / ``season_extractor`` (case-insensitive) so a
    year function whose slug contains ``..._date_season_...`` is not misclassified: those names
    match both substring rules ``year`` and ``season``, which would otherwise exclude them from
    both roles.

    Fallback: exactly one name containing ``year`` but not listed as a season substring candidate,
    and vice versa (legacy hooks without the extractor prefixes).
    """
    names: list[str] = []
    for f in functions:
        if isinstance(f, dict):
            n = f.get("name")
        else:
            n = getattr(f, "name", None)
        if isinstance(n, str) and n.strip():
            names.append(n)

    y_pref = [n for n in names if n.lower().startswith("year_extractor")]
    s_pref = [n for n in names if n.lower().startswith("season_extractor")]
    if len(y_pref) == 1 and len(s_pref) == 1:
        return y_pref, s_pref

    year_like = [n for n in names if "year" in n.lower()]
    season_like = [n for n in names if "season" in n.lower()]
    year_names = [n for n in year_like if n not in season_like]
    season_names = [n for n in season_like if n not in year_like]
    return year_names, season_names


def resolve_year_season_hook_function_names(
    hook_spec: HookSpec | dict[str, Any],
) -> tuple[str, str]:
    """
    Return ``(year_extractor_function_name, season_extractor_function_name)`` from ``hook_spec``.

    Used by :func:`load_term_extractors_from_hook_spec` and by hook-generation validation so
    malformed specs fail before materialize/apply.
    """
    hs = (
        hook_spec.model_dump(mode="json")
        if isinstance(hook_spec, HookSpec)
        else dict(hook_spec)
    )
    year_names, season_names = _function_names_for_year_season_resolution(
        hs.get("functions") or []
    )
    if len(year_names) != 1 or len(season_names) != 1:
        raise ValueError(
            "hook_spec.functions must name exactly one function with 'year' and one with 'season' "
            f"in the identifier (disambiguated when both appear); got year-like {year_names!r}, "
            f"season-like {season_names!r}"
        )
    return year_names[0], season_names[0]


def load_term_extractors_from_hook_spec(
    hook_spec: HookSpec | dict[str, Any],
    *,
    modules_root: str | Path,
) -> tuple[Callable[..., Any], Callable[..., Any]]:
    """
    Import the materialized hook module and return ``(year_extractor, season_extractor)`` callables.

    Chooses functions by name: prefers ``year_extractor*`` / ``season_extractor*`` prefixes; else
    exactly one ``name`` containing ``year`` and one containing ``season`` (case-insensitive),
    excluding names that appear in both substring lists. Typical names are
    ``year_extractor_<slug>`` and ``season_extractor_<slug>`` (slugs may contain ``date_season``).
    """
    from edvise.genai.mapping.shared.hitl.hook_spec.paths import (
        resolve_hook_module_path,
    )

    hs = (
        hook_spec.model_dump(mode="json")
        if isinstance(hook_spec, HookSpec)
        else dict(hook_spec)
    )
    rel = hs.get("file")
    if not rel:
        raise ValueError("hook_spec.file is required to load extractors from disk")
    path = resolve_hook_module_path(rel, root=modules_root)
    spec = importlib.util.spec_from_file_location("_ia_term_hooks", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load hook module spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    yn, sn = resolve_year_season_hook_function_names(hook_spec)
    y = getattr(mod, yn, None)
    s = getattr(mod, sn, None)
    if not callable(y) or not callable(s):
        raise ValueError(f"Module {path} missing callables {yn!r} / {sn!r}")
    return y, s


def _resolve_hook_year_season_callables(
    config: TermOrderConfig,
    *,
    hook_modules_root: str | Path | None,
    year_extractor: Callable[..., Any] | None,
    season_extractor: Callable[..., Any] | None,
) -> tuple[Callable[..., Any] | None, Callable[..., Any] | None]:
    if config.term_extraction != "hook_required":
        return None, None
    if year_extractor is not None and season_extractor is not None:
        return year_extractor, season_extractor
    if year_extractor is not None or season_extractor is not None:
        raise ValueError(
            "Pass both year_extractor and season_extractor, or pass neither to load from hook_spec."
        )
    if config.hook_spec is None:
        raise ValueError(
            "hook_spec is required when term_extraction is 'hook_required'"
        )
    if hook_modules_root is None:
        raise ValueError(
            "term_extraction is 'hook_required': pass hook_modules_root= (directory containing "
            "hook_spec.file, e.g. bronze_volumes_path) to load extractors, or pass year_extractor= "
            "and season_extractor= explicitly."
        )
    return load_term_extractors_from_hook_spec(
        config.hook_spec, modules_root=hook_modules_root
    )


def term_normalization_summary_for_enriched_contract(
    config: TermOrderConfig,
) -> TermNormalizationSummary:
    """
    Build :class:`~edvise.genai.mapping.shared.schema_contract.schemas.TermNormalizationSummary`
    for IdentityAgent enriched schema contracts (same column names as after ``clean_dataset``).
    """
    d = _normalize_term_config_column_names(config.model_dump(mode="json"))
    clean_spec_term = term_order_column_for_clean_dataset(config)
    if config.term_col is not None:
        mode: Literal["single_column", "year_season_columns"] = "single_column"
    else:
        mode = "year_season_columns"
    return TermNormalizationSummary(
        stream_role="entry",
        materialized_column_prefix=None,
        mode=mode,
        term_extraction=config.term_extraction,
        term_col=d.get("term_col"),
        year_col=d.get("year_col"),
        season_col=d.get("season_col"),
        clean_spec_term_column=clean_spec_term,
    )


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
    config: TermOrderConfig | Mapping[str, Any],
    *,
    hook_modules_root: str | Path | None = None,
    year_extractor: Callable[..., Any] | None = None,
    season_extractor: Callable[..., Any] | None = None,
) -> Callable[[pd.DataFrame, str], pd.DataFrame]:
    """
    Build a ``(df, term_column) -> df`` hook compatible with
    :class:`~edvise.data_audit.custom_cleaning.TermOrderFn` / ``clean_dataset``.

    Pass the result as ``term_order_fn`` and set ``term_column`` to
    :func:`term_order_column_for_clean_dataset`\\(config\\) so the optional term-order step runs.

    When ``term_extraction`` is ``hook_required``, pass ``hook_modules_root`` (same root used when
    materializing hooks, e.g. ``bronze_volumes_path``) so extractors are imported from
    ``hook_spec.file``, **or** pass ``year_extractor`` and ``season_extractor`` explicitly.

    For ``hook_required``, extractors are loaded on the **first** invocation of the returned
    function (unless explicit extractors are passed), so constructing the callable does not
    import the hook module; ``standard`` extraction never uses ``hook_modules_root``.
    """
    if not isinstance(config, TermOrderConfig):
        config = TermOrderConfig.model_validate(dict(config))

    if config.term_extraction == "hook_required":
        if year_extractor is not None or season_extractor is not None:
            if year_extractor is None or season_extractor is None:
                raise ValueError(
                    "Pass both year_extractor and season_extractor, or pass neither to load from hook_spec."
                )
    elif year_extractor is not None or season_extractor is not None:
        raise ValueError(
            "year_extractor and season_extractor are only used when term_extraction is 'hook_required'"
        )

    explicit_ye = year_extractor
    explicit_se = season_extractor
    cached_ye: Callable[..., Any] | None = None
    cached_se: Callable[..., Any] | None = None

    expected_column = term_order_column_for_clean_dataset(config)
    tc = _normalize_term_config_column_names(config.model_dump(mode="json"))

    def _fn(df: pd.DataFrame, term_column: str) -> pd.DataFrame:
        nonlocal cached_ye, cached_se
        if term_column != expected_column:
            raise ValueError(
                f"term_column {term_column!r} must be {expected_column!r} for this TermOrderConfig "
                "(use term_order_column_for_clean_dataset(config) when building CleanSpec)."
            )
        ye: Callable[..., Any] | None
        se: Callable[..., Any] | None
        if config.term_extraction != "hook_required":
            ye, se = None, None
        elif explicit_ye is not None and explicit_se is not None:
            ye, se = explicit_ye, explicit_se
        else:
            if cached_ye is None:
                cached_ye, cached_se = _resolve_hook_year_season_callables(
                    config,
                    hook_modules_root=hook_modules_root,
                    year_extractor=None,
                    season_extractor=None,
                )
            ye, se = cached_ye, cached_se
        return add_edvise_term_order(df, tc, year_extractor=ye, season_extractor=se)

    return _fn


def apply_term_order_from_config(
    df: pd.DataFrame,
    config: TermOrderConfig,
    *,
    hook_modules_root: str | Path | None = None,
    year_extractor: Callable[..., Any] | None = None,
    season_extractor: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    """
    Apply a validated :class:`~edvise.genai.mapping.identity_agent.term_normalization.schemas.TermOrderConfig`
    by calling :func:`add_edvise_term_order` with ``config`` serialized as the JSON-compatible dict
    IdentityAgent emits (including :func:`add_edvise_term_labels`).

    When ``term_extraction`` is ``hook_required``, pass ``hook_modules_root`` or explicit extractors
    (same as :func:`term_order_fn_from_term_order_config`).
    """
    ye, se = _resolve_hook_year_season_callables(
        config,
        hook_modules_root=hook_modules_root,
        year_extractor=year_extractor,
        season_extractor=season_extractor,
    )
    tc = _normalize_term_config_column_names(config.model_dump(mode="json"))
    return add_edvise_term_order(df, tc, year_extractor=ye, season_extractor=se)
