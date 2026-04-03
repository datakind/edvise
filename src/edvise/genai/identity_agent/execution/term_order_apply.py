"""Apply IdentityAgent ``term_config`` using :func:`~edvise.feature_generation.term.add_term_order`."""

from __future__ import annotations

import logging

import pandas as pd

from edvise.feature_generation import constants
from edvise.feature_generation.term import add_term_order
from edvise.genai.identity_agent.grain_inference.schemas import TermOrderConfig

logger = logging.getLogger(__name__)


def _build_season_order_map(canonical_mapping: dict[str, str]) -> dict[str, int]:
    """Map short codes (e.g. ``FA``, ``SP``) to season sort order via canonical names (``FALL`` → ``fall``)."""
    norm = constants.DEFAULT_SEASON_ORDER_MAP
    out: dict[str, int] = {}
    for short, long_name in canonical_mapping.items():
        sk = short.strip().lower()
        lk = long_name.strip().lower()
        order = norm.get(lk)
        if order is None:
            first = lk.split()[0] if lk else ""
            order = norm.get(first)
        if order is None:
            raise ValueError(
                f"term_config.canonical_mapping: cannot derive season order for {long_name!r} "
                f"(from short code {short!r}); expected a name matching {set(norm.keys())!r}"
            )
        out[sk] = int(order)
    return out


def _add_term_canonical(
    g: pd.DataFrame, term_col: str, canonical_mapping: dict[str, str]
) -> pd.DataFrame:
    upper_map = {k.strip().upper(): v for k, v in canonical_mapping.items()}

    def canon(val: object) -> object:
        if pd.isna(val):
            return pd.NA
        s = str(val).strip()
        if len(s) < 2:
            return pd.NA
        suf = s[-2:].upper()
        return upper_map.get(suf, pd.NA)

    out = g.copy()
    out["term_canonical"] = out[term_col].map(canon)
    return out


def _add_term_academic_year(g: pd.DataFrame) -> pd.DataFrame:
    """Academic year label from ``year`` + ``season`` (e.g. Fall 2018 → ``2018-19``)."""
    if "year" not in g.columns or "season" not in g.columns:
        return g
    out = g.copy()

    def ay(row: pd.Series) -> object:
        y = row["year"]
        season = str(row["season"]).strip().lower() if pd.notna(row["season"]) else ""
        if pd.isna(y):
            return pd.NA
        yi = int(y)
        if season.startswith("fall") or season.startswith("f"):
            return f"{yi}-{str(yi + 1)[-2:]}"
        return f"{yi - 1}-{str(yi)[-2:]}"

    out["term_academic_year"] = out.apply(ay, axis=1)
    return out


def apply_term_order_from_config(df: pd.DataFrame, config: TermOrderConfig) -> pd.DataFrame:
    """
    Run :func:`~edvise.feature_generation.term.add_term_order` with ``canonical_mapping``-derived
    ``season_order_map`` (for ``YYYYTT``-style codes), then optional ``term_canonical`` /
    ``term_academic_year`` columns per ``config.outputs``.
    """
    col = config.term_column
    if col not in df.columns:
        raise ValueError(f"term_config.term_column {col!r} not in DataFrame columns")

    g = df.copy()
    if config.unmapped_values:
        bad = g[col].isin(config.unmapped_values)
        n_bad = int(bad.sum())
        if n_bad:
            logger.warning(
                "Dropping %s rows whose %r is in term_config.unmapped_values",
                n_bad,
                col,
            )
            g = g.loc[~bad].copy()

    season_order_map = (
        _build_season_order_map(config.canonical_mapping) if config.canonical_mapping else None
    )

    if config.term_format != "YYYYTT":
        logger.warning(
            "term_config.term_format=%r — only YYYYTT is fully supported; using default parsing",
            config.term_format,
        )

    g = add_term_order(g, term_col=col, season_order_map=season_order_map)

    flags = config.outputs
    if flags.term_canonical and config.canonical_mapping:
        g = _add_term_canonical(g, col, config.canonical_mapping)
    if flags.term_academic_year:
        g = _add_term_academic_year(g)
    if not flags.term_sort_key and "term_order" in g.columns:
        g = g.drop(columns=["term_order"])

    return g
