"""
Four-digit academic term codes and human-readable **display** strings.

Many student-information systems expose a term as four decimal digits ``D0 D1 D2 D3``
plus optional labeling (e.g. ``"1179 Fall"`` — code, space, season word).

**Default year rule:** calendar year of the term is ``year_base + int(D1 D2)`` (middle
two digits; leading digit ``D0`` often unused in that calculation).

**Default season digit ``D3``:** ``2`` Spring, ``6`` Summer, ``9`` Fall.

**Display form:** values may appear as ``"{code} {Season}"`` after ETL. These helpers
parse that shape (or bare codes) for gen-ai / raw PDP transforms: **academic_year**
``YYYY-YY`` (US convention: Fall starts the AY; Spring/Summer map to the AY that began
the previous fall) and **canonical term category**
(``FALL`` / ``SPRING`` / ``SUMMER`` / ``WINTER``).

Override :class:`TermCodeDisplayEncoding` when your institution uses different digit
semantics (same general *code + display* shape).

**Edvise genai:** :func:`academic_year_from_term_code_display` and
:func:`academic_term_category_from_term_code_display` are registered in
``schema_mapping_agent.execution.step_dispatcher``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_SEASON_WORD_TO_CAT = {
    "fall": "FALL",
    "spring": "SPRING",
    "summer": "SUMMER",
    "winter": "WINTER",
}

_DEFAULT_COURSE_TERM_CATS = ("FALL", "SPRING", "SUMMER", "WINTER")


@dataclass(frozen=True)
class TermCodeDisplayEncoding:
    """
    Rules for parsing a fixed-width **numeric term code** and optional display suffix.

    Defaults match one widely used 4-digit layout (middle-two year + 2000, season in
    last digit). Override fields for institutions that differ.
    """

    code_width: int = 4
    year_fragment_slice: tuple[int, int] = (1, 3)
    year_base: int = 2000
    season_digit_index: int = 3
    season_digit_to_category: Mapping[str, str] = field(
        default_factory=lambda: {"9": "FALL", "2": "SPRING", "6": "SUMMER"}
    )
    fall_starts_academic_year_digits: frozenset[str] = frozenset({"9"})
    spring_summer_academic_year_digits: frozenset[str] = frozenset({"2", "6"})
    canonical_term_categories: tuple[str, ...] = _DEFAULT_COURSE_TERM_CATS


DEFAULT_TERM_CODE_DISPLAY_ENCODING = TermCodeDisplayEncoding()


def calendar_year_from_term_code(
    code: str,
    encoding: TermCodeDisplayEncoding = DEFAULT_TERM_CODE_DISPLAY_ENCODING,
) -> int | None:
    """
    Map a fixed-width numeric **code** string to the calendar year of term start.

    Uses ``encoding.year_base + int(code[slice])`` for ``encoding.year_fragment_slice``.
    """
    if not code or len(code) != encoding.code_width or not code.isdigit():
        return None
    start, end = encoding.year_fragment_slice
    frag = code[start:end]
    if not frag.isdigit():
        return None
    try:
        return int(frag) + encoding.year_base
    except ValueError:
        return None


def _scalar_to_code(val: Any, encoding: TermCodeDisplayEncoding) -> str | None:
    width = encoding.code_width
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    if re.fullmatch(r"-?\d+\.0", s):
        try:
            s = str(int(float(s)))
        except ValueError:
            pass
    elif s.replace(".", "", 1).isdigit() and "." in s:
        try:
            f = float(s)
            if f == int(f):
                s = str(int(f))
        except ValueError:
            pass
    m = re.compile(rf"^(\d{{{width}}})\b").match(s)
    if not m:
        return None
    code = m.group(1)
    if len(code) != width or not code.isdigit():
        return None
    return code


def _academic_year_label_from_code(
    code: str,
    encoding: TermCodeDisplayEncoding,
) -> Any:
    if len(code) != encoding.code_width:
        return pd.NA
    if encoding.season_digit_index >= len(code):
        return pd.NA
    season_digit = code[encoding.season_digit_index]
    allowed = (
        encoding.fall_starts_academic_year_digits
        | encoding.spring_summer_academic_year_digits
    )
    if season_digit not in allowed:
        return pd.NA
    cy = calendar_year_from_term_code(code, encoding)
    if cy is None:
        return pd.NA
    if season_digit in encoding.fall_starts_academic_year_digits:
        y_end = (cy + 1) % 100
        return f"{cy}-{y_end:02d}"
    if season_digit in encoding.spring_summer_academic_year_digits:
        y_start = cy - 1
        y_end_two = cy % 100
        return f"{y_start}-{y_end_two:02d}"
    return pd.NA


def _season_category_from_code_and_text(
    code: str,
    full: str,
    encoding: TermCodeDisplayEncoding,
) -> Any:
    tail = full[len(code) :].strip() if full else ""
    if tail:
        word = tail.split()[0].lower()
        if word in _SEASON_WORD_TO_CAT:
            return _SEASON_WORD_TO_CAT[word]
    if encoding.season_digit_index >= len(code):
        return pd.NA
    d = code[encoding.season_digit_index]
    return encoding.season_digit_to_category.get(d, pd.NA)


def academic_year_from_term_code_display(
    series: pd.Series,
    encoding: TermCodeDisplayEncoding = DEFAULT_TERM_CODE_DISPLAY_ENCODING,
) -> pd.Series:
    """
    Map a **term** column (display strings or bare codes) to **academic_year** ``YYYY-YY``.
    """
    out: list[Any] = []
    for v in series:
        code = _scalar_to_code(v, encoding)
        if code is None:
            out.append(pd.NA)
            continue
        out.append(_academic_year_label_from_code(code, encoding))
    return pd.Series(out, index=series.index, dtype="string")


def academic_term_category_from_term_code_display(
    series: pd.Series,
    encoding: TermCodeDisplayEncoding = DEFAULT_TERM_CODE_DISPLAY_ENCODING,
) -> pd.Series:
    """
    Map term display strings to **FALL** / **SPRING** / **SUMMER** / **WINTER**.
    """
    cats = encoding.canonical_term_categories
    cat_dtype = pd.CategoricalDtype(categories=list(cats))
    out: list[Any] = []
    for v in series:
        if v is None or (isinstance(v, float) and pd.isna(v)) or pd.isna(v):
            out.append(pd.NA)
            continue
        full = str(v).strip()
        code = _scalar_to_code(full, encoding)
        if code is None:
            out.append(pd.NA)
            continue
        out.append(_season_category_from_code_and_text(code, full, encoding))
    return pd.Series(out, index=series.index, dtype=cat_dtype)
