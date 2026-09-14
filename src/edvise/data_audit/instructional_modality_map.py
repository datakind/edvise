"""Canonicalize ES ``instructional_modality`` onto PDP ``delivery_method`` codes.

PDP ``delivery_method`` is the closed set F (face-to-face), O (online), H (hybrid).
ES schools send free-text labels; dummy-encoding those raw strings produces
unstable ``num_courses_instructional_modality_*`` columns that miss the shared
features table. Map values here *before* feature generation so dummies stay
``_f`` / ``_o`` / ``_h``. Unmapped labels become null and are skipped by
``get_dummies(..., dummy_na=False)``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edvise.utils.data_cleaning import convert_to_snake_case

DELIVERY_METHOD_CATEGORIES: tuple[str, str, str] = ("F", "O", "H")

# Snake-case keys (via ``convert_to_snake_case``) → PDP delivery_method code.
INSTRUCTIONAL_MODALITY_TO_PDP: dict[str, str] = {
    "f": "F",
    "o": "O",
    "h": "H",
    "face_to_face": "F",
    "face_to_face_only": "F",
    "in_person": "F",
    "inperson": "F",
    "f2f": "F",
    "on_campus": "F",
    "on_ground": "F",
    "classroom": "F",
    "online": "O",
    "online_only": "O",
    "fully_online": "O",
    "web_based": "O",
    "web": "O",
    "internet": "O",
    "online_internet_or_web": "O",
    "distance": "O",
    "distance_learning": "O",
    "asynchronous": "O",
    "virtual": "O",
    "remote": "O",
    "e_learning": "O",
    "elearning": "O",
    "hybrid": "H",
    "blended": "H",
    "blended_learning": "H",
    "face_to_face_and_online": "H",
    "in_person_and_online": "H",
    "mixed": "H",
    "mixed_mode": "H",
    "hyflex": "H",
    "hybrid_online": "H",
    "partially_online": "H",
}

_NULL_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "<n/a>", "n/a", "na"})


def instructional_modality_to_pdp_val(val: Any) -> str | None:
    """Map one raw modality label to ``F`` / ``O`` / ``H``, or ``None`` if unknown."""
    if pd.isna(val):
        return None
    if not isinstance(val, str):
        val = str(val)
    val = val.strip()
    if not val or val.lower() in _NULL_TOKENS:
        return None
    return INSTRUCTIONAL_MODALITY_TO_PDP.get(convert_to_snake_case(val))


def instructional_modality_series_to_pdp(series: pd.Series) -> pd.Series:
    """
    Map ES instructional_modality strings to PDP delivery_method codes.

    Args:
        series: Raw labels (e.g. "Face to Face and Online", "web-based", "O").

    Returns:
        String series with ``F``, ``O``, or ``H``, or ``pd.NA`` where unmapped.
    """
    return series.apply(instructional_modality_to_pdp_val).astype(pd.StringDtype())


def es_modality_dummy_aliases() -> dict[str, str]:
    """
    Dummy-suffix aliases for features-table lookup on already-encoded columns.

    Keys are snake-case labels (without the single-letter PDP codes). Values are
    lowercase ``f`` / ``o`` / ``h`` so ``num_courses_instructional_modality_web_based``
    still resolves after historical runs that dummy-encoded raw strings.
    """
    return {
        key: code.lower()
        for key, code in INSTRUCTIONAL_MODALITY_TO_PDP.items()
        if key not in {"f", "o", "h"}
    }
