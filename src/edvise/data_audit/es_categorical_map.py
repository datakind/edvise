"""Canonicalize ES free-text categoricals onto PDP closed sets before dummy encoding.

PDP dummy vocabularies are closed (delivery F/O/H, gateway E/M/NA, instructor FT/PT,
core Y/N, grade status codes). ES keeps unconstrained strings. ``get_dummies`` then
mints ``num_courses_<col>_<snake_label>`` columns that miss ``features_table.toml``.

Apply these maps at ES course validation (and again in feature generation) so
modeling columns stay on the PDP set. Unmapped labels become null and are skipped
by ``get_dummies(..., dummy_na=False)``. Dummy-suffix aliases below keep historical
already-encoded columns resolvable at features-table lookup.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edvise.utils.data_cleaning import convert_to_snake_case

# Do not include "na" — it is a valid PDP gateway code.
_NULL_TOKENS = frozenset({"", "nan", "none", "null", "<na>", "<n/a>", "n/a"})

DELIVERY_METHOD_CATEGORIES: tuple[str, str, str] = ("F", "O", "H")

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

INSTRUCTOR_APPOINTMENT_TO_PDP: dict[str, str] = {
    "ft": "FT",
    "pt": "PT",
    "full_time": "FT",
    "fulltime": "FT",
    "full": "FT",
    "tenured": "FT",
    "tenure_track": "FT",
    "salaried": "FT",
    "part_time": "PT",
    "parttime": "PT",
    "part": "PT",
    "adjunct": "PT",
    "contingent": "PT",
    "hourly": "PT",
    "affiliate": "PT",
}

GATEWAY_FLAG_TO_PDP: dict[str, str] = {
    "e": "E",
    "m": "M",
    "na": "NA",
    "english": "E",
    "eng": "E",
    "gateway_english": "E",
    "gateway_or_english": "E",
    "math": "M",
    "mathematics": "M",
    "gateway_math": "M",
    "n_a": "NA",
    "not_applicable": "NA",
    "not_gateway": "NA",
    "non_gateway": "NA",
    "neither": "NA",
    "developmental": "NA",
    "dev": "NA",
    "remedial": "NA",
}

YES_NO_TO_PDP: dict[str, str] = {
    "y": "Y",
    "n": "N",
    "yes": "Y",
    "no": "N",
    "true": "Y",
    "false": "N",
    "t": "Y",
    "f": "N",
    "1": "Y",
    "0": "N",
}

# Dummy suffixes after course_grade derivation (not raw grade cells).
COURSE_GRADE_DUMMY_ALIASES: dict[str, str] = {
    "pass": "p",
    "sat": "p",
    "s": "p",
    "unsat": "f",
    "u": "f",
    "wd": "w",
    "ip": "i",
    "nr": "m",
    "ng": "m",
}

_ES_COURSE_COLUMN_MAPS: tuple[tuple[str, dict[str, str]], ...] = (
    ("instructional_modality", INSTRUCTIONAL_MODALITY_TO_PDP),
    ("instructor_appointment_status", INSTRUCTOR_APPOINTMENT_TO_PDP),
    ("gateway_or_developmental_flag", GATEWAY_FLAG_TO_PDP),
    ("gen_ed_flag", YES_NO_TO_PDP),
    ("prerequisite_flag", YES_NO_TO_PDP),
    ("intent_to_transfer_flag", YES_NO_TO_PDP),
)


def _to_pdp_val(val: Any, mapping: dict[str, str]) -> str | None:
    if pd.isna(val):
        return None
    if not isinstance(val, str):
        val = str(val)
    val = val.strip()
    if not val:
        return None
    key = convert_to_snake_case(val)
    if key in mapping:
        return mapping[key]
    if val.lower() in _NULL_TOKENS:
        return None
    return None


def _series_to_pdp(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.apply(lambda v: _to_pdp_val(v, mapping)).astype(pd.StringDtype())


def _dummy_aliases(mapping: dict[str, str], *, skip: set[str]) -> dict[str, str]:
    return {key: code.lower() for key, code in mapping.items() if key not in skip}


def instructional_modality_to_pdp_val(val: Any) -> str | None:
    """Map one raw modality label to ``F`` / ``O`` / ``H``, or ``None`` if unknown."""
    return _to_pdp_val(val, INSTRUCTIONAL_MODALITY_TO_PDP)


def instructional_modality_series_to_pdp(series: pd.Series) -> pd.Series:
    """Map ES instructional_modality strings to PDP delivery_method codes."""
    return _series_to_pdp(series, INSTRUCTIONAL_MODALITY_TO_PDP)


def instructor_appointment_series_to_pdp(series: pd.Series) -> pd.Series:
    """Map ES instructor_appointment_status to PDP ``FT`` / ``PT``."""
    return _series_to_pdp(series, INSTRUCTOR_APPOINTMENT_TO_PDP)


def gateway_flag_series_to_pdp(series: pd.Series) -> pd.Series:
    """Map ES gateway_or_developmental_flag to PDP ``E`` / ``M`` / ``NA``."""
    return _series_to_pdp(series, GATEWAY_FLAG_TO_PDP)


def yes_no_series_to_pdp(series: pd.Series) -> pd.Series:
    """Map ES Y/N-like flags (gen ed, prerequisite, intent to transfer) to ``Y`` / ``N``."""
    return _series_to_pdp(series, YES_NO_TO_PDP)


def canonicalize_es_course_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite ES free-text course categoricals in place when those columns exist."""
    out = df
    assigned: dict[str, pd.Series] = {}
    for col, mapping in _ES_COURSE_COLUMN_MAPS:
        if col in df.columns:
            assigned[col] = _series_to_pdp(df[col], mapping)
    if not assigned:
        return out
    return out.assign(**assigned)


def es_modality_dummy_aliases() -> dict[str, str]:
    """Dummy-suffix aliases for already-encoded instructional_modality columns."""
    return _dummy_aliases(INSTRUCTIONAL_MODALITY_TO_PDP, skip={"f", "o", "h"})


def es_dummy_value_aliases() -> dict[str, str]:
    """All ES dummy-suffix aliases for features-table lookup."""
    return {
        **COURSE_GRADE_DUMMY_ALIASES,
        **es_modality_dummy_aliases(),
        **_dummy_aliases(INSTRUCTOR_APPOINTMENT_TO_PDP, skip={"ft", "pt"}),
        **_dummy_aliases(GATEWAY_FLAG_TO_PDP, skip={"e", "m", "na"}),
        # Skip single-letter Y/N tokens so "_f" cannot rewrite modality/grade dummies.
        **_dummy_aliases(YES_NO_TO_PDP, skip={"y", "n", "t", "f", "1", "0"}),
    }
