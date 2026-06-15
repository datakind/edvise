"""
Platform-default ES course grade mappings.

Two layers merged before institution ``preprocessing.features.grade_map`` (school
entries override on duplicate keys):

1. **Status / withdrawal tokens** — common raw codes (``W1``, ``OTH``, ``NC``, …).
2. **Letter → GPA** — standard 4.0 scale including plus/minus (``A-`` → ``3.7``, …).

Schools with different scales override specific letters in bronze ``config.toml``.
Do not map pass/fail tokens (``P``) to GPA.
"""

from __future__ import annotations

# Common raw tokens seen across ES uploads (Kayla schema-error examples + HCC).
DEFAULT_ES_STATUS_GRADE_MAP: dict[str, str] = {
    "W1": "W",
    "W2": "W",
    "NSW": "W",
    "OTHER": "O",
    "OTH": "O",
    "NC": "NR",
    "NCR": "NR",
    "AUD": "AU",
    "COVID_I": "I",
    "IP": "I",
    "ADW": "WD",
    "MP": "M",
}

# Standard US 4.0 letter scale (institution grade_map overrides per letter).
DEFAULT_ES_LETTER_GPA_MAP: dict[str, str] = {
    "A+": "4",
    "A": "4",
    "A-": "3.7",
    "B+": "3.3",
    "B": "3",
    "B-": "2.7",
    "C+": "2.3",
    "C": "2",
    "C-": "1.7",
    "D+": "1.3",
    "D": "1",
    "D-": "0.7",
    "F": "0",
}

DEFAULT_ES_GRADE_MAP: dict[str, str] = {
    **DEFAULT_ES_STATUS_GRADE_MAP,
    **DEFAULT_ES_LETTER_GPA_MAP,
}

# Letter grades expected to map to a numeric GPA string after the full grade_map.
LETTER_GPA_GRADE_CODES: frozenset[str] = frozenset(DEFAULT_ES_LETTER_GPA_MAP)

# Allowed status / non-GPA codes that should not trigger unmapped-GPA warnings.
NON_GPA_STATUS_GRADE_CODES: frozenset[str] = frozenset(
    {
        "P",
        "PASS",
        "S",
        "SAT",
        "U",
        "UNSAT",
        "W",
        "WD",
        "I",
        "IP",
        "AU",
        "NG",
        "NR",
        "M",
        "O",
    }
)
