"""
Platform-default ES course grade token mappings (status / withdrawal codes).

Letter-to-GPA mappings (``A`` -> ``"4"``, etc.) are **institution-specific** and belong
in bronze ``config.toml`` ``preprocessing.features.grade_map`` — not here — because
schools use different scales and plus/minus rules.

Institution ``grade_map`` entries override these defaults when keys collide.
"""

from __future__ import annotations

# Common raw tokens seen across ES uploads (Kayla schema-error examples + HCC).
DEFAULT_ES_GRADE_MAP: dict[str, str] = {
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
