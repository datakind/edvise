"""Unity Catalog model-name encode/decode for decimal time limits.

Unity Catalog three-level names split on ``.``, so a display name like
``graduation_in_3y_ft_4.5Y_pt_...`` must be stored as
``graduation_in_3y_ft_4d5y_pt_...``.

These helpers are the shared contract for:
- pipeline registration (encode before UC)
- model-card / reporting display (decode for humans)
- API boundary (encode for Databricks lookups; decode for frontend responses)

Only decimal *time limits* of the form ``{int}.{int}{y|m}`` (case-insensitive)
are rewritten, so unrelated dots are left alone.
"""

from __future__ import annotations

import re

# 4.5y / 4.5Y / 4.5m → 4d5y (UC-safe, unit always lowercase).
_UC_DECIMAL_DOT = re.compile(r"(\d+)\.(\d+)([yYmM])")
# Inverse: 4d5y → 4.5Y (frontend display matches compact 3Y).
_UC_DECIMAL_PLACEHOLDER = re.compile(r"(\d+)d(\d+)([yYmM])")
_DISPLAY_DECIMAL_UNIT = re.compile(r"(\d+\.\d+)([ym])")


def encode_uc_model_name(name: str) -> str:
    """Encode display decimals for Unity Catalog (``4.5Y`` → ``4d5y``)."""
    return _UC_DECIMAL_DOT.sub(
        lambda m: f"{m.group(1)}d{m.group(2)}{m.group(3).lower()}", name
    )


def decode_uc_model_name(name: str) -> str:
    """Decode UC placeholders for frontend display (``4d5y`` → ``4.5Y``)."""
    decoded = _UC_DECIMAL_PLACEHOLDER.sub(
        lambda m: f"{m.group(1)}.{m.group(2)}{m.group(3).upper()}", name
    )
    return _DISPLAY_DECIMAL_UNIT.sub(lambda m: m.group(1) + m.group(2).upper(), decoded)
