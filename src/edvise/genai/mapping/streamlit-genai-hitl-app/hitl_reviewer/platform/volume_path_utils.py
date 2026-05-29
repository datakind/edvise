"""Helpers for Unity Catalog silver volume paths used by GenAI mapping."""

from __future__ import annotations

import re

# /Volumes/<catalog>/<institution_id>_silver/silver_volume/...
_INST_SILVER = re.compile(
    r"^/Volumes/[^/]+/(?P<inst>[^/]+)_silver/silver_volume/",
)


def institution_id_from_silver_volume_path(volume_path: str) -> str | None:
    """
    Parse ``institution_id`` from a path under ``…/<institution_id>_silver/silver_volume/``.

    Returns None when the pattern does not match.
    """
    p = (volume_path or "").strip()
    m = _INST_SILVER.match(p)
    if not m:
        return None
    return m.group("inst").strip() or None
