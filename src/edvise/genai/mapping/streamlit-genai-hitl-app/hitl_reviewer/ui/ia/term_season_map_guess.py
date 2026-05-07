"""
Heuristic seeds for ``season_map_replace`` when the agent left placeholders empty.

Uses a common US semester-start **month code → canonical** mapping for two-digit month
fragments (e.g. YYYYMM positions 5–6): Jan–Apr → SPRING, May–Jul → SUMMER,
Aug–Nov → FALL, Dec → WINTER.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

_CANON_BY_MM: dict[str, str] = {
    "01": "SPRING",
    "02": "SPRING",
    "03": "SPRING",
    "04": "SPRING",
    "05": "SUMMER",
    "06": "SUMMER",
    "07": "SUMMER",
    "08": "FALL",
    "09": "FALL",
    "10": "FALL",
    "11": "FALL",
    "12": "WINTER",
}


def guess_month_code_rows_from_hitl_text(
    hitl_question: str, hitl_context: Any
) -> list[dict[str, str]]:
    """
    Infer distinct two-digit month tokens mentioned in HITL text and map them with the default US
    semester heuristic. Best-effort — reviewers should verify against institutional calendars.
    """
    chunks: list[str] = [hitl_question or ""]
    if isinstance(hitl_context, str):
        chunks.append(hitl_context)
    elif hitl_context is not None:
        try:
            chunks.append(json.dumps(hitl_context, ensure_ascii=False))
        except TypeError:
            chunks.append(str(hitl_context))
    blob = "\n".join(chunks)

    months: set[str] = set()
    for m in re.finditer(r"""['"](\d{2})['"]""", blob):
        months.add(m.group(1))
    # YYYYMM compact integers (e.g. 201308 or 201308.0)
    for m in re.finditer(r"\b((?:19|20)\d{4})(?:\.\d+)?\b", blob):
        token = m.group(1)
        if len(token) == 6:
            months.add(token[4:6])
    for m in re.finditer(r"\b(\d{6})\b", blob):
        s = m.group(1)
        if len(s) == 6 and s[:2] in ("19", "20"):
            months.add(s[4:6])

    out: list[dict[str, str]] = []
    for mm in sorted(months):
        if mm in _CANON_BY_MM:
            out.append({"raw": mm, "canonical": _CANON_BY_MM[mm]})
    return out


def build_season_map_seed_dataframe(
    *,
    resolution_season_map_replace: Any,
    hitl_question: str,
    hitl_context: Any,
) -> pd.DataFrame:
    """
    Prefer non-empty JSON from the agent; otherwise fill with guessed rows from HITL text;
    otherwise one editable blank row.
    """
    rows: list[dict[str, str]] = []
    if isinstance(resolution_season_map_replace, list):
        for e in resolution_season_map_replace:
            if not isinstance(e, dict):
                continue
            can = str(e.get("canonical") or "").strip().upper()
            if can not in ("FALL", "SPRING", "SUMMER", "WINTER"):
                can = "SPRING"
            raw = str(e.get("raw", "")).strip()
            if raw or can:
                rows.append({"raw": raw, "canonical": can})

    if not rows:
        rows = guess_month_code_rows_from_hitl_text(hitl_question, hitl_context)

    if not rows:
        rows = [{"raw": "", "canonical": "SPRING"}]

    return pd.DataFrame(rows)
