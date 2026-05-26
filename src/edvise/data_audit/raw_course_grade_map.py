"""Map institution-specific raw course grade codes to canonical Edvise grade strings."""

from __future__ import annotations

import pandas as pd


def normalize_grade_map(raw: dict[str, str] | None) -> dict[str, str]:
    """Uppercase/strip keys and values for consistent matching after snake_case / reads."""
    if not raw:
        return {}
    return {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in raw.items()
        if k is not None and v is not None and str(k).strip() != ""
    }


def apply_raw_course_grade_map(
    df: pd.DataFrame,
    grade_map: dict[str, str] | None,
    *,
    grade_col: str = "grade",
) -> pd.DataFrame:
    """
    Replace ``grade_col`` values using ``grade_map`` (keys = raw codes, values = canonical).

    Unmapped values are unchanged. Intended for use before :class:`RawEdviseCourseDataSchema`
    validation so mapped values satisfy ``ALLOWED_LETTER_GRADES`` / numeric rules.
    """
    norm = normalize_grade_map(grade_map)
    if not norm or grade_col not in df.columns:
        return df
    out = df.copy()
    s = out[grade_col].astype("string").str.strip().str.upper()
    out[grade_col] = s.replace(norm)
    return out
