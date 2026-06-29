"""Map institution-specific raw course grade codes to canonical Edvise grade strings."""

from __future__ import annotations

import logging

import pandas as pd

from edvise.data_audit.default_grade_map import (
    DEFAULT_ES_GRADE_MAP,
    LETTER_GPA_GRADE_CODES,
    NON_GPA_STATUS_GRADE_CODES,
)

LOGGER = logging.getLogger(__name__)


def normalize_grade_map(raw: dict[str, str] | None) -> dict[str, str]:
    """Uppercase/strip keys and values for consistent matching after snake_case / reads."""
    if not raw:
        return {}
    return {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in raw.items()
        if k is not None and v is not None and str(k).strip() != ""
    }


def merge_grade_maps(
    base: dict[str, str] | None,
    override: dict[str, str] | None,
) -> dict[str, str]:
    """
    Merge two grade maps; ``override`` wins on duplicate keys (after normalization).
    """
    merged = normalize_grade_map(base)
    merged.update(normalize_grade_map(override))
    return merged


def resolve_es_grade_map(institution_map: dict[str, str] | None) -> dict[str, str]:
    """
    Platform defaults (status codes + letter→GPA) plus institution config.

    Institution ``preprocessing.features.grade_map`` overrides platform entries
    on duplicate keys.
    """
    return merge_grade_maps(DEFAULT_ES_GRADE_MAP, institution_map)


def _is_numeric_gpa_grade(val: object) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip()
    if not s:
        return False
    try:
        return 0.0 <= float(s) <= 4.0
    except (ValueError, TypeError):
        return False


def unmapped_gpa_grade_counts(
    df: pd.DataFrame,
    *,
    grade_col: str = "grade",
) -> pd.Series:
    """
    Value counts for grades that are schema-valid but still non-numeric after grade_map.

    These rows are excluded from ``course_grade_numeric`` in feature generation.
    """
    if grade_col not in df.columns:
        return pd.Series(dtype="int64")
    s = df[grade_col].astype("string").str.strip().str.upper()
    s = s.dropna()
    s = s[s != ""]
    if s.empty:
        return pd.Series(dtype="int64")
    unmapped = s[~s.map(_is_numeric_gpa_grade) & ~s.isin(NON_GPA_STATUS_GRADE_CODES)]
    return unmapped.value_counts()


def log_unmapped_gpa_grades(
    df: pd.DataFrame,
    *,
    grade_col: str = "grade",
) -> None:
    """
    Warn when letter (or other) grades remain non-numeric after the full grade_map.

    Intended for ES data audit after mapping and schema validation.
    """
    counts = unmapped_gpa_grade_counts(df, grade_col=grade_col)
    if counts.empty:
        return
    total = int(counts.sum())
    n_rows = len(df)
    pct = 100.0 * total / n_rows if n_rows else 0.0
    letter_hits = counts[counts.index.isin(LETTER_GPA_GRADE_CODES)]
    LOGGER.warning(
        "⚠️ %s course rows (%.2f%%) have grades still non-numeric after grade_map "
        "and will be excluded from GPA features: %s",
        total,
        pct,
        counts.to_dict(),
    )
    if not letter_hits.empty:
        LOGGER.warning(
            "⚠️ Unmapped letter grades (add to institution grade_map or platform "
            "defaults): %s",
            letter_hits.to_dict(),
        )


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
