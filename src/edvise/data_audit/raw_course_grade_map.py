"""Map institution-specific raw course grade codes to canonical Edvise grade strings."""

from __future__ import annotations

import pandas as pd

from edvise.data_audit.default_grade_map import DEFAULT_ES_GRADE_MAP


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
    Platform defaults plus institution ``preprocessing.features.grade_map``.

    Defaults cover common status/withdrawal tokens only. Each school adds letter-to-GPA
    and any local codes in bronze ``config.toml``.
    """
    return merge_grade_maps(DEFAULT_ES_GRADE_MAP, institution_map)


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
