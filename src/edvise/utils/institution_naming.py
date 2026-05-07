"""
Institution slug ↔ display name transforms (Databricks naming rules).

Kept free of Spark, mlflow, and databricks-connect so lightweight callers
(e.g. the GenAI HITL Streamlit app) can ``import edvise.utils.institution_naming``
without pulling cluster runtime dependencies.
"""

from __future__ import annotations

import re

# Compiled regex patterns for reverse transformation (performance optimization)
_REVERSE_REPLACEMENTS = {
    "ctc": "community technical college",
    "cc": "community college",
    "st": "of science and technology",
    "uni": "university",
    "col": "college",
}

_COMPILED_REVERSE_PATTERNS = {
    abbrev: re.compile(r"\b" + re.escape(abbrev) + r"\b")
    for abbrev in _REVERSE_REPLACEMENTS.keys()
}


def _validate_databricks_name_format(databricks_name: str) -> None:
    if not isinstance(databricks_name, str) or not databricks_name.strip():
        raise ValueError("databricks_name must be a non-empty string")

    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, databricks_name):
        raise ValueError(
            f"Invalid databricks name format '{databricks_name}'. "
            "Must contain only lowercase letters, numbers, and underscores."
        )


def _reverse_abbreviation_replacements(name: str) -> str:
    words = name.split()

    for i in range(len(words)):
        if words[i] == "st" and i > 0:
            words[i] = "of science and technology"

    name = " ".join(words)

    for abbrev, full_form in _REVERSE_REPLACEMENTS.items():
        if abbrev != "st":
            pattern = _COMPILED_REVERSE_PATTERNS[abbrev]
            name = pattern.sub(full_form, name)

    return name


def databricksify_inst_name(inst_name: str) -> str:
    """
    Transform institution name to Databricks-compatible format.

    Follows DK standardized rules for naming conventions used in Databricks:
    - Lowercases the name
    - Replaces common phrases with abbreviations (e.g., "community college" → "cc")
    - Replaces special characters and spaces with underscores
    - Validates final format contains only lowercase letters, numbers, and underscores
    """
    name = inst_name.lower()

    dk_replacements = {
        "community technical college": "ctc",
        "community college": "cc",
        "of science and technology": "st",
        "university": "uni",
        "college": "col",
    }

    for old, new in dk_replacements.items():
        name = name.replace(old, new)

    special_char_replacements = {" & ": " ", "&": " ", "-": " "}
    for old, new in special_char_replacements.items():
        name = name.replace(old, new)

    final_name = name.replace(" ", "_")

    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, final_name):
        raise ValueError(
            f"Unexpected character found in Databricks compatible name: '{final_name}'"
        )

    return final_name


def reverse_databricksify_inst_name(databricks_name: str) -> str:
    """
    Reverse the databricksify transformation to get back the original institution name.

    This function attempts to reverse the transformation done by databricksify_inst_name.
    Since the transformation is lossy (multiple original names can map to the same
    databricks name), this function produces the most likely original name.
    """
    databricks_name = databricks_name.lower()
    _validate_databricks_name_format(databricks_name)

    name = databricks_name.replace("_", " ")
    name = _reverse_abbreviation_replacements(name)
    return name.title()


def format_institution_display_name(institution_id: str) -> str:
    """
    Human-readable label for an envelope ``institution_id`` / ``school_id`` slug.

    Used by lightweight UIs that must not import :mod:`edvise.utils.databricks`.
    """
    raw = (institution_id or "").strip()
    if not raw:
        return "Unknown institution"
    try:
        return reverse_databricksify_inst_name(raw.lower())
    except ValueError:
        return raw
