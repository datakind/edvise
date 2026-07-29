"""Institution slug ↔ display name transforms (Databricks naming rules)."""

from __future__ import annotations

import re

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


def reverse_databricksify_inst_name(databricks_name: str) -> str:
    databricks_name = databricks_name.lower()
    _validate_databricks_name_format(databricks_name)

    name = databricks_name.replace("_", " ")
    name = _reverse_abbreviation_replacements(name)
    return name.title()


def format_institution_display_name(institution_id: str) -> str:
    raw = (institution_id or "").strip()
    if not raw:
        return "Unknown institution"
    try:
        return reverse_databricksify_inst_name(raw.lower())
    except ValueError:
        return raw
