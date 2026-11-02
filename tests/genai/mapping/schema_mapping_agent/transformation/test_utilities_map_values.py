"""map_values must match boolean Series against JSON string keys True/False."""

from __future__ import annotations

import pandas as pd

from edvise.genai.mapping.schema_mapping_agent.transformation.utilities import (
    cast_boolean,
    map_values,
)


def test_map_values_string_keys_match_boolean_series() -> None:
    s = pd.Series([True, False, True, pd.NA], dtype="boolean")
    out = map_values(s, {"True": "Yes", "False": "No"}, default=None)
    assert list(out[:3]) == ["Yes", "No", "Yes"]
    assert pd.isna(out.iloc[3])


def test_cast_boolean_then_map_values_does_not_null_column() -> None:
    """The Miami Dade plans: cast_boolean, then map_values with default null."""
    raw = pd.Series(["True", "False", "True", None], dtype="object")
    out = map_values(
        cast_boolean(raw),
        {"True": "Y", "False": "N"},
        default=None,
    )
    assert list(out[:3]) == ["Y", "N", "Y"]
    assert pd.isna(out.iloc[3])


def test_map_values_boolean_keys_are_case_insensitive() -> None:
    s = pd.Series([True, False], dtype="boolean")
    out = map_values(s, {"true": "Veteran", "FALSE": "Non-Veteran"}, default=None)
    assert list(out) == ["Veteran", "Non-Veteran"]


def test_map_values_string_series_still_matches_string_keys() -> None:
    s = pd.Series(["True", "False", "Unknown", None], dtype="object")
    out = map_values(s, {"True": "Yes", "False": "No"}, default=None)
    assert list(out[:2]) == ["Yes", "No"]
    assert pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3])


def test_map_values_passthrough_unchanged_for_unmapped_strings() -> None:
    s = pd.Series(["Hispanic", "Unknown", "Asian"])
    out = map_values(s, {"Unknown": None}, default="passthrough")
    assert list(out) == ["Hispanic", None, "Asian"]


def test_map_values_boolean_explicit_null_stays_null() -> None:
    s = pd.Series([True, False, pd.NA], dtype="boolean")
    out = map_values(s, {"True": None, "False": "No"}, default="passthrough")
    assert out.iloc[0] is None or pd.isna(out.iloc[0])
    assert out.iloc[1] == "No"
    assert pd.isna(out.iloc[2])


def test_map_values_does_not_fold_yes_onto_boolean_true() -> None:
    s = pd.Series([True, False], dtype="boolean")
    out = map_values(s, {"Yes": "Y", "No": "N"}, default=None)
    assert out.isna().all()
