"""Tests for cohort param parsing in pdp_data_audit (Task 5)."""

import pytest

from edvise.scripts.pdp_data_audit import _parse_cohort_param


def test_parse_cohort_param_none_returns_none():
    """None value is treated as not provided; returns None."""
    assert _parse_cohort_param(None) is None


def test_parse_cohort_param_empty_string_returns_none():
    """Empty string is treated as not provided; returns None."""
    assert _parse_cohort_param("") is None
    assert _parse_cohort_param("   ") is None


def test_parse_cohort_param_null_string_returns_none():
    """String 'null' or 'None' is treated as not provided; returns None."""
    assert _parse_cohort_param("null") is None
    assert _parse_cohort_param("None") is None
    assert _parse_cohort_param("  null  ") is None


def test_parse_cohort_param_valid_json_returns_list():
    """Valid JSON list of cohort labels returns list of strings."""
    assert _parse_cohort_param('["fall 2024-25"]') == ["fall 2024-25"]
    assert _parse_cohort_param('["fall 2024-25", "spring 2024-25"]') == [
        "fall 2024-25",
        "spring 2024-25",
    ]


def test_parse_cohort_param_valid_json_stripped_whitespace():
    """Leading/trailing whitespace on the string is stripped before parsing."""
    assert _parse_cohort_param('  ["fall 2024-25"]  ') == ["fall 2024-25"]


def test_parse_cohort_param_empty_list_returns_none():
    """Empty JSON list is treated as not provided; returns None."""
    assert _parse_cohort_param("[]") is None


def test_parse_cohort_param_invalid_json_raises():
    """Invalid JSON raises ValueError with message about --cohort or invalid JSON."""
    with pytest.raises(ValueError, match="Invalid JSON|--cohort"):
        _parse_cohort_param("not json")
    with pytest.raises(ValueError, match="Invalid JSON|--cohort"):
        _parse_cohort_param('["unclosed')


def test_parse_cohort_param_non_list_raises():
    """JSON that is not a list raises ValueError."""
    with pytest.raises(ValueError, match="must be a JSON list"):
        _parse_cohort_param('"fall 2024-25"')
    with pytest.raises(ValueError, match="must be a JSON list"):
        _parse_cohort_param("123")


def test_parse_cohort_param_list_with_empty_elements_stripped():
    """List elements that are empty or whitespace are dropped; non-empty stripped."""
    assert _parse_cohort_param('["fall 2024-25", ""]') == ["fall 2024-25"]
    assert _parse_cohort_param('["  fall 2024-25  "]') == ["fall 2024-25"]
    assert _parse_cohort_param('["  ", ""]') is None
