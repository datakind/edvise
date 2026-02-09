"""Tests for term_filter param parsing in pdp_inf_prep."""

import pytest

from edvise.scripts.pdp_inf_prep import parse_term_filter_param


def testparse_term_filter_param_none_returns_none():
    """None value is treated as not provided; returns None."""
    assert parse_term_filter_param(None) is None


def testparse_term_filter_param_empty_string_returns_none():
    """Empty string is treated as not provided; returns None."""
    assert parse_term_filter_param("") is None
    assert parse_term_filter_param("   ") is None


def testparse_term_filter_param_null_string_returns_none():
    """String 'null' or 'None' is treated as not provided; returns None."""
    assert parse_term_filter_param("null") is None
    assert parse_term_filter_param("None") is None
    assert parse_term_filter_param("  null  ") is None


def testparse_term_filter_param_valid_json_returns_list():
    """Valid JSON list of labels returns list of strings."""
    assert parse_term_filter_param('["fall 2024-25"]') == ["fall 2024-25"]
    assert parse_term_filter_param('["fall 2024-25", "spring 2024-25"]') == [
        "fall 2024-25",
        "spring 2024-25",
    ]


def testparse_term_filter_param_valid_json_stripped_whitespace():
    """Leading/trailing whitespace on the string is stripped before parsing."""
    assert parse_term_filter_param('  ["fall 2024-25"]  ') == ["fall 2024-25"]


def testparse_term_filter_param_empty_list_returns_none():
    """Empty JSON list is treated as not provided; returns None."""
    assert parse_term_filter_param("[]") is None


def testparse_term_filter_param_invalid_json_raises():
    """Invalid JSON raises ValueError with message about --term_filter or invalid JSON."""
    with pytest.raises(ValueError, match="Invalid JSON|--term_filter"):
        parse_term_filter_param("not json")
    with pytest.raises(ValueError, match="Invalid JSON|--term_filter"):
        parse_term_filter_param('["unclosed')


def testparse_term_filter_param_non_list_raises():
    """JSON that is not a list raises ValueError."""
    with pytest.raises(ValueError, match="must be a JSON list"):
        parse_term_filter_param('"fall 2024-25"')
    with pytest.raises(ValueError, match="must be a JSON list"):
        parse_term_filter_param("123")


def testparse_term_filter_param_list_with_empty_elements_stripped():
    """List elements that are empty or whitespace are dropped; non-empty stripped."""
    assert parse_term_filter_param('["fall 2024-25", ""]') == ["fall 2024-25"]
    assert parse_term_filter_param('["  fall 2024-25  "]') == ["fall 2024-25"]
    assert parse_term_filter_param('["  ", ""]') is None
