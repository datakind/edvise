"""Tests for edvise.utils.databricks module."""

import pytest

from edvise.utils.databricks import (
    databricksify_inst_name,
    reverse_databricksify_inst_name,
)


class TestDatabricksifyInstName:
    """Test cases for databricksify_inst_name function."""

    def test_community_college(self):
        """Test community college abbreviation."""
        assert (
            databricksify_inst_name("Motlow State Community College")
            == "motlow_state_cc"
        )
        assert (
            databricksify_inst_name("Northwest State Community College")
            == "northwest_state_cc"
        )

    def test_university(self):
        """Test university abbreviation."""
        assert (
            databricksify_inst_name("Kentucky State University") == "kentucky_state_uni"
        )
        assert (
            databricksify_inst_name("Metro State University Denver")
            == "metro_state_uni_denver"
        )

    def test_college(self):
        """Test college abbreviation."""
        assert (
            databricksify_inst_name("Central Arizona College") == "central_arizona_col"
        )

    def test_community_technical_college(self):
        """Test community technical college abbreviation."""
        assert (
            databricksify_inst_name("Southeast Kentucky community technical college")
            == "southeast_kentucky_ctc"
        )

    def test_science_and_technology(self):
        """Test 'of science and technology' abbreviation."""
        assert (
            databricksify_inst_name("Harrisburg University of Science and Technology")
            == "harrisburg_uni_st"
        )

    def test_special_characters(self):
        """Test handling of special characters like & and -."""
        assert (
            databricksify_inst_name("University of Science & Technology")
            == "uni_of_st_technology"
        )
        assert (
            databricksify_inst_name("State-Community College") == "state_community_col"
        )

    def test_invalid_characters(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            databricksify_inst_name("Northwest (invalid)")
        error_msg = str(exc_info.value)
        assert "Unexpected character found in Databricks compatible name" in error_msg
        assert (
            "northwest" in error_msg.lower()
        )  # Error message includes the problematic name

    def test_simple_name(self):
        """Test simple name without abbreviations."""
        assert databricksify_inst_name("Big State University") == "big_state_uni"


class TestReverseDatabricksifyInstName:
    """Test cases for reverse_databricksify_inst_name function."""

    def test_reverse_community_college(self):
        """Test reversing community college abbreviation."""
        result = reverse_databricksify_inst_name("motlow_state_cc")
        assert result == "Motlow State Community College"

    def test_reverse_university(self):
        """Test reversing university abbreviation."""
        result = reverse_databricksify_inst_name("kentucky_state_uni")
        assert result == "Kentucky State University"

    def test_reverse_college(self):
        """Test reversing college abbreviation."""
        result = reverse_databricksify_inst_name("central_arizona_col")
        assert result == "Central Arizona College"

    def test_reverse_community_technical_college(self):
        """Test reversing community technical college abbreviation."""
        result = reverse_databricksify_inst_name("southeast_kentucky_ctc")
        assert result == "Southeast Kentucky Community Technical College"

    def test_reverse_science_and_technology(self):
        """Test reversing 'of science and technology' abbreviation."""
        result = reverse_databricksify_inst_name("harrisburg_uni_st")
        assert result == "Harrisburg University Of Science And Technology"

    def test_reverse_saint_at_beginning(self):
        """Test that 'st' at the beginning is kept as abbreviation 'St'."""
        result = reverse_databricksify_inst_name("st_johns_uni")
        assert result == "St Johns University"

    def test_reverse_saint_vs_science_technology(self):
        """Test that 'st' at beginning is St (abbreviation), but in middle is 'of science and technology'."""
        # "st" at beginning should be "St" (abbreviation)
        result1 = reverse_databricksify_inst_name("st_marys_col")
        assert result1 == "St Marys College"

        # "st" in middle should be "of science and technology"
        result2 = reverse_databricksify_inst_name("harrisburg_uni_st")
        assert result2 == "Harrisburg University Of Science And Technology"

        # Both in same name (edge case)
        result3 = reverse_databricksify_inst_name("st_paul_uni_st")
        assert result3 == "St Paul University Of Science And Technology"

    def test_reverse_multiple_words(self):
        """Test reversing name with multiple words."""
        result = reverse_databricksify_inst_name("metro_state_uni_denver")
        assert result == "Metro State University Denver"

    def test_reverse_simple_name(self):
        """Test reversing name without abbreviations."""
        result = reverse_databricksify_inst_name("test_institution")
        assert result == "Test Institution"

    def test_reverse_with_numbers(self):
        """Test reversing name with numbers."""
        result = reverse_databricksify_inst_name("college_123")
        assert result == "College 123"

    def test_reverse_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            reverse_databricksify_inst_name("")
        assert "non-empty string" in str(exc_info.value).lower()

    def test_reverse_invalid_characters(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            reverse_databricksify_inst_name("invalid-name!")
        assert "invalid" in str(exc_info.value).lower()

    def test_reverse_uppercase_normalized(self):
        """Test that uppercase characters are normalized to lowercase."""
        # Uppercase input should be normalized to lowercase and processed
        result = reverse_databricksify_inst_name("MOTLOW_STATE_CC")
        assert result == "Motlow State Community College"

        # Mixed case should also be normalized
        result2 = reverse_databricksify_inst_name("St_Paul_Uni")
        assert result2 == "St Paul University"

        # Invalid characters (even after normalization) should still raise error
        with pytest.raises(ValueError) as exc_info:
            reverse_databricksify_inst_name("Invalid-Name!")
        assert "invalid" in str(exc_info.value).lower()
        # Verify error message includes the problematic value (normalized)
        assert "invalid-name!" in str(exc_info.value).lower()

    def test_reverse_whitespace_stripping(self):
        """Test that whitespace is handled correctly in databricks names."""
        # Databricks names shouldn't have spaces, but test edge case
        with pytest.raises(ValueError):
            reverse_databricksify_inst_name("  test_name  ")

    def test_reverse_multiple_abbreviations(self):
        """Test reversing name with multiple abbreviations."""
        # Test case: name with both "uni" and "col"
        result = reverse_databricksify_inst_name("test_uni_col")
        assert result == "Test University College"

    def test_reverse_error_message_includes_value(self):
        """Test that error messages include the problematic value."""
        with pytest.raises(ValueError) as exc_info:
            reverse_databricksify_inst_name("bad-name!")
        error_msg = str(exc_info.value)
        assert "bad-name!" in error_msg
        assert "Invalid databricks name format" in error_msg
