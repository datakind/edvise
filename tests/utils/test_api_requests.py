"""Tests for edvise.utils.api_requests module."""

import pytest
from unittest.mock import Mock, patch
import requests

from edvise.utils import api_requests
from edvise.utils.databricks import (
    databricksify_inst_name,
    reverse_databricksify_inst_name,
)


class TestGetInstitutionIdByName:
    """Test cases for get_institution_id_by_name function."""

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_successful_lookup(self, mock_session_class, mock_get_tokens):
        """Test successful institution ID lookup with valid name."""
        # Setup mocks
        mock_token = "test-access-token-123"
        mock_get_tokens.return_value = mock_token

        mock_response = Mock()
        mock_response.json.return_value = {
            "inst_id": "abc123-def456-ghi789",
            "name": "Test University",
            "retention_days": 365,
            "state": "NY",
            "pdp_id": None,
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute
        result = api_requests.get_institution_id_by_name(
            "Test University", "test-api-key"
        )

        # Assert
        assert result == "abc123-def456-ghi789"
        mock_get_tokens.assert_called_once_with(api_key="test-api-key")
        mock_session.get.assert_called_once()
        call_args = mock_session.get.call_args
        # Name is normalized to lowercase before API call
        assert "institutions/name/test%20university" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == f"Bearer {mock_token}"
        assert call_args[1]["timeout"] == 15

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_url_encoding_special_characters(self, mock_session_class, mock_get_tokens):
        """Test that special characters in institution name are URL-encoded."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {"inst_id": "test-id"}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Test with various special characters (names are normalized to lowercase)
        # Note: URL encoding uses uppercase hex (e.g., %2F not %2f), but we check for lowercase text
        test_cases = [
            ("University & College", "university", "college"),
            ("State/Community College", "state", "community", "college"),
            ("College (Main Campus)", "college", "main", "campus"),
        ]

        for institution_name, *expected_parts in test_cases:
            api_requests.get_institution_id_by_name(institution_name, "api-key")
            call_url = mock_session.get.call_args[0][0].lower()
            # Check that all lowercase parts of the name appear in the URL
            for part in expected_parts:
                assert part in call_url, f"Expected '{part}' in URL, got '{call_url}'"

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_url_encoding_strips_whitespace(self, mock_session_class, mock_get_tokens):
        """Test that whitespace is stripped before URL encoding."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {"inst_id": "test-id"}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute with leading/trailing whitespace
        api_requests.get_institution_id_by_name("  Test University  ", "api-key")

        # Assert URL doesn't have leading/trailing encoded spaces
        # Name is normalized to lowercase before API call
        call_url = mock_session.get.call_args[0][0]
        assert not call_url.endswith("%20")
        assert "test%20university" in call_url.lower()

    def test_validation_error_empty_institution_name(self):
        """Test that empty institution name returns error dict."""
        result = api_requests.get_institution_id_by_name("", "test-api-key")

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"
        assert "institution_name" in result["error"].lower()

    def test_validation_error_whitespace_only_institution_name(self):
        """Test that whitespace-only institution name returns error dict."""
        result = api_requests.get_institution_id_by_name("   ", "test-api-key")

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"
        assert "institution_name" in result["error"].lower()

    def test_validation_error_none_institution_name(self):
        """Test that None institution name returns error dict."""
        result = api_requests.get_institution_id_by_name(None, "test-api-key")  # type: ignore

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"

    def test_validation_error_non_string_institution_name(self):
        """Test that non-string institution name returns error dict."""
        result = api_requests.get_institution_id_by_name(123, "test-api-key")  # type: ignore

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"

    def test_validation_error_empty_api_key(self):
        """Test that empty API key returns error dict."""
        result = api_requests.get_institution_id_by_name("Test University", "")

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"
        assert "api_key" in result["error"].lower()

    def test_validation_error_none_api_key(self):
        """Test that None API key returns error dict."""
        result = api_requests.get_institution_id_by_name("Test University", None)  # type: ignore

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_http_error_404_not_found(self, mock_session_class, mock_get_tokens):
        """Test that 404 HTTP error is raised when institution not found."""
        mock_get_tokens.return_value = "test-token"

        # Mock response returning 404
        mock_response_404 = Mock()
        http_error_404 = requests.HTTPError("404 Client Error: Not Found")
        http_error_404.response = Mock()
        http_error_404.response.status_code = 404
        mock_response_404.raise_for_status.side_effect = http_error_404
        mock_response_404.status_code = 404
        mock_response_404.text = "Institution not found."

        mock_session = Mock()
        mock_session.get.return_value = mock_response_404
        mock_session_class.return_value = mock_session

        with pytest.raises(requests.HTTPError) as exc_info:
            api_requests.get_institution_id_by_name("Nonexistent University", "api-key")

        assert "404" in str(exc_info.value)
        # Verify that the name was normalized to lowercase in the API call
        assert mock_session.get.call_count == 1
        call_args = mock_session.get.call_args[0][0]
        assert "nonexistent%20university" in call_args.lower()  # URL-encoded lowercase

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_name_normalized_to_lowercase(self, mock_session_class, mock_get_tokens):
        """Test that institution name is normalized to lowercase before API call."""
        mock_get_tokens.return_value = "test-token"

        mock_response_success = Mock()
        mock_response_success.json.return_value = {"inst_id": "test-id-123"}
        mock_response_success.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response_success
        mock_session_class.return_value = mock_session

        # Test with mixed case input
        result = api_requests.get_institution_id_by_name("Test University", "api-key")

        assert result == "test-id-123"
        # Verify that the API was called with lowercase version
        assert mock_session.get.call_count == 1
        call_args = mock_session.get.call_args[0][0]
        assert "test%20university" in call_args.lower()  # URL-encoded lowercase

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_http_error_500_server_error(self, mock_session_class, mock_get_tokens):
        """Test that 500 HTTP error is raised on server error."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )
        mock_response.status_code = 500

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(requests.HTTPError):
            api_requests.get_institution_id_by_name("Test University", "api-key")

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_missing_inst_id_in_response(self, mock_session_class, mock_get_tokens):
        """Test that KeyError is raised when response doesn't contain inst_id."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "Test University",
            "retention_days": 365,
            # Missing inst_id
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(KeyError) as exc_info:
            api_requests.get_institution_id_by_name("Test University", "api-key")

        assert "inst_id" in str(exc_info.value)

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_non_dict_response(self, mock_session_class, mock_get_tokens):
        """Test that KeyError is raised when response is not a dict."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = ["not", "a", "dict"]
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(KeyError) as exc_info:
            api_requests.get_institution_id_by_name("Test University", "api-key")

        assert "inst_id" in str(exc_info.value)

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_non_json_response(self, mock_session_class, mock_get_tokens):
        """Test that ValueError is raised when response is not valid JSON."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "<html>Not JSON</html>"
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(ValueError) as exc_info:
            api_requests.get_institution_id_by_name("Test University", "api-key")

        assert "non-json" in str(exc_info.value).lower()
        assert "Not JSON" in str(exc_info.value)

    @patch("edvise.utils.api_requests.get_access_tokens")
    def test_token_retrieval_failure(self, mock_get_tokens):
        """Test that token retrieval failure propagates."""
        mock_get_tokens.side_effect = KeyError("No access_token in response")

        with pytest.raises(KeyError) as exc_info:
            api_requests.get_institution_id_by_name("Test University", "api-key")

        assert "access_token" in str(exc_info.value)

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_headers_include_accept_and_authorization(
        self, mock_session_class, mock_get_tokens
    ):
        """Test that GET request headers include Accept and Authorization."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {"inst_id": "test-id"}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        api_requests.get_institution_id_by_name("Test University", "api-key")

        headers = mock_session.get.call_args[1]["headers"]
        assert "Accept" in headers
        assert headers["Accept"] == "application/json"
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-token"

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_databricks_name_reverse_transformation(
        self, mock_session_class, mock_get_tokens
    ):
        """Test that databricks names are correctly reverse-transformed before querying."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {
            "inst_id": "abc123-def456-ghi789",
            "name": "Motlow State Community College",
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute with databricks name
        result = api_requests.get_institution_id_by_name(
            "motlow_state_cc", "test-api-key", is_databricks_name=True
        )

        # Assert
        assert result == "abc123-def456-ghi789"
        # Verify the URL contains the reversed name (normalized to lowercase)
        call_url = mock_session.get.call_args[0][0]
        assert (
            "institutions/name/motlow%20state%20community%20college" in call_url.lower()
        )

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_databricks_name_invalid_format(self, mock_session_class, mock_get_tokens):
        """Test that invalid databricks name format returns error."""
        result = api_requests.get_institution_id_by_name(
            "Invalid Name!", "test-api-key", is_databricks_name=True
        )

        assert isinstance(result, dict)
        assert result["ok"] is False
        assert result["stage"] == "validation"
        assert "databricks name" in result["error"].lower()
        # Verify error message includes context
        assert "Invalid databricks name format" in result["error"]

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_databricks_name_whitespace_handling(
        self, mock_session_class, mock_get_tokens
    ):
        """Test that whitespace in databricks name is stripped before reverse transformation."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {
            "inst_id": "abc123",
            "name": "Test University",
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute with databricks name that has whitespace
        result = api_requests.get_institution_id_by_name(
            "  test_uni  ", "test-api-key", is_databricks_name=True
        )

        # Should work - whitespace is stripped and name is normalized to lowercase
        assert result == "abc123"
        call_url = mock_session.get.call_args[0][0]
        assert "institutions/name/test%20university" in call_url.lower()

    @patch("edvise.utils.api_requests.LOGGER")
    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_logging_on_databricks_name_error(
        self, mock_session_class, mock_get_tokens, mock_logger
    ):
        """Test that errors are logged when databricks name is invalid."""
        result = api_requests.get_institution_id_by_name(
            "Invalid Name!", "test-api-key", is_databricks_name=True
        )

        # Verify result is an error dict
        assert isinstance(result, dict)
        assert result.get("ok") is False
        assert "validation" in result.get("stage", "")

        # Verify logging was called
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0][0]
        assert "Invalid databricks name format" in log_call_args
        assert "Invalid Name!" in log_call_args

    @patch("edvise.utils.api_requests.LOGGER")
    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_logging_on_non_json_response(
        self, mock_session_class, mock_get_tokens, mock_logger
    ):
        """Test that non-JSON responses are logged."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "<html>Not JSON</html>"
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(ValueError):
            api_requests.get_institution_id_by_name("Test University", "api-key")

        # Verify logging was called with context (name is normalized to lowercase)
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0][0]
        assert (
            "non-JSON" in log_call_args or "non-JSON".lower() in log_call_args.lower()
        )
        assert "test university" in log_call_args.lower()

    @patch("edvise.utils.api_requests.LOGGER")
    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_logging_on_missing_inst_id(
        self, mock_session_class, mock_get_tokens, mock_logger
    ):
        """Test that missing inst_id in response is logged."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "Test University",
            # Missing inst_id
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(KeyError):
            api_requests.get_institution_id_by_name("Test University", "api-key")

        # Verify logging was called with context (name is normalized to lowercase)
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0][0]
        assert "inst_id" in log_call_args
        assert "test university" in log_call_args.lower()

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_error_message_includes_institution_name(
        self, mock_session_class, mock_get_tokens
    ):
        """Test that error messages include the institution name for context."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.text = "<html>Error</html>"
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(ValueError) as exc_info:
            api_requests.get_institution_id_by_name("My Test University", "api-key")

        error_msg = str(exc_info.value)
        # Name is normalized to lowercase in error messages
        assert "my test university" in error_msg.lower()
        assert "non-json" in error_msg.lower() or "non-JSON" in error_msg

    @patch("edvise.utils.api_requests.get_access_tokens")
    @patch("edvise.utils.api_requests.requests.Session")
    def test_error_message_includes_institution_name_for_missing_inst_id(
        self, mock_session_class, mock_get_tokens
    ):
        """Test that KeyError includes institution name in error message."""
        mock_get_tokens.return_value = "test-token"

        mock_response = Mock()
        mock_response.json.return_value = {"name": "Test University"}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(KeyError) as exc_info:
            api_requests.get_institution_id_by_name("My Test University", "api-key")

        error_msg = str(exc_info.value)
        # Name is normalized to lowercase in error messages
        assert "my test university" in error_msg.lower()
        assert "inst_id" in error_msg


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
            databricksify_inst_name("Kentucky State University")
            == "kentucky_state_uni"
        )
        assert (
            databricksify_inst_name("Metro State University Denver")
            == "metro_state_uni_denver"
        )

    def test_college(self):
        """Test college abbreviation."""
        assert (
            databricksify_inst_name("Central Arizona College")
            == "central_arizona_col"
        )

    def test_community_technical_college(self):
        """Test community technical college abbreviation."""
        assert (
            databricksify_inst_name(
                "Southeast Kentucky community technical college"
            )
            == "southeast_kentucky_ctc"
        )

    def test_science_and_technology(self):
        """Test 'of science and technology' abbreviation."""
        assert (
            databricksify_inst_name(
                "Harrisburg University of Science and Technology"
            )
            == "harrisburg_uni_st"
        )

    def test_special_characters(self):
        """Test handling of special characters like & and -."""
        assert (
            databricksify_inst_name("University of Science & Technology")
            == "uni_of_st_technology"
        )
        assert (
            databricksify_inst_name("State-Community College")
            == "state_community_col"
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
        assert (
            databricksify_inst_name("Big State University")
            == "big_state_uni"
        )


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
