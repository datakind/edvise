"""Tests for edvise.utils.api_requests module."""

import pytest
from unittest.mock import Mock, patch
import requests

from edvise.utils import api_requests


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
        assert "institutions/name/Test%20University" in call_args[0][0]
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

        # Test with various special characters
        test_cases = [
            ("University & College", "University%20%26%20College"),
            ("State/Community College", "State%2FCommunity%20College"),
            ("College (Main Campus)", "College%20%28Main%20Campus%29"),
            ("College with émojis 🎓", "College%20with%20%C3%A9mojis%20%F0%9F%8E%93"),
        ]

        for institution_name, expected_encoded in test_cases:
            api_requests.get_institution_id_by_name(institution_name, "api-key")
            call_url = mock_session.get.call_args[0][0]
            assert expected_encoded in call_url, (
                f"Expected '{expected_encoded}' in URL, got '{call_url}'"
            )

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
        call_url = mock_session.get.call_args[0][0]
        assert not call_url.endswith("%20")
        assert "Test%20University" in call_url

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

        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "404 Client Error: Not Found"
        )
        mock_response.status_code = 404
        mock_response.text = "Institution not found."

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        with pytest.raises(requests.HTTPError) as exc_info:
            api_requests.get_institution_id_by_name("Nonexistent University", "api-key")

        assert "404" in str(exc_info.value)

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
        # Verify the URL contains the reversed name
        call_url = mock_session.get.call_args[0][0]
        assert "institutions/name/Motlow%20State%20Community%20College" in call_url

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

        # Should work - whitespace is stripped
        assert result == "abc123"
        call_url = mock_session.get.call_args[0][0]
        assert "institutions/name/Test%20University" in call_url

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

        # Verify logging was called with context
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0][0]
        assert (
            "non-JSON" in log_call_args or "non-JSON".lower() in log_call_args.lower()
        )
        assert "Test University" in log_call_args

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

        # Verify logging was called with context
        mock_logger.error.assert_called_once()
        log_call_args = mock_logger.error.call_args[0][0]
        assert "inst_id" in log_call_args
        assert "Test University" in log_call_args

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
        assert "My Test University" in error_msg
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
        assert "My Test University" in error_msg
        assert "inst_id" in error_msg


class TestReverseDatabricksifyInstName:
    """Test cases for reverse_databricksify_inst_name function."""

    def test_reverse_community_college(self):
        """Test reversing community college abbreviation."""
        result = api_requests.reverse_databricksify_inst_name("motlow_state_cc")
        assert result == "Motlow State Community College"

    def test_reverse_university(self):
        """Test reversing university abbreviation."""
        result = api_requests.reverse_databricksify_inst_name("kentucky_state_uni")
        assert result == "Kentucky State University"

    def test_reverse_college(self):
        """Test reversing college abbreviation."""
        result = api_requests.reverse_databricksify_inst_name("central_arizona_col")
        assert result == "Central Arizona College"

    def test_reverse_community_technical_college(self):
        """Test reversing community technical college abbreviation."""
        result = api_requests.reverse_databricksify_inst_name("southeast_kentucky_ctc")
        assert result == "Southeast Kentucky Community Technical College"

    def test_reverse_science_and_technology(self):
        """Test reversing 'of science and technology' abbreviation."""
        result = api_requests.reverse_databricksify_inst_name("harrisburg_uni_st")
        assert result == "Harrisburg University Of Science And Technology"

    def test_reverse_multiple_words(self):
        """Test reversing name with multiple words."""
        result = api_requests.reverse_databricksify_inst_name("metro_state_uni_denver")
        assert result == "Metro State University Denver"

    def test_reverse_simple_name(self):
        """Test reversing name without abbreviations."""
        result = api_requests.reverse_databricksify_inst_name("test_institution")
        assert result == "Test Institution"

    def test_reverse_with_numbers(self):
        """Test reversing name with numbers."""
        result = api_requests.reverse_databricksify_inst_name("college_123")
        assert result == "College 123"

    def test_reverse_empty_string(self):
        """Test that empty string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            api_requests.reverse_databricksify_inst_name("")
        assert "non-empty string" in str(exc_info.value).lower()

    def test_reverse_invalid_characters(self):
        """Test that invalid characters raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            api_requests.reverse_databricksify_inst_name("invalid-name!")
        assert "invalid" in str(exc_info.value).lower()

    def test_reverse_uppercase(self):
        """Test that uppercase characters raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            api_requests.reverse_databricksify_inst_name("Invalid_Name")
        assert "invalid" in str(exc_info.value).lower()
        # Verify error message includes the problematic value
        assert "Invalid_Name" in str(exc_info.value)

    def test_reverse_whitespace_stripping(self):
        """Test that whitespace is handled correctly in databricks names."""
        # Databricks names shouldn't have spaces, but test edge case
        with pytest.raises(ValueError):
            api_requests.reverse_databricksify_inst_name("  test_name  ")

    def test_reverse_multiple_abbreviations(self):
        """Test reversing name with multiple abbreviations."""
        # Test case: name with both "uni" and "col"
        result = api_requests.reverse_databricksify_inst_name("test_uni_col")
        assert result == "Test University College"

    def test_reverse_error_message_includes_value(self):
        """Test that error messages include the problematic value."""
        with pytest.raises(ValueError) as exc_info:
            api_requests.reverse_databricksify_inst_name("bad-name!")
        error_msg = str(exc_info.value)
        assert "bad-name!" in error_msg
        assert "Invalid databricks name format" in error_msg
