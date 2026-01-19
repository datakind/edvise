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
