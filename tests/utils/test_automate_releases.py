"""Tests for automate_releases utility functions."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from edvise.utils import automate_releases


class TestGetLastVersionTag:
    """Test get_last_version_tag function."""

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_last_version_tag_success(self, mock_run):
        """Test successful tag retrieval."""
        mock_run.return_value = MagicMock(
            stdout="v0.1.8\nv0.1.7\nv0.1.6\n", returncode=0
        )
        result = automate_releases.get_last_version_tag()
        assert result == "v0.1.8"

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_last_version_tag_no_tags(self, mock_run):
        """Test when no tags exist."""
        mock_run.return_value = MagicMock(stdout="\n", returncode=0)
        result = automate_releases.get_last_version_tag()
        assert result is None

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_last_version_tag_error(self, mock_run):
        """Test when git command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = automate_releases.get_last_version_tag()
        assert result is None


class TestGetPRNumbersFromGitLog:
    """Test get_pr_numbers_from_git_log function."""

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_success(self, mock_run):
        """Test successful PR number extraction."""
        mock_run.return_value = MagicMock(
            stdout="Merge pull request #123 from branch\nMerge pull request #124 from branch\n",
            returncode=0,
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == [123, 124]

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_no_since_tag(self, mock_run):
        """Test PR extraction without since_tag."""
        mock_run.return_value = MagicMock(
            stdout="Merge pull request #125 from branch\n", returncode=0
        )
        result = automate_releases.get_pr_numbers_from_git_log()
        assert result == [125]

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_error(self, mock_run):
        """Test when git log command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == []


class TestFetchPRTitle:
    """Test fetch_pr_title function."""

    @patch("edvise.utils.automate_releases.urllib.request.urlopen")
    @patch("edvise.utils.automate_releases.json.loads")
    def test_fetch_pr_title_success(self, mock_json_loads, mock_urlopen):
        """Test successful PR title fetch."""
        mock_response = MagicMock()
        mock_response.read.return_value.decode.return_value = '{"title": "Test PR"}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        mock_json_loads.return_value = {"title": "Test PR"}

        result = automate_releases.fetch_pr_title("owner/repo", 123, "token")
        assert result == "Test PR"

    @patch("edvise.utils.automate_releases.urllib.request.urlopen")
    def test_fetch_pr_title_error(self, mock_urlopen):
        """Test when API call fails."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )
        result = automate_releases.fetch_pr_title("owner/repo", 123, "token")
        assert result is None


class TestGetPRTitlesSinceLastVersion:
    """Test get_pr_titles_since_last_version function."""

    @patch("edvise.utils.automate_releases.get_pr_numbers_from_git_log")
    @patch("edvise.utils.automate_releases.get_last_version_tag")
    @patch("edvise.utils.automate_releases.fetch_pr_title")
    def test_get_pr_titles_with_token(self, mock_fetch, mock_get_tag, mock_get_prs):
        """Test PR title fetching with token."""
        mock_get_tag.return_value = "v0.1.8"
        mock_get_prs.return_value = [123, 124]
        mock_fetch.side_effect = ["PR Title 1", "PR Title 2"]

        result = automate_releases.get_pr_titles_since_last_version(
            "owner/repo", "token"
        )
        assert result == ["PR Title 1 (#123)", "PR Title 2 (#124)"]
        assert mock_fetch.call_count == 2

    @patch("edvise.utils.automate_releases.get_pr_numbers_from_git_log")
    @patch("edvise.utils.automate_releases.get_last_version_tag")
    def test_get_pr_titles_no_token(self, mock_get_tag, mock_get_prs):
        """Test PR title fetching without token."""
        mock_get_tag.return_value = "v0.1.8"
        mock_get_prs.return_value = [123, 124]

        result = automate_releases.get_pr_titles_since_last_version("owner/repo", None)
        assert result == ["PR #123", "PR #124"]
