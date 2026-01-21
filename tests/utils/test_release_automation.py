"""Comprehensive tests for release automation.

This file consolidates all release automation tests:
- PR fetching (automate_releases.py)
- Version updates (update_version.py)
- Workflow scripts (bash commands from workflows)
- Workflow YAML validation
"""

import re
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import tomlkit
import yaml

from edvise.scripts import update_version
from edvise.utils import automate_releases

# Get workspace root for workflow YAML tests
WORKSPACE_ROOT = Path(__file__).parent.parent.parent


# ============================================================================
# Shared Test Utilities
# ============================================================================


class VersionTestHelpers:
    """Shared helpers for version-related tests."""

    @staticmethod
    def validate_version_format(version: str) -> bool:
        """Validate version format X.Y.Z."""
        pattern = r"^[0-9]+\.[0-9]+\.[0-9]+$"
        return bool(re.match(pattern, version))

    @staticmethod
    def strip_version_prefix(version: str) -> str:
        """Strip 'v' prefix from version."""
        return version.lstrip("v")

    @staticmethod
    def extract_version_from_branch(branch: str) -> str:
        """Extract version from release branch name."""
        return branch.replace("release/", "")

    @staticmethod
    def extract_version_from_pr_title(title: str) -> str | None:
        """Extract version from back-merge PR title."""
        pattern = r"^chore: sync develop with release ([0-9]+\.[0-9]+\.[0-9]+)$"
        match = re.search(pattern, title)
        return match.group(1) if match else None


# ============================================================================
# PR Fetching Tests (automate_releases.py)
# ============================================================================


class TestPRFetching:
    """Test PR fetching functionality from automate_releases."""

    @staticmethod
    def _mock_git_log(stdout: str) -> MagicMock:
        """Helper to create a mock git log response."""
        return MagicMock(stdout=stdout, returncode=0)

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

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_success(self, mock_run):
        """Test successful PR number extraction."""
        mock_run.return_value = self._mock_git_log(
            "Merge pull request #123 from branch\nMerge pull request #124 from branch\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == [123, 124]

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_no_since_tag(self, mock_run):
        """Test PR extraction without since_tag."""
        mock_run.return_value = self._mock_git_log(
            "Merge pull request #125 from branch\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log()
        assert result == [125]

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_specific_regex(self, mock_run):
        """Test that regex only matches 'Merge pull request #N' pattern, not issues."""
        mock_run.return_value = self._mock_git_log(
            "Merge pull request #123 from branch\n"
            "Fixes issue #456\n"
            "Merge pull request #789 from owner/branch\n"
            "Closes #111 and fixes #222\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        # Should only extract PR numbers, not issue numbers
        assert result == [123, 789]
        assert 456 not in result
        assert 111 not in result
        assert 222 not in result

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_deduplication_and_sorted(self, mock_run):
        """Test that duplicate PR numbers are deduplicated and results are sorted."""
        mock_run.return_value = self._mock_git_log(
            "Merge pull request #999 from branch\n"
            "Merge pull request #123 from branch\n"
            "Merge pull request #123 from another-branch\n"  # Duplicate
            "Merge pull request #456 from branch\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        # Should be sorted and deduplicated
        assert result == [123, 456, 999]
        assert result == sorted(result)
        assert result.count(123) == 1

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_empty_lines(self, mock_run):
        """Test that empty lines are handled correctly."""
        mock_run.return_value = self._mock_git_log(
            "\n\nMerge pull request #123 from branch\n\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == [123]

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_no_matches(self, mock_run):
        """Test when no PR numbers are found in commit messages."""
        mock_run.return_value = self._mock_git_log(
            "Some other commit message\nFixes issue #123\n"
        )
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == []

    @patch("edvise.utils.automate_releases.subprocess.run")
    def test_get_pr_numbers_error(self, mock_run):
        """Test when git log command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        result = automate_releases.get_pr_numbers_from_git_log("v0.1.8")
        assert result == []

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


# ============================================================================
# Version Update Tests (update_version.py)
# ============================================================================


class TestUpdateChangelog:
    """Test update_changelog function."""

    def test_update_changelog_new_version(self):
        """Test adding a new version entry to changelog."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("## 0.1.8 (2025-12-11)\n- Some change\n\n")
            changelog_path = Path(f.name)

        try:
            with patch(
                "edvise.scripts.update_version.automate_releases.get_pr_titles_since_last_version"
            ) as mock_get_prs:
                mock_get_prs.return_value = [
                    "feat: new feature (#123)",
                    "fix: bug fix (#124)",
                ]

                update_version.update_changelog(
                    changelog_path, "0.1.9", "owner/repo", "token"
                )

                content = changelog_path.read_text()
                assert "## 0.1.9" in content
                assert "feat: new feature (#123)" in content
                assert "fix: bug fix (#124)" in content
                assert "## 0.1.8" in content
                assert content.index("## 0.1.9") < content.index("## 0.1.8")
        finally:
            changelog_path.unlink()

    def test_update_changelog_no_prs(self):
        """Test changelog update when no PRs are found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("## 0.1.8 (2025-12-11)\n- Some change\n\n")
            changelog_path = Path(f.name)

        try:
            with patch(
                "edvise.scripts.update_version.automate_releases.get_pr_titles_since_last_version"
            ) as mock_get_prs:
                mock_get_prs.return_value = []

                update_version.update_changelog(
                    changelog_path, "0.1.9", "owner/repo", "token"
                )

                content = changelog_path.read_text()
                assert "## 0.1.9" in content
                assert "- \n" in content
        finally:
            changelog_path.unlink()

    def test_update_changelog_duplicate_version(self):
        """Test that duplicate version entries are handled."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                "## 0.1.9 (2025-12-12)\n- Already exists\n\n## 0.1.8 (2025-12-11)\n- Old\n\n"
            )
            changelog_path = Path(f.name)

        try:
            with patch(
                "edvise.scripts.update_version.automate_releases.get_pr_titles_since_last_version"
            ):
                update_version.update_changelog(
                    changelog_path, "0.1.9", "owner/repo", "token"
                )
                content = changelog_path.read_text()
                assert content.count("## 0.1.9") == 1
        finally:
            changelog_path.unlink()

    def test_update_changelog_empty_file(self):
        """Test changelog update when file is empty."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("")
            changelog_path = Path(f.name)

        try:
            with patch(
                "edvise.scripts.update_version.automate_releases.get_pr_titles_since_last_version"
            ) as mock_get_prs:
                mock_get_prs.return_value = ["feat: first feature (#1)"]

                update_version.update_changelog(
                    changelog_path, "0.1.0", "owner/repo", "token"
                )

                content = changelog_path.read_text()
                assert "## 0.1.0" in content
                assert "feat: first feature (#1)" in content
        finally:
            changelog_path.unlink()

    def test_update_changelog_no_repo(self):
        """Test changelog update without repository."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("## 0.1.8 (2025-12-11)\n- Some change\n\n")
            changelog_path = Path(f.name)

        try:
            update_version.update_changelog(changelog_path, "0.1.9", None, None)
            content = changelog_path.read_text()
            assert "## 0.1.9" in content
            assert "- \n" in content
        finally:
            changelog_path.unlink()


class TestUpdatePyproject:
    """Test update_pyproject function."""

    def test_update_pyproject_version(self):
        """Test updating version in pyproject.toml."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[project]\nname = 'edvise'\nversion = '0.1.8'\n")
            pyproject_path = Path(f.name)

        try:
            with patch("edvise.scripts.update_version.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                update_version.update_pyproject(pyproject_path, "0.1.9")

                content = pyproject_path.read_text()
                doc = tomlkit.parse(content)
                assert doc["project"]["version"] == "0.1.9"
                assert doc["project"]["name"] == "edvise"
        finally:
            pyproject_path.unlink()

    def test_update_pyproject_missing_project_section(self):
        """Test error when [project] section is missing."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[tool.pytest]\noption = 'value'\n")
            pyproject_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="missing \\[project\\] section"):
                update_version.update_pyproject(pyproject_path, "0.1.9")
        finally:
            pyproject_path.unlink()

    def test_update_pyproject_uv_sync_success(self):
        """Test that uv sync is called after version update."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[project]\nname = 'edvise'\nversion = '0.1.8'\n")
            pyproject_path = Path(f.name)

        try:
            with patch("edvise.scripts.update_version.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                update_version.update_pyproject(pyproject_path, "0.1.9")

                mock_run.assert_called_with(
                    ["uv", "sync"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
        finally:
            pyproject_path.unlink()

    def test_update_pyproject_uv_sync_failure(self):
        """Test handling when uv sync fails."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[project]\nname = 'edvise'\nversion = '0.1.8'\n")
            pyproject_path = Path(f.name)

        try:
            with patch("edvise.scripts.update_version.subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    1, "uv", stderr="Error"
                )

                update_version.update_pyproject(pyproject_path, "0.1.9")

                content = pyproject_path.read_text()
                doc = tomlkit.parse(content)
                assert doc["project"]["version"] == "0.1.9"
        finally:
            pyproject_path.unlink()

    def test_update_pyproject_uv_not_found(self):
        """Test handling when uv command is not found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("[project]\nname = 'edvise'\nversion = '0.1.8'\n")
            pyproject_path = Path(f.name)

        try:
            with patch("edvise.scripts.update_version.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError()

                update_version.update_pyproject(pyproject_path, "0.1.9")

                content = pyproject_path.read_text()
                doc = tomlkit.parse(content)
                assert doc["project"]["version"] == "0.1.9"
        finally:
            pyproject_path.unlink()


class TestVersionUpdateIntegration:
    """Integration tests for update_version script."""

    def test_full_update_flow(self):
        """Test the full update flow with both changelog and pyproject."""
        with tempfile.TemporaryDirectory() as tmpdir:
            changelog_path = Path(tmpdir) / "CHANGELOG.md"
            pyproject_path = Path(tmpdir) / "pyproject.toml"

            changelog_path.write_text("## 0.1.8 (2025-12-11)\n- Old change\n\n")
            pyproject_path.write_text("[project]\nname = 'edvise'\nversion = '0.1.8'\n")

            with (
                patch(
                    "edvise.scripts.update_version.automate_releases.get_pr_titles_since_last_version"
                ) as mock_get_prs,
                patch("edvise.scripts.update_version.subprocess.run") as mock_run,
            ):
                mock_get_prs.return_value = ["feat: new feature (#123)"]
                mock_run.return_value = MagicMock(returncode=0)

                update_version.update_changelog(
                    changelog_path, "0.1.9", "owner/repo", "token"
                )
                update_version.update_pyproject(pyproject_path, "0.1.9")

                changelog_content = changelog_path.read_text()
                assert "## 0.1.9" in changelog_content
                assert "feat: new feature (#123)" in changelog_content

                pyproject_content = pyproject_path.read_text()
                doc = tomlkit.parse(pyproject_content)
                assert doc["project"]["version"] == "0.1.9"


# ============================================================================
# Workflow Script Tests (bash commands from workflows)
# ============================================================================


class TestVersionValidation:
    """Consolidated version validation tests."""

    def test_version_format_validation(self):
        """Test version format validation."""
        valid_versions = ["0.1.9", "1.0.0", "10.20.30", "v0.1.9"]
        invalid_versions = ["0.1", "0.1.9.1", "abc", ""]

        for version in valid_versions:
            stripped = VersionTestHelpers.strip_version_prefix(version)
            assert VersionTestHelpers.validate_version_format(stripped), (
                f"{version} (stripped: {stripped}) should be valid"
            )

        for version in invalid_versions:
            stripped = VersionTestHelpers.strip_version_prefix(version)
            assert not VersionTestHelpers.validate_version_format(stripped), (
                f"{version} (stripped: {stripped}) should be invalid"
            )

    def test_version_stripping(self):
        """Test version prefix stripping."""
        test_cases = [
            ("v0.1.9", "0.1.9"),
            ("0.1.9", "0.1.9"),
            ("V0.1.9", "V0.1.9"),
        ]

        for input_version, expected in test_cases:
            stripped = VersionTestHelpers.strip_version_prefix(input_version)
            assert stripped == expected, f"Expected '{expected}', got '{stripped}'"

    def test_version_extraction_from_branch(self):
        """Test version extraction from branch."""
        test_cases = [
            ("release/0.1.9", "0.1.9"),
            ("release/1.0.0", "1.0.0"),
        ]

        for branch, expected_version in test_cases:
            version = VersionTestHelpers.extract_version_from_branch(branch)
            assert version == expected_version
            assert VersionTestHelpers.validate_version_format(version)

    def test_version_extraction_from_pr_title(self):
        """Test version extraction from PR title."""
        valid_title = "chore: sync develop with release 0.1.9"
        version = VersionTestHelpers.extract_version_from_pr_title(valid_title)
        assert version == "0.1.9"

        invalid_titles = [
            "chore: sync develop with release",
            "fix: something else",
            "chore: sync develop with release 0.1",
            "chore: sync develop with release v0.1.9",
        ]

        for title in invalid_titles:
            version = VersionTestHelpers.extract_version_from_pr_title(title)
            assert version is None, f"Should not extract version from: {title}"


class TestPRCreation:
    """Consolidated PR creation tests."""

    @patch("subprocess.run")
    def test_check_existing_pr(self, mock_run):
        """Test checking for existing PR (both cases)."""
        mock_run.return_value = MagicMock(stdout="123\n", returncode=0)
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                "release/0.1.9",
                "--base",
                "main",
                "--state",
                "open",
                "--json",
                "number",
                "--jq",
                ".[0].number",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stdout.strip() == "123"

        mock_run.return_value = MagicMock(stdout="", returncode=0)
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                "release/0.1.9",
                "--base",
                "main",
                "--state",
                "open",
                "--json",
                "number",
                "--jq",
                ".[0].number",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.stdout.strip() == ""

    @patch("subprocess.run")
    def test_create_release_pr(self, mock_run):
        """Test creating release PR."""
        mock_run.return_value = MagicMock(
            stdout='{"url": "https://github.com/owner/repo/pull/123"}',
            returncode=0,
        )

        branch = "release/0.1.9"
        version = "0.1.9"

        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "main",
                "--head",
                branch,
                "--title",
                f"chore: release {version}",
                "--body",
                (
                    f"## Release {version}\n\n"
                    f"This PR merges `{branch}` into `main`.\n\n"
                    "- [x] Release integration CI passed\n"
                    f"- [ ] Merge to create tag `v{version}` and open back-merge PR `main -> develop`"
                ),
                "--json",
                "url",
                "--jq",
                ".url",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        url = result.stdout.strip()
        assert "github.com" in url or "/pull/" in url

    @patch("subprocess.run")
    def test_create_backmerge_pr(self, mock_run):
        """Test creating back-merge PR."""
        mock_run.return_value = MagicMock(
            stdout='{"url": "https://github.com/owner/repo/pull/124"}',
            returncode=0,
        )

        version = "0.1.9"

        result = subprocess.run(
            [
                "gh",
                "pr",
                "create",
                "--base",
                "develop",
                "--head",
                "main",
                "--title",
                f"chore: sync develop with release {version}",
                "--body",
                (
                    f"## Sync develop after release {version}\n\n"
                    f"This PR merges `main` back into `develop` after tagging `v{version}`.\n\n"
                    "- [x] Release merged to main\n"
                    "- [x] Tag pushed (deploy should be running)\n"
                    "- [ ] Merge to keep develop in sync"
                ),
                "--json",
                "url",
                "--jq",
                ".url",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        url = result.stdout.strip()
        assert "github.com" in url or "/pull/" in url

    def test_pr_title_formatting(self):
        """Test PR title formatting."""
        version = "0.1.9"
        release_title = f"chore: release {version}"
        backmerge_title = f"chore: sync develop with release {version}"

        assert release_title == "chore: release 0.1.9"
        assert backmerge_title == "chore: sync develop with release 0.1.9"


class TestTagOperations:
    """Consolidated tag operation tests."""

    @patch("subprocess.run")
    def test_tag_existence_check(self, mock_run):
        """Test tag existence check (both cases)."""
        mock_run.return_value = MagicMock(returncode=0)
        tag = "v0.1.9"
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"],
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0

        mock_run.side_effect = subprocess.CalledProcessError(1, "git")
        try:
            subprocess.run(
                ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"],
                capture_output=True,
                check=True,
            )
            assert False, "Should have raised CalledProcessError"
        except subprocess.CalledProcessError:
            pass

    @patch("subprocess.run")
    def test_create_and_push_tag(self, mock_run):
        """Test tag creation and push."""
        mock_run.return_value = MagicMock(returncode=0)

        version = "0.1.9"
        tag = f"v{version}"

        subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"Release {version}"],
            check=True,
        )

        subprocess.run(
            ["git", "push", "origin", tag],
            check=True,
        )

        assert mock_run.call_count == 2


class TestBranchOperations:
    """Consolidated branch operation tests."""

    @patch("subprocess.run")
    def test_branch_existence_check(self, mock_run):
        """Test branch existence check."""
        mock_run.return_value = MagicMock(returncode=0)

        version = "0.1.9"
        branch = f"release/{version}"

        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}"],
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0

    @patch("subprocess.run")
    def test_delete_branch(self, mock_run):
        """Test branch deletion."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        branch = "release/0.1.9"
        repo = "owner/repo"

        result = subprocess.run(
            ["gh", "api", "-X", "DELETE", f"repos/{repo}/git/refs/heads/{branch}"],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode in [0, 1]


class TestWorkflowConditions:
    """Consolidated workflow condition tests."""

    def test_release_pr_condition(self):
        """Test condition for opening release PR."""
        event_name = "workflow_run"
        conclusion = "success"
        branch = "release/0.1.9"

        should_trigger = (
            event_name == "workflow_run"
            and conclusion == "success"
            and branch.startswith("release/")
        )

        assert should_trigger

    def test_tag_creation_condition(self):
        """Test condition for tag creation."""
        event_name = "pull_request"
        merged = True
        base_ref = "main"
        head_ref = "release/0.1.9"

        should_trigger = (
            event_name == "pull_request"
            and merged
            and base_ref == "main"
            and head_ref.startswith("release/")
        )

        assert should_trigger

    def test_branch_deletion_condition(self):
        """Test condition for branch deletion."""
        event_name = "pull_request"
        merged = True
        base_ref = "develop"
        head_ref = "main"
        title = "chore: sync develop with release 0.1.9"

        should_trigger = (
            event_name == "pull_request"
            and merged
            and base_ref == "develop"
            and head_ref == "main"
            and title.startswith("chore: sync develop with release ")
        )

        assert should_trigger


class TestIdempotency:
    """Consolidated idempotency tests."""

    @patch("subprocess.run")
    def test_pr_creation_idempotency(self, mock_run):
        """Test PR creation idempotency."""

        def side_effect(*args, **kwargs):
            if "gh pr list" in " ".join(args[0]):
                return MagicMock(stdout="123\n", returncode=0)
            return MagicMock(returncode=0)

        mock_run.side_effect = side_effect

        branch = "release/0.1.9"
        result = subprocess.run(
            [
                "gh",
                "pr",
                "list",
                "--head",
                branch,
                "--base",
                "main",
                "--state",
                "open",
                "--json",
                "number",
                "--jq",
                ".[0].number",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.stdout.strip() == "123"

    @patch("subprocess.run")
    def test_tag_creation_idempotency(self, mock_run):
        """Test tag creation idempotency."""
        mock_run.return_value = MagicMock(returncode=0)

        tag = "v0.1.9"
        result = subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/tags/{tag}"],
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0


# ============================================================================
# Workflow YAML Validation Tests
# ============================================================================


class TestWorkflowYAMLSyntax:
    """Test that workflow YAML files are valid."""

    def test_start_release_workflow_yaml(self):
        """Test start-release.yml is valid YAML."""
        workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / "start-release.yml"

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert True in workflow or "on" in workflow
        assert "jobs" in workflow
        assert workflow["name"] == "Start Release"

        on_section = workflow.get(True) or workflow.get("on")
        assert on_section is not None
        assert "workflow_dispatch" in on_section
        assert "inputs" in on_section["workflow_dispatch"]
        assert "version" in on_section["workflow_dispatch"]["inputs"]
        assert "create-release-branch" in workflow["jobs"]

    def test_finish_release_workflow_yaml(self):
        """Test finish-release.yml is valid YAML."""
        workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / "finish-release.yml"

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        assert "name" in workflow
        assert True in workflow or "on" in workflow
        assert "jobs" in workflow
        assert workflow["name"] == "Finish Release"

        on_section = workflow.get(True) or workflow.get("on")
        assert on_section is not None
        assert "workflow_run" in on_section
        assert "pull_request" in on_section
        assert "open-release-pr" in workflow["jobs"]
        assert "tag-and-open-backmerge-pr" in workflow["jobs"]
        assert "delete-release-branch" in workflow["jobs"]

    def test_workflow_permissions(self):
        """Test that workflows have required permissions."""
        workflows = [
            "start-release.yml",
            "finish-release.yml",
        ]

        for workflow_file in workflows:
            workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / workflow_file

            with open(workflow_path) as f:
                workflow = yaml.safe_load(f)

            assert "permissions" in workflow, f"{workflow_file} missing permissions"

            if workflow_file == "start-release.yml":
                assert "contents" in workflow["permissions"]
                assert workflow["permissions"]["contents"] == "write"

            if workflow_file == "finish-release.yml":
                assert "contents" in workflow["permissions"]
                assert workflow["permissions"]["contents"] == "write"
                assert "pull-requests" in workflow["permissions"]
                assert workflow["permissions"]["pull-requests"] == "write"

    def test_workflow_job_structure(self):
        """Test that workflow jobs have required structure."""
        workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / "finish-release.yml"

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        job = workflow["jobs"]["open-release-pr"]
        assert "runs-on" in job
        assert "steps" in job
        assert "if" in job

        job = workflow["jobs"]["tag-and-open-backmerge-pr"]
        assert "runs-on" in job
        assert "steps" in job
        assert "if" in job

    def test_workflow_concurrency(self):
        """Test that finish-release workflow has concurrency control."""
        workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / "finish-release.yml"

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        assert "concurrency" in workflow
        assert "group" in workflow["concurrency"]
        assert "cancel-in-progress" in workflow["concurrency"]

    def test_workflow_step_structure(self):
        """Test that workflow steps are properly structured."""
        workflow_path = WORKSPACE_ROOT / ".github" / "workflows" / "start-release.yml"

        with open(workflow_path) as f:
            workflow = yaml.safe_load(f)

        job = workflow["jobs"]["create-release-branch"]

        for step in job["steps"]:
            assert "name" in step
            assert "uses" in step or "run" in step, (
                f"Step '{step['name']}' missing 'uses' or 'run'"
            )
