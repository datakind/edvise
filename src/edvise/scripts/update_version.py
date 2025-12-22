#!/usr/bin/env python3
"""
Script to update version metadata for releases.
Updates CHANGELOG.md and pyproject.toml with the new version.
Automatically fetches PR titles merged to develop since the last version.
"""

import argparse
import logging
import os
import re
import typing as t
from datetime import datetime
from pathlib import Path

import tomlkit
from tomlkit.items import Table

from edvise.utils import automate_releases

LOGGER = logging.getLogger(__name__)


def update_changelog(
    changelog_path: Path,
    version: str,
    repo: t.Optional[str] = None,
    token: t.Optional[str] = None,
) -> None:
    """Add a new version entry at the top of CHANGELOG.md with PR titles."""
    if not changelog_path.exists():
        raise FileNotFoundError(f"CHANGELOG.md not found at {changelog_path}")

    content = changelog_path.read_text()
    date_str = datetime.now().strftime("%Y-%m-%d")

    # Check if version already exists
    if re.search(rf"## {re.escape(version)}", content):
        LOGGER.warning(f"Version {version} already exists in CHANGELOG.md")
        return

    # Fetch PR titles since last version
    pr_titles = []
    repo_name: str | None = repo or os.getenv("GITHUB_REPOSITORY")
    if repo_name:
        pr_titles = automate_releases.get_pr_titles_since_last_version(repo_name, token)
    else:
        LOGGER.warning("No repository specified, creating empty changelog entry")

    # Format PR titles as bullet points
    if pr_titles:
        bullet_points = "\n".join(f"- {title}" for title in pr_titles)
        new_entry = f"## {version} ({date_str})\n{bullet_points}\n\n"
    else:
        new_entry = f"## {version} ({date_str})\n\n- \n\n"

    # Insert after the first line (or at the beginning if file is empty)
    if content.strip():
        # Find the first version entry to insert before it
        match = re.search(r"^## \d+\.\d+\.\d+", content, re.MULTILINE)
        if match:
            insert_pos = match.start()
            content = content[:insert_pos] + new_entry + content[insert_pos:]
        else:
            # No version entry found, prepend to file
            content = new_entry + content
    else:
        content = new_entry

    changelog_path.write_text(content)
    LOGGER.info(f"Updated CHANGELOG.md with version {version}")
    if pr_titles:
        LOGGER.info(f"Added {len(pr_titles)} PR entries to changelog")


def update_pyproject(pyproject_path: Path, version: str) -> None:
    """Update version in pyproject.toml."""
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")

    content = pyproject_path.read_text()
    doc = tomlkit.parse(content)

    # Update version in [project] section
    if "project" not in doc:
        raise ValueError("pyproject.toml missing [project] section")

    # Cast to MutableMapping for type safety (tomlkit Table implements MutableMapping)
    project_section = t.cast(t.MutableMapping[str, t.Any], doc["project"])
    if not isinstance(project_section, Table):
        raise ValueError("pyproject.toml [project] section is not a table")

    old_version_obj = project_section.get("version", "unknown")
    old_version = str(old_version_obj) if old_version_obj else "unknown"
    project_section["version"] = version

    pyproject_path.write_text(tomlkit.dumps(doc))
    LOGGER.info(f"Updated pyproject.toml version from {old_version} to {version}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update version metadata for releases")
    parser.add_argument("--version", required=True, help="Version number (e.g., 0.1.9)")
    parser.add_argument(
        "--changelog",
        default="CHANGELOG.md",
        help="Path to CHANGELOG.md (default: CHANGELOG.md)",
    )
    parser.add_argument(
        "--pyproject",
        default="pyproject.toml",
        help="Path to pyproject.toml (default: pyproject.toml)",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository (e.g., owner/repo). Defaults to GITHUB_REPOSITORY env var",
    )
    parser.add_argument(
        "--token",
        help="GitHub token for API access. Defaults to GITHUB_TOKEN env var",
    )

    args = parser.parse_args()

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+$", args.version):
        raise ValueError(f"Version must be in format X.Y.Z, got: {args.version}")

    # Configure logging first
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Resolve paths relative to current working directory (should be repo root)
    changelog_path = Path(args.changelog).resolve()
    pyproject_path = Path(args.pyproject).resolve()

    LOGGER.info(f"Using changelog path: {changelog_path}")
    LOGGER.info(f"Using pyproject path: {pyproject_path}")

    # Get repo and token from args or environment
    # Environment variables (automatically set in GitHub Actions):
    # - GITHUB_REPOSITORY: Repository in format "owner/repo" (e.g., "datakind/edvise")
    # - GITHUB_TOKEN: GitHub token for API access (automatically provided by GitHub Actions)
    repo = args.repo or os.getenv("GITHUB_REPOSITORY")
    token = args.token or os.getenv("GITHUB_TOKEN")

    if not repo:
        LOGGER.warning("No repository specified. PR titles will not be fetched.")
    else:
        LOGGER.info(f"Using repository: {repo}")

    if not token:
        LOGGER.warning("No GitHub token provided. PR titles may not be fetched.")
    else:
        LOGGER.info(f"GitHub token provided: Yes (length: {len(token)})")

    update_changelog(changelog_path, args.version, repo, token)
    update_pyproject(pyproject_path, args.version)

    LOGGER.info(f"Successfully updated version to {args.version}")


if __name__ == "__main__":
    main()
