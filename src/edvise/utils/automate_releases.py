"""Utilities for release management and version updates."""

import json
import logging
import re
import subprocess
from typing import Optional

import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)


def get_last_version_tag() -> Optional[str]:
    """Get the last version tag from git."""
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-version:refname", "--list", "v*"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
        return tags[0] if tags else None
    except subprocess.CalledProcessError:
        return None


def get_pr_numbers_from_git_log(since_tag: Optional[str] = None) -> list[int]:
    """Extract PR numbers from git log merge commits since a tag."""
    if since_tag:
        log_range = f"{since_tag}..origin/develop"
    else:
        log_range = "origin/develop"

    try:
        result = subprocess.run(
            ["git", "log", log_range, "--merges", "--pretty=format:%s"],
            capture_output=True,
            text=True,
            check=True,
        )

        pr_numbers = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                # Look for PR number in format #N
                match = re.search(r"#(\d+)", line)
                if match:
                    pr_num = int(match.group(1))
                    if pr_num not in pr_numbers:
                        pr_numbers.append(pr_num)

        return pr_numbers
    except subprocess.CalledProcessError:
        return []


def fetch_pr_title(repo: str, pr_number: int, token: str) -> Optional[str]:
    """Fetch PR title from GitHub API."""
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}"
    LOGGER.debug(f"Fetching PR #{pr_number} from {repo} via {url}")
    
    req = urllib.request.Request(url)
    # Use Bearer format for all tokens (GitHub accepts both, but Bearer is preferred)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")
    req.add_header("User-Agent", "edvise-release-automation")
    
    try:
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode()
            pr_data = json.loads(response_data)
            title = pr_data.get("title")
            if not title:
                LOGGER.warning(f"PR #{pr_number} has no title in API response")
            else:
                LOGGER.debug(f"Successfully fetched PR #{pr_number}: {title}")
            return title
    except urllib.error.HTTPError as e:
        # Read error response before logging
        error_body = ""
        try:
            error_body = e.read().decode()
            error_json = json.loads(error_body)
            error_message = error_json.get("message", error_body)
        except Exception:
            error_message = f"Could not parse error response: {error_body or str(e)}"
        
        if e.code == 404:
            LOGGER.warning(
                f"PR #{pr_number} not found in {repo}. "
                f"Error: {error_message}. "
                f"URL was: {url}"
            )
        elif e.code == 401:
            LOGGER.warning(
                f"Authentication failed for PR #{pr_number}. "
                f"Check token permissions. Error: {error_message}"
            )
        elif e.code == 403:
            LOGGER.warning(
                f"Access forbidden for PR #{pr_number}. "
                f"Token may lack required permissions. Error: {error_message}"
            )
        else:
            LOGGER.warning(
                f"Failed to fetch PR #{pr_number} from {repo}: HTTP {e.code} - {e.reason}. "
                f"Error: {error_message}"
            )
        return None
    except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
        LOGGER.warning(f"Failed to fetch PR #{pr_number}: {type(e).__name__} - {e}")
        return None


def get_pr_titles_since_last_version(
    repo: str, token: Optional[str] = None
) -> list[str]:
    """Get PR titles merged to develop since the last version tag."""
    last_tag = get_last_version_tag()

    if not last_tag:
        LOGGER.warning(
            "No previous version tag found, fetching all PRs merged to develop"
        )
        since = None
    else:
        LOGGER.info(f"Fetching PRs merged to develop since {last_tag}")
        since = last_tag

    pr_numbers = get_pr_numbers_from_git_log(since)

    if not pr_numbers:
        LOGGER.warning("No PR numbers found in merge commits")
        return []

    LOGGER.info(f"Found {len(pr_numbers)} PR numbers: {pr_numbers}")
    LOGGER.info(f"Using repository: {repo}")

    # Fetch PR titles using GitHub API
    if token:
        LOGGER.info(f"Token provided: {'Yes' if token else 'No'} (length: {len(token) if token else 0})")
        LOGGER.info(f"Fetching titles for {len(pr_numbers)} PRs using GitHub API from {repo}")
        pr_titles = []
        failed_count = 0
        for pr_num in pr_numbers:
            title = fetch_pr_title(repo, pr_num, token)
            if title:
                # Append PR number to title for easy tracking
                pr_titles.append(f"{title} (#{pr_num})")
                LOGGER.debug(f"Fetched PR #{pr_num}: {title}")
            else:
                failed_count += 1
                # Fallback to PR number if fetch fails
                pr_titles.append(f"PR #{pr_num}")
        
        if failed_count > 0:
            LOGGER.warning(
                f"Failed to fetch {failed_count} out of {len(pr_numbers)} PR titles. "
                f"Using PR numbers as fallback."
            )
        else:
            LOGGER.info(f"Successfully fetched all {len(pr_titles)} PR titles")
        
        return pr_titles
    else:
        LOGGER.warning("No GitHub token provided, using PR numbers as titles")
        # No token, return PR numbers as titles
        return [f"PR #{num}" for num in pr_numbers]
