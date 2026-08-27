"""Utilities for release management and version updates."""

import json
import logging
import re
import subprocess
from typing import Literal, Optional

import urllib.error
import urllib.request

LOGGER = logging.getLogger(__name__)

ReleaseBumpType = Literal["initial", "patch", "minor", "major"]
ConventionalBumpType = Literal["patch", "minor", "major"]

_BUMP_RANK: dict[str, int] = {"patch": 0, "minor": 1, "major": 2, "initial": 3}

# Conventional Commits title: type(scope)!: description  (same types as semantic-pr)
_CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(?P<type>[A-Za-z]+)(?:\([^)]*\))?(?P<breaking>!)?:\s+",
)


def parse_semver(version: str) -> tuple[int, int, int]:
    """Parse a X.Y.Z version string into integer components."""
    normalized = version.lstrip("v")
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)\.([0-9]+)", normalized)
    if not match:
        raise ValueError(f"Invalid semver: {version}")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def classify_release_bump(
    current_version: str, previous_version: Optional[str]
) -> ReleaseBumpType:
    """Classify how the current release version changed from the previous tag."""
    if not previous_version:
        return "initial"

    current = parse_semver(current_version)
    previous = parse_semver(previous_version)

    if current[0] > previous[0]:
        return "major"
    if current[1] > previous[1]:
        return "minor"
    if current[2] > previous[2]:
        return "patch"

    raise ValueError(
        f"Version {current_version} is not a forward semver bump from {previous_version}"
    )


def should_run_release_integration(
    current_version: str, previous_version: Optional[str]
) -> bool:
    """Return True when release integration CI should run (major/minor/initial)."""
    bump = classify_release_bump(current_version, previous_version)
    return bump in {"initial", "minor", "major"}


def classify_conventional_titles_bump(titles: list[str]) -> ConventionalBumpType:
    """Classify the highest semver bump implied by conventional PR titles.

    ``feat`` is a minor bump. Any ``type!`` (including ``feat!`` and ``fix!``)
    or a title containing ``BREAKING CHANGE`` is a major bump. Other types
    default to patch.
    """
    bump: ConventionalBumpType = "patch"
    for raw in titles:
        title = raw.strip()
        if "breaking change" in title.lower():
            return "major"
        match = _CONVENTIONAL_COMMIT_RE.match(title)
        if not match:
            continue
        if match.group("breaking"):
            return "major"
        if match.group("type").lower() == "feat":
            bump = "minor"
    return bump


def next_semver(previous_version: str, bump: ConventionalBumpType) -> str:
    """Return the next X.Y.Z version for a conventional bump."""
    major, minor, patch = parse_semver(previous_version)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def assert_numeric_bump_covers_conventional(
    numeric_bump: ReleaseBumpType,
    conventional_bump: ConventionalBumpType,
    *,
    version: str,
    expected_version: Optional[str] = None,
) -> None:
    """Fail when the chosen version is a smaller bump than merged PRs require."""
    if numeric_bump == "initial":
        return
    if _BUMP_RANK[numeric_bump] < _BUMP_RANK[conventional_bump]:
        expected = (
            f" (expected {expected_version} or higher)" if expected_version else ""
        )
        raise ValueError(
            f"Version {version} is a {numeric_bump} bump, but merged PRs require a "
            f"{conventional_bump} bump{expected}."
        )


def resolve_release_version(
    requested_version: Optional[str],
    previous_version: Optional[str],
    pr_titles: list[str],
) -> tuple[str, ReleaseBumpType]:
    """Pick the release version from conventional PR titles.

    When ``requested_version`` is omitted, bump from ``previous_version`` using
    ``feat`` → minor and ``type!`` → major. An explicit version may be higher
    than that, but must not under-bump.
    """
    if not previous_version:
        if not requested_version:
            raise ValueError("Explicit version is required for the initial release")
        parse_semver(requested_version)
        return requested_version.lstrip("v"), "initial"

    inferred = classify_conventional_titles_bump(pr_titles)
    computed = next_semver(previous_version, inferred)
    if not requested_version:
        return computed, inferred

    requested = requested_version.lstrip("v")
    numeric = classify_release_bump(requested, previous_version)
    assert_numeric_bump_covers_conventional(
        numeric,
        inferred,
        version=requested,
        expected_version=computed,
    )
    return requested, numeric


def collect_pr_titles_for_release(
    repo: str, token: str, since_tag: Optional[str] = None
) -> list[str]:
    """Return every merged PR title since ``since_tag``.

    Fails closed: git-history errors and any missing title raise, so bump
    inference never runs on a partial set of conventional-commit titles.
    """
    pr_numbers = get_pr_numbers_from_git_log(since_tag, strict=True)
    titles: list[str] = []
    missing: list[int] = []
    for pr_num in pr_numbers:
        title = fetch_pr_title(repo, pr_num, token)
        if title:
            titles.append(title)
        else:
            missing.append(pr_num)
    if missing:
        raise RuntimeError(
            "Failed to fetch PR title(s) for "
            f"{missing}; cannot infer the release bump from incomplete data."
        )
    return titles


def plan_release(
    repo: str,
    token: Optional[str] = None,
    requested_version: Optional[str] = None,
) -> tuple[str, ReleaseBumpType]:
    """Resolve the next release version from git tags and merged PR titles."""
    last_tag = get_last_version_tag(strict=True)
    previous = last_tag.lstrip("v") if last_tag else None

    titles: list[str] = []
    if token:
        titles = collect_pr_titles_for_release(repo, token, last_tag)
    elif not requested_version:
        raise ValueError(
            "GitHub token is required to infer the release version from PR titles"
        )

    version, bump = resolve_release_version(requested_version, previous, titles)
    LOGGER.info(
        "Planned release %s (%s) from previous %s using %s PR title(s)",
        version,
        bump,
        previous or "none",
        len(titles),
    )
    return version, bump


def get_last_version_tag(*, strict: bool = False) -> Optional[str]:
    """Get the last version tag from git.

    Empty tag lists return ``None``. When ``strict`` is True, a failed git
    command raises instead of looking like "no previous release".
    """
    try:
        result = subprocess.run(
            ["git", "tag", "--sort=-version:refname", "--list", "v*"],
            capture_output=True,
            text=True,
            check=True,
        )
        tags = [tag.strip() for tag in result.stdout.strip().split("\n") if tag.strip()]
        return tags[0] if tags else None
    except subprocess.CalledProcessError as exc:
        if strict:
            raise RuntimeError("Failed to list version tags from git") from exc
        return None


def get_pr_numbers_from_git_log(
    since_tag: Optional[str] = None, *, strict: bool = False
) -> list[int]:
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

        pr_numbers: set[int] = set()
        for line in result.stdout.splitlines():
            m = re.search(r"^Merge pull request #(\d+)\b", line.strip())
            if m:
                pr_numbers.add(int(m.group(1)))

        return sorted(pr_numbers)
    except subprocess.CalledProcessError as exc:
        if strict:
            raise RuntimeError(
                "Failed to read git history for merged PRs; cannot infer the "
                "release bump from incomplete data."
            ) from exc
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
            pr_data: dict[str, object] = json.loads(response_data)
            title_obj = pr_data.get("title")
            if title_obj and isinstance(title_obj, str):
                title: str = title_obj
                LOGGER.debug(f"Successfully fetched PR #{pr_number}: {title}")
                return title
            else:
                LOGGER.warning(f"PR #{pr_number} has no title in API response")
                return None
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
        LOGGER.info(
            f"Token provided: {'Yes' if token else 'No'} (length: {len(token) if token else 0})"
        )
        LOGGER.info(
            f"Fetching titles for {len(pr_numbers)} PRs using GitHub API from {repo}"
        )
        pr_titles = []
        failed_count = 0
        skipped_count = 0
        for pr_num in pr_numbers:
            title = fetch_pr_title(repo, pr_num, token)
            if title:
                # Filter out chore PRs (maintenance tasks that don't affect users)
                if title.lower().startswith("chore:"):
                    LOGGER.debug(f"Skipping chore PR #{pr_num}: {title}")
                    skipped_count += 1
                    continue

                # Append PR number to title for easy tracking
                pr_titles.append(f"{title} (#{pr_num})")
                LOGGER.debug(f"Fetched PR #{pr_num}: {title}")
            else:
                failed_count += 1
                # Fallback to PR number if fetch fails
                pr_titles.append(f"PR #{pr_num}")

        if skipped_count > 0:
            LOGGER.info(f"Skipped {skipped_count} chore PR(s)")
        if failed_count > 0:
            LOGGER.warning(
                f"Failed to fetch {failed_count} out of {len(pr_numbers)} PR titles. "
                f"Using PR numbers as fallback."
            )
        else:
            LOGGER.info(
                f"Successfully fetched {len(pr_titles)} PR title(s) for changelog"
            )

        return pr_titles
    else:
        LOGGER.warning("No GitHub token provided, using PR numbers as titles")
        # No token, return PR numbers as titles
        return [f"PR #{num}" for num in pr_numbers]
