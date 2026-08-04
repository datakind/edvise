"""
Materialize ``databricks_bundle_snapshot/`` on a release volume path.

Fetches archived DAB YAML from GitHub at ``pipeline_version`` (Git SHA or tag).
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from pathlib import Path

from edvise.runtime.versioned_inference.bundle.from_dab import inference_yml_path
from edvise.runtime.versioned_inference.dab_layout import (
    DabBundleLayout,
    default_dab_bundle_layout,
    resolve_dab_bundle_layout,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO = "datakind/edvise"


def github_raw_url(github_repo: str, git_ref: str, repo_path: str) -> str:
    """``git_ref`` may be a commit SHA or a release tag."""
    return f"https://raw.githubusercontent.com/{github_repo}/{git_ref}/{repo_path}"


def fetch_github_file(
    github_repo: str,
    git_ref: str,
    repo_path: str,
    *,
    token: str | None = None,
    timeout_s: int = 120,
) -> bytes:
    url = github_raw_url(github_repo, git_ref, repo_path)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return bytes(resp.read())


def materialize_dab_snapshot_from_github(
    release_dir: Path,
    git_ref: str,
    *,
    layout: DabBundleLayout | None = None,
    github_repo: str = DEFAULT_GITHUB_REPO,
    skip_if_present: bool = True,
    token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Download DAB YAML files into ``release_dir/databricks_bundle_snapshot/``."""
    resolved_layout = layout or default_dab_bundle_layout()
    marker = inference_yml_path(release_dir, resolved_layout.inference_yml_snapshot_rel)
    if skip_if_present and marker.is_file():
        logger.info(
            "DAB snapshot already present at %s; skipping GitHub fetch.", marker
        )
        return

    for repo_path, rel_dest in resolved_layout.github_snapshot_sources().items():
        dest = release_dir / rel_dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Fetching %s from GitHub (%s @ %s)", repo_path, github_repo, git_ref
        )
        try:
            content = fetch_github_file(github_repo, git_ref, repo_path, token=token)
        except urllib.error.HTTPError as exc:
            msg = f"GitHub fetch failed for {repo_path} at {git_ref}: HTTP {exc.code}"
            raise OSError(msg) from exc
        dest.write_bytes(content)
        logger.info("Wrote %s (%s bytes)", dest, len(content))


def materialize_runtime_bundle_dir(
    release_dir: Path,
    pipeline_version: str,
    *,
    schema_type: str = "pdp",
    github_repo: str | None = None,
    git_ref: str | None = None,
    skip_snapshot_if_present: bool = True,
    github_token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> Path:
    """Ensure ``release_dir`` contains ``databricks_bundle_snapshot/`` (DAB YAML only)."""
    release_dir.mkdir(parents=True, exist_ok=True)
    layout = resolve_dab_bundle_layout(schema_type)
    token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    resolved_ref = (git_ref or "").strip() or None
    if not resolved_ref:
        msg = "materialize_runtime_bundle_dir requires git_ref"
        raise ValueError(msg)

    materialize_dab_snapshot_from_github(
        release_dir,
        resolved_ref,
        layout=layout,
        github_repo=github_repo or DEFAULT_GITHUB_REPO,
        skip_if_present=skip_snapshot_if_present,
        token=token,
        logger=logger,
    )

    logger.info(
        "Runtime bundle snapshot ready at %s (pipeline_version=%s, schema_type=%s)",
        release_dir,
        pipeline_version,
        layout.schema_type,
    )
    return release_dir
