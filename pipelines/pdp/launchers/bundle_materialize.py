"""
Materialize ``databricks_bundle_snapshot/`` on a release volume path.

On Databricks: fetch bundle YAML from GitHub at ``pipeline_version`` (Git SHA or tag).
Locally: copy from a checked-out ``repo_root`` (see ``materialize_release_bundle.py``).

Inference is submitted from Git at that SHA using the archived job YAML; no wheel,
``release.json``, ``pyproject.toml``, or ``release_requirements.txt`` are written here.
"""

from __future__ import annotations

import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO = "datakind/edvise"

DAB_SNAPSHOT_FILES = (
    "pipelines/pdp/databricks.yml",
    "pipelines/pdp/resources/github_pdp_inference.yml",
)

INFERENCE_YML_REL = "databricks_bundle_snapshot/resources/github_pdp_inference.yml"


def inference_yml_in_bundle(release_dir: Path) -> Path:
    return release_dir / INFERENCE_YML_REL


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
        return resp.read()


def materialize_dab_snapshot_from_github(
    release_dir: Path,
    git_ref: str,
    *,
    github_repo: str = DEFAULT_GITHUB_REPO,
    skip_if_present: bool = True,
    token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Download DAB YAML files into ``release_dir/databricks_bundle_snapshot/``."""
    marker = inference_yml_in_bundle(release_dir)
    if skip_if_present and marker.is_file():
        logger.info(
            "DAB snapshot already present at %s; skipping GitHub fetch.", marker
        )
        return

    snap_root = release_dir / "databricks_bundle_snapshot"
    snap_root.mkdir(parents=True, exist_ok=True)

    for repo_path in DAB_SNAPSHOT_FILES:
        dest = release_dir / repo_path.replace(
            "pipelines/pdp/", "databricks_bundle_snapshot/"
        )
        if repo_path.endswith("databricks.yml"):
            dest = snap_root / "databricks.yml"
        elif "github_pdp_inference.yml" in repo_path:
            dest = snap_root / "resources" / "github_pdp_inference.yml"
        else:
            dest = snap_root / Path(repo_path).name
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


def materialize_dab_snapshot_from_repo_root(
    release_dir: Path,
    repo_root: Path,
    *,
    skip_if_present: bool = True,
    logger: logging.Logger = LOGGER,
) -> None:
    """Copy DAB YAML from a local Edvise checkout (``materialize_release_bundle`` CLI)."""
    marker = inference_yml_in_bundle(release_dir)
    if skip_if_present and marker.is_file():
        logger.info("DAB snapshot already present at %s; skipping copy.", marker)
        return

    snap_dst = release_dir / "databricks_bundle_snapshot"
    if snap_dst.exists():
        shutil.rmtree(snap_dst)
    snap_dst.mkdir(parents=True)

    dab_yml = repo_root / "pipelines" / "pdp" / "databricks.yml"
    inf_yml = repo_root / "pipelines" / "pdp" / "resources" / "github_pdp_inference.yml"
    if dab_yml.is_file():
        shutil.copy2(dab_yml, snap_dst / "databricks.yml")
    if inf_yml.is_file():
        (snap_dst / "resources").mkdir(parents=True, exist_ok=True)
        shutil.copy2(inf_yml, snap_dst / "resources" / "github_pdp_inference.yml")


def materialize_runtime_bundle_dir(
    release_dir: Path,
    pipeline_version: str,
    *,
    github_repo: str | None = None,
    git_ref: str | None = None,
    git_sha: str | None = None,
    repo_root: Path | None = None,
    skip_snapshot_if_present: bool = True,
    github_token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> Path:
    """
    Ensure ``release_dir`` contains ``databricks_bundle_snapshot/`` (DAB YAML only).

    Provide ``git_ref`` / ``git_sha`` (Databricks) **or** ``repo_root`` (local publish script).
    """
    release_dir.mkdir(parents=True, exist_ok=True)
    token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    resolved_ref = (git_ref or git_sha or "").strip() or None

    if repo_root is not None:
        materialize_dab_snapshot_from_repo_root(
            release_dir,
            repo_root.expanduser().resolve(),
            skip_if_present=skip_snapshot_if_present,
            logger=logger,
        )
    elif resolved_ref:
        materialize_dab_snapshot_from_github(
            release_dir,
            resolved_ref,
            github_repo=github_repo or DEFAULT_GITHUB_REPO,
            skip_if_present=skip_snapshot_if_present,
            token=token,
            logger=logger,
        )
    else:
        msg = "materialize_runtime_bundle_dir requires git_ref/git_sha or repo_root"
        raise ValueError(msg)

    logger.info(
        "Runtime bundle snapshot ready at %s (pipeline_version=%s)",
        release_dir,
        pipeline_version,
    )
    return release_dir
