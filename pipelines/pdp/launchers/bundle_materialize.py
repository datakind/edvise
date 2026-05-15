"""
Materialize ``databricks_bundle_snapshot/`` and ``release.json`` on a release volume path.

On Databricks: fetch YAML from GitHub at ``pipeline_version`` (git SHA in dev).
Locally: copy from a checked-out ``repo_root`` (see ``materialize_release_bundle.py``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

try:
    from pipelines.pdp.launchers.bundle_from_dab import (
        DEFAULT_ENTRYPOINT,
        discover_wheel_filename,
        release_json_path,
        write_release_json,
    )
except ImportError:  # flat import when only launchers/ is on sys.path
    from bundle_from_dab import (
        DEFAULT_ENTRYPOINT,
        discover_wheel_filename,
        release_json_path,
        write_release_json,
    )
    from release_deps import (  # noqa: F401
        PYPROJECT_FILENAME,
        RELEASE_REQUIREMENTS_FILENAME,
        pyproject_path,
        release_requirements_path,
        write_release_requirements,
    )
else:
    from pipelines.pdp.launchers.release_deps import (
        PYPROJECT_FILENAME,
        RELEASE_REQUIREMENTS_FILENAME,
        pyproject_path,
        release_requirements_path,
        write_release_requirements,
    )

LOGGER = logging.getLogger(__name__)

DEFAULT_GITHUB_REPO = "datakind/edvise"

DAB_SNAPSHOT_FILES = (
    "pipelines/pdp/databricks.yml",
    "pipelines/pdp/resources/github_pdp_inference.yml",
)

INFERENCE_YML_REL = "databricks_bundle_snapshot/resources/github_pdp_inference.yml"


def inference_yml_in_bundle(release_dir: Path) -> Path:
    return release_dir / INFERENCE_YML_REL


def github_raw_url(github_repo: str, git_sha: str, repo_path: str) -> str:
    return f"https://raw.githubusercontent.com/{github_repo}/{git_sha}/{repo_path}"


def fetch_github_file(
    github_repo: str,
    git_sha: str,
    repo_path: str,
    *,
    token: str | None = None,
    timeout_s: int = 120,
) -> bytes:
    url = github_raw_url(github_repo, git_sha, repo_path)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def materialize_dab_snapshot_from_github(
    release_dir: Path,
    git_sha: str,
    *,
    github_repo: str = DEFAULT_GITHUB_REPO,
    skip_if_present: bool = True,
    token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Download DAB YAML files into ``release_dir/databricks_bundle_snapshot/``."""
    marker = inference_yml_in_bundle(release_dir)
    if skip_if_present and marker.is_file():
        logger.info("DAB snapshot already present at %s; skipping GitHub fetch.", marker)
        return

    snap_root = release_dir / "databricks_bundle_snapshot"
    snap_root.mkdir(parents=True, exist_ok=True)

    for repo_path in DAB_SNAPSHOT_FILES:
        dest = release_dir / repo_path.replace("pipelines/pdp/", "databricks_bundle_snapshot/")
        if repo_path.endswith("databricks.yml"):
            dest = snap_root / "databricks.yml"
        elif "github_pdp_inference.yml" in repo_path:
            dest = snap_root / "resources" / "github_pdp_inference.yml"
        else:
            dest = snap_root / Path(repo_path).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching %s from GitHub (%s @ %s)", repo_path, github_repo, git_sha)
        try:
            content = fetch_github_file(
                github_repo, git_sha, repo_path, token=token
            )
        except urllib.error.HTTPError as exc:
            msg = f"GitHub fetch failed for {repo_path} at {git_sha}: HTTP {exc.code}"
            raise OSError(msg) from exc
        dest.write_bytes(content)
        logger.info("Wrote %s (%s bytes)", dest, len(content))


def materialize_pyproject_from_github(
    release_dir: Path,
    git_sha: str,
    *,
    github_repo: str = DEFAULT_GITHUB_REPO,
    token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> None:
    """Fetch repo-root ``pyproject.toml`` and write ``release_requirements.txt``."""
    dest = pyproject_path(release_dir)
    logger.info(
        "Fetching %s from GitHub (%s @ %s)",
        PYPROJECT_FILENAME,
        github_repo,
        git_sha,
    )
    try:
        content = fetch_github_file(
            github_repo, git_sha, PYPROJECT_FILENAME, token=token
        )
    except urllib.error.HTTPError as exc:
        msg = f"GitHub fetch failed for {PYPROJECT_FILENAME} at {git_sha}: HTTP {exc.code}"
        raise OSError(msg) from exc
    dest.write_bytes(content)
    write_release_requirements(
        dest, release_requirements_path(release_dir), logger=logger
    )


def materialize_pyproject_from_repo_root(
    release_dir: Path,
    repo_root: Path,
    *,
    logger: logging.Logger = LOGGER,
) -> None:
    """Copy ``pyproject.toml`` from a local checkout and write requirements."""
    src = repo_root.expanduser().resolve() / PYPROJECT_FILENAME
    if not src.is_file():
        msg = f"{PYPROJECT_FILENAME} not found at {src}"
        raise FileNotFoundError(msg)
    shutil.copy2(src, pyproject_path(release_dir))
    write_release_requirements(
        pyproject_path(release_dir),
        release_requirements_path(release_dir),
        logger=logger,
    )


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


def write_minimal_release_json(
    release_dir: Path,
    *,
    wheel: str | None,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    logger: logging.Logger = LOGGER,
) -> dict[str, Any]:
    """Write ``release.json``; discover wheel in ``release_dir`` when omitted."""
    wheel_name = wheel or discover_wheel_filename(release_dir, None)
    body: dict[str, Any] = {"entrypoint": entrypoint}
    if release_requirements_path(release_dir).is_file():
        body["requirements"] = RELEASE_REQUIREMENTS_FILENAME
    if wheel_name:
        body["wheel"] = wheel_name
    write_release_json(release_json_path(release_dir), body)
    if not wheel_name:
        logger.warning(
            "No wheel in %s yet — upload *.whl before the launcher task runs.",
            release_dir,
        )
    else:
        logger.info("release.json wheel=%s", wheel_name)
    return body


def materialize_runtime_bundle_dir(
    release_dir: Path,
    pipeline_version: str,
    *,
    github_repo: str | None = None,
    git_sha: str | None = None,
    repo_root: Path | None = None,
    wheel: str | None = None,
    entrypoint: str = DEFAULT_ENTRYPOINT,
    skip_snapshot_if_present: bool = True,
    github_token: str | None = None,
    logger: logging.Logger = LOGGER,
) -> Path:
    """
    Ensure ``release_dir`` contains DAB snapshot + ``release.json``.

    Provide ``git_sha`` (Databricks) **or** ``repo_root`` (local publish script).
    """
    release_dir.mkdir(parents=True, exist_ok=True)
    token = github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")

    if repo_root is not None:
        materialize_dab_snapshot_from_repo_root(
            release_dir,
            repo_root.expanduser().resolve(),
            skip_if_present=skip_snapshot_if_present,
            logger=logger,
        )
    elif git_sha:
        materialize_dab_snapshot_from_github(
            release_dir,
            git_sha,
            github_repo=github_repo or DEFAULT_GITHUB_REPO,
            skip_if_present=skip_snapshot_if_present,
            token=token,
            logger=logger,
        )
    else:
        msg = "materialize_runtime_bundle_dir requires git_sha or repo_root"
        raise ValueError(msg)

    if repo_root is not None:
        materialize_pyproject_from_repo_root(
            release_dir, repo_root.expanduser().resolve(), logger=logger
        )
    elif git_sha:
        materialize_pyproject_from_github(
            release_dir,
            git_sha,
            github_repo=github_repo or DEFAULT_GITHUB_REPO,
            token=token,
            logger=logger,
        )

    write_minimal_release_json(
        release_dir, wheel=wheel, entrypoint=entrypoint, logger=logger
    )
    logger.info(
        "Runtime bundle ready at %s (pipeline_version=%s)",
        release_dir,
        pipeline_version,
    )
    return release_dir
