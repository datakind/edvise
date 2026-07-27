"""Git ref helpers for versioned inference (dev SHA vs staging/prod tag)."""

from __future__ import annotations

import re

_GIT_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def is_git_commit_sha(ref: str) -> bool:
    """Return whether ``ref`` looks like a Git commit SHA (7–40 hex chars)."""
    return bool(_GIT_SHA_RE.match((ref or "").strip()))


def sanitize_release_dir_name(pipeline_version: str) -> str:
    """Filesystem-safe release bundle folder name for a pipeline_version ref."""
    return (pipeline_version or "").strip().replace("/", "_").replace("\\", "_")


def build_git_source(git_url: str, pipeline_version: str) -> dict[str, str]:
    """
    Build Databricks ``git_source`` for ``runs/submit``.

    Dev models store ``pipeline_version`` as a Git SHA; staging/prod models store a
    release tag — same as PDP/ES training and inference DAB targets.
    """
    pv = (pipeline_version or "").strip()
    if not pv:
        msg = "pipeline_version must be non-empty for git_source"
        raise ValueError(msg)
    source: dict[str, str] = {
        "git_url": git_url.rstrip("/"),
        "git_provider": "gitHub",
    }
    if is_git_commit_sha(pv):
        source["git_commit"] = pv
    else:
        source["git_tag"] = pv
    return source


def git_ref_kind(pipeline_version: str) -> str:
    """Human-readable ref kind for logs (``sha`` or ``tag``)."""
    return "sha" if is_git_commit_sha(pipeline_version) else "tag"
