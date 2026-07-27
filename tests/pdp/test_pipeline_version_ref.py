"""Tests for Git ref handling (dev SHA vs staging tag)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers import pipeline_version_ref as pvr


def test_is_git_commit_sha_full_and_short() -> None:
    assert pvr.is_git_commit_sha("87b641939205110d03ce8c300e68980327dd6732")
    assert pvr.is_git_commit_sha("87b6419")
    assert not pvr.is_git_commit_sha("v1.2.3")
    assert not pvr.is_git_commit_sha("abc123sha456")


def test_build_git_source_sha_vs_tag() -> None:
    sha = "87b641939205110d03ce8c300e68980327dd6732"
    sha_src = pvr.build_git_source("https://github.com/datakind/edvise", sha)
    assert sha_src["git_commit"] == sha
    assert "git_tag" not in sha_src

    tag_src = pvr.build_git_source("https://github.com/datakind/edvise", "v2.4.0")
    assert tag_src["git_tag"] == "v2.4.0"
    assert "git_commit" not in tag_src


def test_sanitize_release_dir_name() -> None:
    assert pvr.sanitize_release_dir_name("v1.0/release") == "v1.0_release"
    assert pvr.git_ref_kind("87b641939205110d03ce8c300e68980327dd6732") == "sha"
    assert pvr.git_ref_kind("v2.4.0") == "tag"
