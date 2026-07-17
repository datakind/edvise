"""Tests for per-workspace release bundle paths."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers import release_config as rc


def test_default_release_base_path_known_workspaces() -> None:
    assert rc.default_release_base_path("dev_sst_02").endswith(
        "/dev_sst_02/default/edvise_releases"
    )
    assert rc.default_release_base_path("staging_sst_01").endswith(
        "/staging_sst_01/default/edvise_releases"
    )


def test_resolve_release_base_path_prefers_explicit() -> None:
    assert rc.resolve_release_base_path("dev_sst_02", "/custom/path") == "/custom/path"


def test_default_release_base_path_empty_raises() -> None:
    with pytest.raises(ValueError):
        rc.default_release_base_path("")
