"""Tests for versioned inference launcher path bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

from pipelines.pdp.launchers._paths import (
    edvise_repo_root,
    ensure_repo_root_on_sys_path,
)


def test_edvise_repo_root_contains_pipelines() -> None:
    root = edvise_repo_root()
    assert (root / "pipelines" / "pdp" / "launchers").is_dir()


def test_ensure_repo_root_on_sys_path_adds_src_for_edvise_import() -> None:
    root = ensure_repo_root_on_sys_path()
    src_str = str(root / "src")
    assert src_str in sys.path
    assert (root / "src" / "edvise").is_dir()
