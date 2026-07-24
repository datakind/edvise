"""Locate Edvise repo root for launcher imports (entry scripts may lack ``__file__`` on Databricks)."""

from __future__ import annotations

import sys
from pathlib import Path


def edvise_repo_root() -> Path:
    """Repository root (directory containing ``pipelines/``)."""
    return Path(__file__).resolve().parents[3]


def ensure_repo_root_on_sys_path() -> Path:
    """
    Insert repo root and ``src/`` on ``sys.path``.

    Repo root enables ``pipelines.pdp.launchers.*`` imports; ``src/`` enables
    ``import edvise`` (e.g. ``pipeline_runs`` metadata from launcher tasks).
    """
    root = edvise_repo_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    src = root / "src"
    if src.is_dir():
        src_str = str(src)
        if src_str not in sys.path:
            sys.path.insert(0, src_str)
    return root
