"""Tests for pyproject.toml → release_requirements.txt."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.release_deps import (  # noqa: E402
    parse_pyproject_dependencies,
    write_release_requirements,
)


def test_parse_pyproject_dependencies_reads_project_table() -> None:
    text = """
[project]
name = "edvise"
dependencies = [
  "pandas==2.2.3",
  "pydantic~=2.10",
]
"""
    deps = parse_pyproject_dependencies(text)
    assert deps == ["pandas==2.2.3", "pydantic~=2.10"]


def test_write_release_requirements(tmp_path: Path) -> None:
    pp = tmp_path / "pyproject.toml"
    pp.write_text(
        '[project]\nname = "x"\ndependencies = ["numpy==1.26.4"]\n',
        encoding="utf-8",
    )
    out = tmp_path / "release_requirements.txt"
    write_release_requirements(pp, out)
    assert out.read_text(encoding="utf-8").strip() == "numpy==1.26.4"
