"""Tests for ``pipelines/pdp/launchers/bundle_from_dab.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAUNCHERS = _REPO_ROOT / "pipelines" / "pdp" / "launchers"
if str(_LAUNCHERS) not in sys.path:
    sys.path.insert(0, str(_LAUNCHERS))

import bundle_from_dab as bfd  # noqa: E402

_FIXTURE_YML = (
    Path(__file__).resolve().parent / "fixtures" / "inference_job_minimal.yml"
)


def test_parse_inference_job_minimal() -> None:
    parsed = bfd.parse_inference_job_from_yaml(_FIXTURE_YML)
    assert parsed["expected_steps"] == ["feature_generation", "inference_h2o"]
    assert parsed["required_runtime"]["databricks_runtime"] == "15.4.x-cpu-ml-scala2.12"
    assert "pandas==2.2.3" in parsed["pypi_packages"]
    assert "databricks_institution_name" in parsed["job_parameters"]
    assert isinstance(parsed["parameter_contract"], list)


def test_build_effective_release(tmp_path: Path) -> None:
    snap = tmp_path / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    snap_yml = snap / "github_pdp_inference.yml"
    snap_yml.write_text(_FIXTURE_YML.read_text(encoding="utf-8"), encoding="utf-8")
    eff = bfd.build_effective_release(tmp_path, "sha123")
    assert "wheel" not in eff
    assert eff["pipeline_version"] == "sha123"
    assert eff["expected_steps"] == ["feature_generation", "inference_h2o"]
    assert eff["entrypoint"] == bfd.DEFAULT_ENTRYPOINT
