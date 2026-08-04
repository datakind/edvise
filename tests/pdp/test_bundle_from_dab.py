"""Tests for ``edvise.runtime.versioned_inference.bundle.from_dab``."""

from __future__ import annotations

from pathlib import Path

from edvise.runtime.versioned_inference.bundle import from_dab as bfd

_FIXTURE_YML = (
    Path(__file__).resolve().parent / "fixtures" / "inference_job_minimal.yml"
)


def test_parse_inference_job_minimal() -> None:
    parsed = bfd.parse_inference_job_from_yaml(_FIXTURE_YML)
    assert parsed["job_key"] == bfd.DEFAULT_INFERENCE_JOB_KEY
    assert parsed["expected_steps"]
    assert parsed["execution_mode"] == "git_submit"


def test_build_effective_release(tmp_path: Path) -> None:
    release_dir = tmp_path / "v1"
    snap = release_dir / "databricks_bundle_snapshot" / "resources"
    snap.mkdir(parents=True)
    (snap / "github_pdp_inference.yml").write_text(
        _FIXTURE_YML.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    effective = bfd.build_effective_release(release_dir, "v1")
    assert effective["pipeline_version"] == "v1"
    assert effective["expected_steps"]
