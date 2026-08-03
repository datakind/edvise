"""Tests for runtime bundle materialization (DAB YAML snapshot only)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference.bundle.materialize import (
    github_raw_url,
    materialize_dab_snapshot_from_github,
    materialize_runtime_bundle_dir,
)

_FIXTURE_YML = (
    Path(__file__).resolve().parent / "fixtures" / "inference_job_minimal.yml"
)


def test_github_raw_url() -> None:
    url = github_raw_url("datakind/edvise", "abc123", "pipelines/pdp/databricks.yml")
    assert url == (
        "https://raw.githubusercontent.com/datakind/edvise/"
        "abc123/pipelines/pdp/databricks.yml"
    )


def test_materialize_dab_snapshot_from_github(tmp_path: Path) -> None:
    release_dir = tmp_path / "abc123"
    yml_bytes = _FIXTURE_YML.read_bytes()
    dab_bytes = b"bundle:\n  name: test\n"

    def fake_fetch(repo: str, sha: str, path: str, **kwargs: object) -> bytes:
        if path.endswith("github_pdp_inference.yml"):
            return yml_bytes
        return dab_bytes

    with patch(
        "edvise.runtime.versioned_inference.bundle.materialize.fetch_github_file",
        side_effect=fake_fetch,
    ):
        materialize_dab_snapshot_from_github(
            release_dir, "abc123", skip_if_present=False
        )

    inf = release_dir / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
    assert inf.is_file()
    assert inf.read_bytes() == yml_bytes
    dab = release_dir / "databricks_bundle_snapshot/databricks.yml"
    assert dab.read_bytes() == dab_bytes


def test_materialize_skips_when_snapshot_present(tmp_path: Path) -> None:
    release_dir = tmp_path / "sha"
    inf = release_dir / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
    inf.parent.mkdir(parents=True)
    inf.write_text("existing", encoding="utf-8")

    with patch("edvise.runtime.versioned_inference.bundle.materialize.fetch_github_file") as fetch:
        materialize_dab_snapshot_from_github(release_dir, "sha", skip_if_present=True)
        fetch.assert_not_called()


def test_materialize_runtime_bundle_dir_requires_git_ref(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="git_ref"):
        materialize_runtime_bundle_dir(tmp_path / "release", "sha1")
