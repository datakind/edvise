"""Tests for runtime bundle materialization (DAB YAML snapshot only)."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.bundle_materialize import (  # noqa: E402
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
        "pipelines.pdp.launchers.bundle_materialize.fetch_github_file",
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

    with patch(
        "pipelines.pdp.launchers.bundle_materialize.fetch_github_file"
    ) as fetch:
        materialize_dab_snapshot_from_github(release_dir, "sha", skip_if_present=True)
        fetch.assert_not_called()


def test_materialize_runtime_bundle_dir_local(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    (repo / "pipelines/pdp/resources").mkdir(parents=True)
    (repo / "pipelines/pdp/databricks.yml").write_text("bundle: x\n", encoding="utf-8")
    (repo / "pipelines/pdp/resources/github_pdp_inference.yml").write_text(
        _FIXTURE_YML.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    out = tmp_path / "release" / "sha1"
    out.mkdir(parents=True)

    materialize_runtime_bundle_dir(
        out,
        "sha1",
        repo_root=repo,
        skip_snapshot_if_present=False,
    )

    assert (
        out / "databricks_bundle_snapshot/resources/github_pdp_inference.yml"
    ).is_file()
    assert not (out / "release.json").exists()
    assert not (out / "pyproject.toml").exists()
    assert not (out / "release_requirements.txt").exists()
