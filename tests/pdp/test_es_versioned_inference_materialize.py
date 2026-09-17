"""Tests for ES versioned-inference materialize paths (PDP-safe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from edvise.runtime.versioned_inference.bundle.materialize import (  # noqa: E402
    materialize_runtime_bundle_dir,
)
from edvise.runtime.versioned_inference.dab_layout import (  # noqa: E402
    GENAI_SNAPSHOT_DIRNAME,
    genai_execute_dab_bundle_layout,
    genai_snapshot_dir,
    resolve_dab_bundle_layout,
)
from edvise.runtime.versioned_inference.genai_registry import (  # noqa: E402
    genai_active_root,
    resolve_genai_pipeline_version_from_registry,
    silver_volume_root,
)
from edvise.runtime.versioned_inference.release_config import (  # noqa: E402
    default_es_release_base_path,
    default_release_base_path,
    resolve_es_release_base_path,
)


def test_es_release_base_path_is_under_edvise_releases_es() -> None:
    assert default_es_release_base_path("dev_sst_02") == (
        "/Volumes/dev_sst_02/default/edvise_releases/es"
    )
    # PDP default must remain without the /es segment.
    assert default_release_base_path("dev_sst_02") == (
        "/Volumes/dev_sst_02/default/edvise_releases"
    )


def test_resolve_es_release_base_path_prefers_explicit() -> None:
    assert (
        resolve_es_release_base_path("dev_sst_02", "/Volumes/custom/es")
        == "/Volumes/custom/es"
    )


def test_genai_snapshot_dir_under_es_version() -> None:
    es_dir = Path("/Volumes/dev_sst_02/default/edvise_releases/es/abc123")
    assert genai_snapshot_dir(es_dir) == es_dir / GENAI_SNAPSHOT_DIRNAME


def test_genai_execute_layout_points_at_genai_mapping_bundle() -> None:
    layout = genai_execute_dab_bundle_layout()
    assert layout.pipeline_dir == "genai_mapping"
    assert layout.inference_yml_filename == "github_genai_mapping_execute.yml"
    assert (
        layout.inference_job_key == "edvise_genai_mapping_execute_pipeline"
    )


def test_edvise_layout_unchanged_for_es_inference_yml() -> None:
    layout = resolve_dab_bundle_layout("edvise")
    assert layout.inference_yml_filename == "github_es_inference.yml"
    assert layout.pipeline_dir == "es"


def test_materialize_es_and_genai_snapshots_to_expected_dirs(tmp_path: Path) -> None:
    es_version = "es_sha_aaa"
    genai_version = "genai_sha_bbb"
    release_base = tmp_path / "edvise_releases" / "es"
    es_release_dir = release_base / es_version
    genai_dir = genai_snapshot_dir(es_release_dir)

    es_layout = resolve_dab_bundle_layout("edvise")
    genai_layout = genai_execute_dab_bundle_layout()

    def fake_fetch(repo: str, sha: str, path: str, **kwargs: object) -> bytes:
        return f"# {sha}:{path}\n".encode()

    with patch(
        "edvise.runtime.versioned_inference.bundle.materialize.fetch_github_file",
        side_effect=fake_fetch,
    ):
        materialize_runtime_bundle_dir(
            es_release_dir,
            es_version,
            schema_type="edvise",
            git_ref=es_version,
            skip_snapshot_if_present=False,
        )
        materialize_runtime_bundle_dir(
            genai_dir,
            genai_version,
            layout=genai_layout,
            git_ref=genai_version,
            skip_snapshot_if_present=False,
        )

    es_inf = es_release_dir / es_layout.inference_yml_snapshot_rel
    genai_inf = genai_dir / genai_layout.inference_yml_snapshot_rel
    assert es_inf.is_file()
    assert genai_inf.is_file()
    assert b"es_sha_aaa" in es_inf.read_bytes()
    assert b"genai_sha_bbb" in genai_inf.read_bytes()
    assert GENAI_SNAPSHOT_DIRNAME in str(genai_inf)


def test_resolve_genai_pipeline_version_from_registry(tmp_path: Path) -> None:
    active = tmp_path / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "onboard_run_id": "run1",
                "institution_id": "demo",
                "pipeline_version": "deadbeef",
            }
        ),
        encoding="utf-8",
    )

    with patch(
        "edvise.runtime.versioned_inference.genai_registry.genai_active_root",
        return_value=active,
    ):
        assert (
            resolve_genai_pipeline_version_from_registry("dev_sst_02", "demo")
            == "deadbeef"
        )


def test_resolve_genai_pipeline_version_missing_registry(tmp_path: Path) -> None:
    active = tmp_path / "missing_active"
    active.mkdir(parents=True)
    with patch(
        "edvise.runtime.versioned_inference.genai_registry.genai_active_root",
        return_value=active,
    ):
        with pytest.raises(FileNotFoundError, match="genai_active_registry"):
            resolve_genai_pipeline_version_from_registry("dev_sst_02", "demo")


def test_silver_volume_root_shape() -> None:
    assert silver_volume_root("dev_sst_02", "acme") == Path(
        "/Volumes/dev_sst_02/acme_silver/silver_volume"
    )
    assert genai_active_root("dev_sst_02", "acme") == Path(
        "/Volumes/dev_sst_02/acme_silver/silver_volume/genai_mapping/active"
    )
