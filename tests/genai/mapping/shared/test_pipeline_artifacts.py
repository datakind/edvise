"""Tests for versioned GenAI pipeline artifact paths and UC registry helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.genai.mapping.shared.pipeline_artifacts import (
    GenaiPipelineLayout,
    build_genai_pipeline_artifact_rows,
    discover_artifact_files,
    parse_uc_catalog_from_volume_path,
    resolve_pipeline_run_id,
    resolve_pipeline_version,
    versioned_genai_run_root,
    write_genai_pipeline_run_metadata,
)


def test_parse_uc_catalog_from_volume_path():
    assert parse_uc_catalog_from_volume_path("/Volumes/sst_dev/foo/bar") == "sst_dev"
    assert parse_uc_catalog_from_volume_path("/tmp/local") is None


def test_versioned_genai_run_root():
    root = versioned_genai_run_root(
        "/Volumes/cat/foo_bronze/bronze_volume",
        "run_abc",
    )
    assert root == Path(
        "/Volumes/cat/foo_bronze/bronze_volume/genai_pipeline/run_abc"
    )


def test_versioned_genai_run_root_rejects_empty_run_id():
    with pytest.raises(ValueError, match="pipeline_run_id"):
        versioned_genai_run_root("/Volumes/c/x", "")


def test_genai_pipeline_layout():
    layout = GenaiPipelineLayout.from_bronze("/tmp/bronze", "r1")
    assert layout.identity_hitl == Path("/tmp/bronze/genai_pipeline/r1/identity_hitl")


def test_resolve_pipeline_run_id_explicit_over_env(monkeypatch):
    monkeypatch.setenv("GENAI_PIPELINE_RUN_ID", "from_env")
    monkeypatch.delenv("DATABRICKS_JOB_RUN_ID", raising=False)
    assert resolve_pipeline_run_id("explicit") == "explicit"
    assert resolve_pipeline_run_id(None) == "from_env"
    monkeypatch.delenv("GENAI_PIPELINE_RUN_ID", raising=False)
    assert resolve_pipeline_run_id(None, create_if_missing=False) is None


def test_resolve_pipeline_run_id_prefers_databricks_job_env_over_manual(monkeypatch):
    monkeypatch.setenv("GENAI_PIPELINE_RUN_ID", "manual")
    monkeypatch.setenv("DATABRICKS_JOB_RUN_ID", "987654321")
    assert resolve_pipeline_run_id(None) == "987654321"


def test_resolve_pipeline_version_explicit_and_env(monkeypatch):
    assert resolve_pipeline_version("1.2.3") == "1.2.3"
    monkeypatch.setenv("GENAI_GIT_TAG", "v0.1.0")
    assert resolve_pipeline_version(None) == "v0.1.0"
    monkeypatch.delenv("GENAI_GIT_TAG", raising=False)
    monkeypatch.delenv("GIT_TAG", raising=False)
    monkeypatch.delenv("GENAI_PIPELINE_VERSION", raising=False)
    v = resolve_pipeline_version(None)
    assert v  # installed edvise version


def test_discover_artifact_files(tmp_path: Path):
    inst = "demo_col"
    run = tmp_path / "genai_pipeline" / "r1"
    ih = run / "identity_hitl"
    ih.mkdir(parents=True)
    sm = run / "schema_mapping"
    sm.mkdir(parents=True)
    enc = run / "enriched_schema_contracts"
    enc.mkdir(parents=True)
    (ih / "identity_grain_output.json").write_text("{}")
    (sm / f"{inst}_mapping_manifest.json").write_text("{}")
    (sm / f"{inst}_transformation_map.json").write_text("{}")
    (sm / "sma_hitl_cohort.json").write_text("{}")
    (enc / f"{inst}_schema_contract.json").write_text("{}")
    (run / "run_log.json").write_text("{}")
    write_genai_pipeline_run_metadata(
        run,
        institution_id=inst,
        pipeline_run_id="r1",
        pipeline_version="0.2.0",
    )

    found = discover_artifact_files(run, inst)
    assert set(found.keys()) == {
        "identity_grain_output",
        "mapping_manifest",
        "transformation_map",
        "sma_hitl",
        "enriched_schema_contract",
        "run_log",
        "pipeline_run_metadata",
    }


def test_build_genai_pipeline_artifact_rows(tmp_path: Path):
    p = tmp_path / "a.json"
    p.write_text('{"x": 1}')
    rows = build_genai_pipeline_artifact_rows(
        institution_id="x",
        pipeline_run_id="r",
        pipeline_version="0.2.0",
        bronze_volumes_path="/Volumes/c/p",
        artifact_paths={"mapping_manifest": p},
        uc_catalog="c",
    )
    assert len(rows) == 1
    assert rows[0]["pipeline_version"] == "0.2.0"
    assert rows[0]["artifact_kind"] == "mapping_manifest"
    assert rows[0]["uc_catalog"] == "c"
    assert rows[0]["content_sha256"]
    assert "a.json" in rows[0]["relative_path"]


def test_write_genai_pipeline_run_metadata(tmp_path: Path):
    p = write_genai_pipeline_run_metadata(
        tmp_path,
        institution_id="demo_col",
        pipeline_run_id="99",
        pipeline_version="0.2.0",
    )
    assert p.name == "genai_pipeline_run.json"
    text = p.read_text(encoding="utf-8")
    assert '"pipeline_version": "0.2.0"' in text
    assert '"pipeline_run_id": "99"' in text
