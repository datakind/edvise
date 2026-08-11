"""Tests for :mod:`edvise.genai.mapping.shared.reference_pin` and reference volume paths."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.shared import reference_pin as rp
from edvise.genai.mapping.shared.volume_paths import (
    genai_mapping_schema_volume_root,
    genai_reference_current_root,
    genai_reference_root,
)


def test_genai_reference_volume_paths() -> None:
    assert (
        genai_mapping_schema_volume_root(catalog="dev_sst_02")
        == "/Volumes/dev_sst_02/genai_mapping/references"
    )
    assert (
        genai_reference_root("ref_cc_01", catalog="dev_sst_02")
        == "/Volumes/dev_sst_02/genai_mapping/references/ref_cc_01"
    )
    assert (
        genai_reference_current_root("ref_cc_01", catalog="dev_sst_02")
        == "/Volumes/dev_sst_02/genai_mapping/references/ref_cc_01/current"
    )


def test_compute_reference_content_hash_stable(tmp_path: Path) -> None:
    a = tmp_path / "manifest_map.json"
    b = tmp_path / "transformation_map.json"
    a.write_text('{"m": 1}\n', encoding="utf-8")
    b.write_text('{"t": 1}\n', encoding="utf-8")
    h1 = rp.compute_reference_content_hash(
        {"transformation_map.json": b, "manifest_map.json": a}
    )
    h2 = rp.compute_reference_content_hash(
        {"manifest_map.json": a, "transformation_map.json": b}
    )
    assert h1 == h2
    assert h1.startswith("sha256:")
    b.write_text('{"t": 2}\n', encoding="utf-8")
    h3 = rp.compute_reference_content_hash(
        {"manifest_map.json": a, "transformation_map.json": b}
    )
    assert h3 != h1


def _seed_active(
    tmp_path: Path,
    *,
    catalog: str,
    institution_id: str,
    with_registry: bool = True,
) -> Path:
    """Create a silver active/ tree under tmp_path (volume roots patched in tests)."""
    active = (
        tmp_path
        / "Volumes"
        / catalog
        / f"{institution_id}_silver"
        / "silver_volume"
        / "genai_mapping"
        / "active"
    )
    active.mkdir(parents=True)
    (active / "manifest_map.json").write_text('{"m": 1}', encoding="utf-8")
    (active / "transformation_map.json").write_text('{"t": 1}', encoding="utf-8")
    (active / "enriched_schema_contract.json").write_text('{"c": 1}', encoding="utf-8")
    if with_registry:
        (active / "genai_active_registry.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "onboard_run_id": "school_20260101_1",
                    "institution_id": institution_id,
                    "pipeline_version": "abc123",
                    "promoted_at": "2026-01-01T00:00:00Z",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return active


def _patch_volume_roots(monkeypatch: pytest.MonkeyPatch, volumes: Path) -> None:
    monkeypatch.setattr(
        rp,
        "silver_genai_mapping_root",
        lambda institution_id, *, catalog: str(
            volumes
            / catalog
            / f"{institution_id}_silver"
            / "silver_volume"
            / "genai_mapping"
        ),
    )
    monkeypatch.setattr(
        rp,
        "genai_reference_current_root",
        lambda reference_id, *, catalog: str(
            volumes
            / catalog
            / "genai_mapping"
            / "references"
            / reference_id
            / "current"
        ),
    )


def test_pin_reference_snapshot_from_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_cat"
    ref = "demo_col"
    volumes = tmp_path / "Volumes"
    _patch_volume_roots(monkeypatch, volumes)
    _seed_active(tmp_path, catalog=catalog, institution_id=ref)

    pin = rp.pin_reference_snapshot(catalog=catalog, reference_id=ref)

    current = Path(rp.genai_reference_current_root(ref, catalog=catalog))
    assert (current / "manifest_map.json").read_text(encoding="utf-8") == '{"m": 1}'
    assert (current / "transformation_map.json").read_text(
        encoding="utf-8"
    ) == '{"t": 1}'
    assert (current / "enriched_schema_contract.json").is_file()
    assert pin["content_hash"].startswith("sha256:")
    assert pin["reference_id"] == ref
    assert pin["source_institution_id"] == ref
    assert pin["source_onboard_run_id"] == "school_20260101_1"
    assert pin["pipeline_version"] == "abc123"
    assert "pinned_by" not in pin
    assert "archetype" not in pin
    assert rp.verify_reference_pin_hash(current) == pin["content_hash"]

    hist_dirs = [
        p for p in current.parent.iterdir() if p.is_dir() and p.name.startswith("v")
    ]
    assert len(hist_dirs) == 1
    assert (hist_dirs[0] / rp.GENAI_REFERENCE_PIN_BASENAME).is_file()


def test_verify_reference_pin_hash_detects_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = "dev_cat"
    ref = "demo_col"
    volumes = tmp_path / "Volumes"
    _patch_volume_roots(monkeypatch, volumes)
    _seed_active(tmp_path, catalog=catalog, institution_id=ref)
    rp.pin_reference_snapshot(
        catalog=catalog, reference_id=ref, write_history_copy=False
    )
    current = Path(rp.genai_reference_current_root(ref, catalog=catalog))
    (current / "manifest_map.json").write_text('{"m": "tampered"}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        rp.verify_reference_pin_hash(current)


def test_upsert_reference_pin_row_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeSpark:
        def __init__(self) -> None:
            self.statements: list[str] = []

        def sql(self, q: str) -> None:
            self.statements.append(q)

    fake = _FakeSpark()
    monkeypatch.setattr(
        "edvise.genai.mapping.state.table_setup.create_state_tables",
        lambda catalog, spark=None: None,
    )
    pin = {
        "reference_id": "school_a",
        "content_hash": "sha256:abc",
        "pinned_at": "2026-06-22T12:00:00Z",
        "source_institution_id": "school_a",
        "source_onboard_run_id": "run_1",
        "pipeline_version": "v1",
        "artifacts": ["manifest_map.json", "transformation_map.json"],
    }
    rp.upsert_reference_pin_row("my_cat", pin, spark=fake)
    assert any("UPDATE" in s and "superseded" in s for s in fake.statements)
    insert = next(s for s in fake.statements if "INSERT INTO" in s)
    assert "reference_pins" in insert
    assert "school_a" in insert
    assert "sha256:abc" in insert
    assert "active" in insert


def test_pull_reference_snapshot_copies_bytes_not_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_cat = "dev_sst_02"
    dest_cat = "staging_sst_01"
    ref = "demo_col"
    volumes = tmp_path / "Volumes"
    _patch_volume_roots(monkeypatch, volumes)

    # Publish on staging from active/
    _seed_active(tmp_path, catalog=src_cat, institution_id=ref)
    published = rp.pin_reference_snapshot(
        catalog=src_cat, reference_id=ref, write_history_copy=False
    )

    # Divergent active on dest must not affect pull
    dest_active = (
        volumes
        / dest_cat
        / f"{ref}_silver"
        / "silver_volume"
        / "genai_mapping"
        / "active"
    )
    dest_active.mkdir(parents=True)
    (dest_active / "manifest_map.json").write_text('{"m": "WRONG"}', encoding="utf-8")
    (dest_active / "transformation_map.json").write_text(
        '{"t": "WRONG"}', encoding="utf-8"
    )

    pulled = rp.pull_reference_snapshot(
        catalog=dest_cat,
        reference_id=ref,
        source_catalog=src_cat,
        write_history_copy=False,
    )

    dest_current = Path(rp.genai_reference_current_root(ref, catalog=dest_cat))
    assert (dest_current / "manifest_map.json").read_text(
        encoding="utf-8"
    ) == '{"m": 1}'
    assert pulled["content_hash"] == published["content_hash"]
    assert pulled["pulled_from_catalog"] == src_cat
    assert pulled["uc_catalog"] == dest_cat
    assert pulled["source_onboard_run_id"] == "school_20260101_1"
    assert rp.verify_reference_pin_hash(dest_current) == published["content_hash"]


def test_pull_rejects_same_catalog() -> None:
    with pytest.raises(ValueError, match="source_catalog != catalog"):
        rp.pull_reference_snapshot(
            catalog="dev_sst_02",
            reference_id="x",
            source_catalog="dev_sst_02",
        )
