"""Tests for :mod:`edvise.genai.mapping.shared.active_promotion`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from edvise.genai.mapping.shared.active_promotion import promote_genai_mapping_to_active


@dataclass
class _FakeSMAPaths:
    active_root: Path
    active_enriched_schema_contract: Path
    active_manifest_map: Path
    active_transformation_map: Path
    active_transform_hooks: Path
    ia_enriched_schema_contract: Path
    manifest_map: Path
    transformation_map: Path
    transform_hooks: Path


def test_promote_genai_mapping_to_active_copies_required_and_optional(tmp_path: Path) -> None:
    genai = tmp_path / "genai"
    run_id = "school_20260101"
    ia = genai / "runs" / run_id / "identity_agent"
    sma = genai / "runs" / run_id / "schema_mapping_agent"
    active = genai / "active"
    ia.mkdir(parents=True)
    sma.mkdir(parents=True)

    (ia / "enriched_schema_contract.json").write_text('{"ia": true}')
    (sma / "manifest_map.json").write_text('{"m": 1}')
    (sma / "transformation_map.json").write_text('{"t": 1}')
    (sma / "transform_hooks.py").write_text("# hooks")
    (ia / "identity_grain_output.json").write_text("{}")
    (ia / "identity_term_output.json").write_text("{}")

    paths = _FakeSMAPaths(
        active_root=active,
        active_enriched_schema_contract=active / "enriched_schema_contract.json",
        active_manifest_map=active / "manifest_map.json",
        active_transformation_map=active / "transformation_map.json",
        active_transform_hooks=active / "transform_hooks.py",
        ia_enriched_schema_contract=ia / "enriched_schema_contract.json",
        manifest_map=sma / "manifest_map.json",
        transformation_map=sma / "transformation_map.json",
        transform_hooks=sma / "transform_hooks.py",
    )

    promote_genai_mapping_to_active(paths)

    assert paths.active_enriched_schema_contract.read_text() == '{"ia": true}'
    assert paths.active_manifest_map.read_text() == '{"m": 1}'
    assert paths.active_transformation_map.read_text() == '{"t": 1}'
    assert paths.active_transform_hooks.read_text() == "# hooks"
    assert (active / "grain_output.json").read_text() == "{}"
    assert (active / "term_output.json").read_text() == "{}"


def test_promote_genai_mapping_to_active_missing_required_raises(tmp_path: Path) -> None:
    genai = tmp_path / "genai"
    run_id = "r2"
    ia = genai / "runs" / run_id / "identity_agent"
    sma = genai / "runs" / run_id / "schema_mapping_agent"
    active = genai / "active"
    ia.mkdir(parents=True)
    sma.mkdir(parents=True)
    (ia / "enriched_schema_contract.json").write_text("{}")
    (sma / "manifest_map.json").write_text("{}")
    # missing transformation_map.json

    paths = _FakeSMAPaths(
        active_root=active,
        active_enriched_schema_contract=active / "enriched_schema_contract.json",
        active_manifest_map=active / "manifest_map.json",
        active_transformation_map=active / "transformation_map.json",
        active_transform_hooks=active / "transform_hooks.py",
        ia_enriched_schema_contract=ia / "enriched_schema_contract.json",
        manifest_map=sma / "manifest_map.json",
        transformation_map=sma / "transformation_map.json",
        transform_hooks=sma / "transform_hooks.py",
    )

    with pytest.raises(FileNotFoundError, match="transformation_map"):
        promote_genai_mapping_to_active(paths)
