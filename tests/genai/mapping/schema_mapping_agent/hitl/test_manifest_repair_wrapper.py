"""Tests for :mod:`edvise.genai.mapping.schema_mapping_agent.manifest.hitl.repair`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.artifacts import (
    write_sma_manifest_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.repair import (
    ManifestRepairError,
    load_correction_json,
    repair_manifest_mapping,
    repair_manifest_mapping_at_path,
    repair_manifest_mapping_on_volume,
    unmapped_field_mapping_record,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    ReviewStatus,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.hitl.run_log import RepairLog


def _fmr(**overrides: object) -> FieldMappingRecord:
    base = {
        "target_field": "learner_id",
        "source_column": "student_id",
        "source_table": "cohort",
        "confidence": 0.4,
        "rationale": "original rationale",
        "row_selection": {"strategy": "any_row"},
    }
    base.update(overrides)
    return FieldMappingRecord.model_validate(base)


def test_unmapped_field_mapping_record() -> None:
    rec = unmapped_field_mapping_record("learner_id")
    assert rec.target_field == "learner_id"
    assert rec.source_column is None
    assert rec.source_table is None
    assert rec.join is None
    assert rec.row_selection is None


def test_load_correction_json(tmp_path: Path) -> None:
    p = tmp_path / "corr.json"
    p.write_text(
        json.dumps(
            {
                "target_field": "learner_id",
                "source_column": "sid",
                "source_table": "cohort",
                "confidence": 0.5,
                "row_selection": {"strategy": "any_row"},
            }
        )
    )
    rec = load_correction_json(p)
    assert rec.source_column == "sid"


def test_load_correction_json_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('"not-an-object"')
    with pytest.raises(ManifestRepairError, match="JSON object"):
        load_correction_json(p)


def test_repair_manifest_mapping_local_wrapper(tmp_path: Path) -> None:
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="m.json")
    repair_log_path = tmp_path / "repair_log.json"

    repair_manifest_mapping(
        manifest_path,
        "cohort",
        "learner_id",
        _fmr(source_column="student_id_v2"),
        repair_log_path=repair_log_path,
        repaired_by="ops",
        original_db_run_id="db-1",
        institution_id="u9",
        reviewer_notes="fixed",
    )

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    assert out.mappings[0].source_column == "student_id_v2"
    assert out.mappings[0].review_status == ReviewStatus.corrected_by_repair

    log = read_pydantic_json(repair_log_path, RepairLog)
    assert log.institution_id == "u9"
    assert len(log.events) == 1


def test_repair_manifest_mapping_at_path_routes_local(tmp_path: Path) -> None:
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="m.json")
    repair_log_path = tmp_path / "repair_log.json"

    repair_manifest_mapping_at_path(
        manifest_path,
        "cohort",
        "learner_id",
        unmapped_field_mapping_record("learner_id"),
        repair_log_path=repair_log_path,
        repaired_by="ops",
        original_db_run_id="db-1",
        institution_id="u9",
    )

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    assert out.mappings[0].source_column is None
    assert out.mappings[0].source_table is None


def test_repair_manifest_mapping_on_volume_round_trip(tmp_path: Path) -> None:
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="m.json")
    repair_log_path = tmp_path / "repair_log.json"

    manifest_uc = "/Volumes/cat/sch/vol/manifest_map.json"
    repair_uc = "/Volumes/cat/sch/vol/repair_log.json"
    store: dict[str, str] = {}

    def _read(path: str) -> str:
        if path not in store:
            raise FileNotFoundError(path)
        return store[path]

    def _write(path: str, text: str) -> None:
        store[path] = text

    store[manifest_uc] = manifest_path.read_text(encoding="utf-8")

    with (
        patch(
            "edvise.genai.mapping.schema_mapping_agent.manifest.hitl.repair.read_unity_file_text",
            side_effect=_read,
        ),
        patch(
            "edvise.genai.mapping.schema_mapping_agent.manifest.hitl.repair.write_unity_file_text",
            side_effect=_write,
        ),
    ):
        repair_manifest_mapping_on_volume(
            manifest_uc,
            "cohort",
            "learner_id",
            _fmr(source_column="fixed"),
            repair_log_uc_path=repair_uc,
            repaired_by="ops",
            original_db_run_id="db-1",
            institution_id="u9",
        )

    updated = FieldMappingManifest.model_validate_json(store[manifest_uc])
    assert updated.mappings[0].source_column == "fixed"
    assert repair_uc in store
    log = RepairLog.model_validate_json(store[repair_uc])
    assert log.institution_id == "u9"
