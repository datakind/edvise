"""Tests for :mod:`edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.artifacts import (
    write_sma_manifest_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.override import (
    ManifestMappingOverrideRequest,
    ManifestOverrideError,
    load_overrides_json,
    override_manifest_mapping,
    override_manifest_mappings,
    unmapped_field_mapping_record,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    ReviewStatus,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.hitl.run_log import MappingOverrideLog


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


def test_override_manifest_mapping_local_wrapper(tmp_path: Path) -> None:
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="m.json")
    override_log_path = tmp_path / "mapping_override_log.json"

    override_manifest_mapping(
        manifest_path,
        "cohort",
        "learner_id",
        _fmr(source_column="student_id_v2"),
        override_log_path=override_log_path,
        overridden_by="ops",
        original_db_run_id="db-1",
        institution_id="u9",
        reviewer_notes="fixed",
    )

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    assert out.mappings[0].source_column == "student_id_v2"
    assert out.mappings[0].review_status == ReviewStatus.corrected_by_override

    log = read_pydantic_json(override_log_path, MappingOverrideLog)
    assert log.institution_id == "u9"
    assert len(log.events) == 1


def test_override_manifest_mappings_batch_local(tmp_path: Path) -> None:
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            _fmr(),
            _fmr(
                target_field="entry_term",
                source_column="term_code",
                source_table="cohort",
            ),
        ],
    )
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="m.json")
    override_log_path = tmp_path / "mapping_override_log.json"

    count = override_manifest_mappings(
        manifest_path,
        [
            ManifestMappingOverrideRequest(
                entity_type="cohort",
                target_field="learner_id",
                corrected=_fmr(source_column="student_id_v2"),
                reviewer_notes="fix id",
            ),
            ManifestMappingOverrideRequest(
                entity_type="cohort",
                target_field="entry_term",
                corrected=unmapped_field_mapping_record("entry_term"),
            ),
        ],
        override_log_path=override_log_path,
        overridden_by="ops",
        original_db_run_id="db-1",
        institution_id="u9",
    )
    assert count == 2

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    by_field = {m.target_field: m for m in out.mappings}
    assert by_field["learner_id"].source_column == "student_id_v2"
    assert by_field["entry_term"].source_column is None

    log = read_pydantic_json(override_log_path, MappingOverrideLog)
    assert len(log.events) == 2


def test_load_overrides_json(tmp_path: Path) -> None:
    p = tmp_path / "batch.json"
    p.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "entity_type": "cohort",
                        "target_field": "learner_id",
                        "correction": {
                            "target_field": "learner_id",
                            "source_column": "sid",
                            "source_table": "cohort",
                            "confidence": 1.0,
                            "row_selection": {"strategy": "any_row"},
                        },
                    },
                    {
                        "entity_type": "course",
                        "target_field": "course_id",
                        "unmap": True,
                    },
                ]
            }
        )
    )
    loaded = load_overrides_json(p)
    assert len(loaded) == 2
    assert loaded[0].corrected.source_column == "sid"
    assert loaded[1].corrected.source_column is None


def test_load_overrides_json_invalid_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text('"not-an-object"')
    with pytest.raises(ManifestOverrideError, match="JSON object or array"):
        load_overrides_json(p)
