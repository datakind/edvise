"""Tests for :func:`apply_2a_manifest_repair`."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.artifacts import (
    write_sma_manifest_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.resolver import (
    SMAHITLResolverError,
    apply_2a_manifest_repair,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    EntityType,
    FieldMappingManifest,
    FieldMappingRecord,
    MappingManifestEnvelope,
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


def test_apply_2a_manifest_repair_preserves_confidence_and_rationale(
    tmp_path: Path,
) -> None:
    tmp = tmp_path
    original = _fmr()
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[original],
    )
    manifest_path = write_sma_manifest_artifact(tmp, fm, basename="m.json")
    repair_log_path = tmp / "repair_log.json"

    corrected = _fmr(
        source_column="student_id_v2",
        confidence=0.99,
        rationale="should not appear",
    )

    apply_2a_manifest_repair(
        manifest_path,
        "cohort",
        "learner_id",
        corrected,
        repair_log_path=repair_log_path,
        repaired_by="ops",
        original_db_run_id="db-42",
        original_task_run_id="t-7",
        reviewer_notes="fixed column",
        institution_id="u9",
    )

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    m0 = out.mappings[0]
    assert m0.source_column == "student_id_v2"
    assert m0.confidence == 0.4
    assert m0.rationale == "original rationale"
    assert m0.review_status == ReviewStatus.corrected_by_repair
    assert m0.reviewer_notes == "fixed column"

    log = read_pydantic_json(repair_log_path, RepairLog)
    assert log.institution_id == "u9"
    assert len(log.events) == 1
    e0 = log.events[0]
    assert e0.original_value["confidence"] == 0.4
    assert e0.corrected_value["confidence"] == 0.4
    assert e0.corrected_value["source_column"] == "student_id_v2"


def test_apply_2a_manifest_repair_envelope(tmp_path: Path) -> None:
    tmp = tmp_path
    original = _fmr()
    cohort_fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[original],
    )
    course_fm = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            _fmr(
                target_field="course_id",
                source_column="c1",
                source_table="course",
                confidence=1.0,
                rationale="course r",
            )
        ],
    )
    envelope = MappingManifestEnvelope(
        institution_id="u1",
        manifests={EntityType.cohort: cohort_fm, EntityType.course: course_fm},
    )
    manifest_path = tmp / "env.json"
    manifest_path.write_text(envelope.model_dump_json(indent=2))
    repair_log_path = tmp / "repair_log.json"

    corrected = _fmr(source_column="s_fixed")
    apply_2a_manifest_repair(
        manifest_path,
        "cohort",
        "learner_id",
        corrected,
        repair_log_path=repair_log_path,
        repaired_by="r",
        original_db_run_id="d",
        reviewer_notes=None,
    )

    env2 = MappingManifestEnvelope.model_validate_json(manifest_path.read_text())
    assert env2.manifests[EntityType.cohort].mappings[0].source_column == "s_fixed"
    assert env2.manifests[EntityType.course].mappings[0].source_column == "c1"

    log = read_pydantic_json(repair_log_path, RepairLog)
    assert log.institution_id == "u1"


def test_apply_2a_manifest_repair_requires_institution_for_standalone(
    tmp_path: Path,
) -> None:
    tmp = tmp_path
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    path = write_sma_manifest_artifact(tmp, fm)
    with pytest.raises(SMAHITLResolverError, match="institution_id"):
        apply_2a_manifest_repair(
            path,
            "cohort",
            "learner_id",
            _fmr(source_column="x"),
            repair_log_path=tmp / "r.json",
            repaired_by="x",
            original_db_run_id="d",
        )


def test_apply_2a_manifest_repair_wrong_target_field_raises(
    tmp_path: Path,
) -> None:
    tmp = tmp_path
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )
    path = write_sma_manifest_artifact(tmp, fm)
    bad = _fmr(target_field="other")
    with pytest.raises(SMAHITLResolverError, match="target_field"):
        apply_2a_manifest_repair(
            path,
            "cohort",
            "learner_id",
            bad,
            repair_log_path=tmp / "r.json",
            repaired_by="x",
            original_db_run_id="d",
            institution_id="u1",
        )
