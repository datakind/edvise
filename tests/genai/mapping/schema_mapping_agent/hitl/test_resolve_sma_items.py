"""Tests for :func:`resolve_sma_items`."""

from __future__ import annotations

import json

from edvise.genai.mapping.schema_mapping_agent.hitl.resolver import (
    SMAHITLResolverError,
    resolve_sma_items,
)
from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    InstitutionSMAHITLItems,
    SMAFailureMode,
    SMAHITLItem,
    SMAHITLOption,
    SMAReentryDepth,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    EntityType,
    FieldMappingManifest,
    FieldMappingRecord,
    MappingManifestEnvelope,
    ReviewStatus,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.hitl.run_log import RunLog


def _fmr(**overrides):
    base = {
        "target_field": "learner_id",
        "source_column": "student_id",
        "source_table": "cohort",
        "confidence": 0.4,
        "row_selection": {"strategy": "any_row"},
    }
    base.update(overrides)
    return FieldMappingRecord.model_validate(base)


def test_resolve_sma_items_single_manifest(tmp_path):
    from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import (
        write_sma_hitl_artifact,
        write_sma_manifest_artifact,
    )

    cur = _fmr()
    alt = _fmr(source_column="sid_renamed", confidence=1.0)

    opt_a = SMAHITLOption(
        option_id="use_sid",
        label="Use sid",
        description="Rename source column.",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=alt,
    )
    opt_de = SMAHITLOption(
        option_id="direct_edit",
        label="Direct",
        description="Edit manually.",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
    )
    item = SMAHITLItem(
        item_id="u1_cohort_learner_id_low_confidence",
        institution_id="u1",
        entity_type="cohort",
        target_field="learner_id",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="Which column?",
        hitl_context=None,
        current_field_mapping=cur,
        validation_errors=[],
        options=[opt_a, opt_de],
        choice=1,
        reviewer_note="picked option 1",
    )
    hitl = InstitutionSMAHITLItems(institution_id="u1", entity_type="cohort", items=[item])

    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[cur],
    )

    hitl_path = write_sma_hitl_artifact(tmp_path, hitl, basename="sma_hitl_cohort.json")
    manifest_path = write_sma_manifest_artifact(tmp_path, fm, basename="manifest.json")
    run_log_path = tmp_path / "run_log.json"

    n = resolve_sma_items(
        hitl_path,
        manifest_path,
        resolved_by="test",
        run_log_path=run_log_path,
    )
    assert n == 1

    out_fm = read_pydantic_json(manifest_path, FieldMappingManifest)
    assert len(out_fm.mappings) == 1
    m0 = out_fm.mappings[0]
    assert m0.source_column == "sid_renamed"
    assert m0.review_status == ReviewStatus.corrected_by_hitl
    assert m0.reviewer_notes == "picked option 1"

    log = read_pydantic_json(run_log_path, RunLog)
    assert log.institution_id == "u1"
    assert len(log.events) == 1
    assert log.events[0].item_id == item.item_id


def test_resolve_sma_items_envelope(tmp_path):
    from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import (
        write_sma_hitl_artifact,
    )

    cur_c = _fmr()
    alt_c = _fmr(source_column="sid_renamed", confidence=1.0)
    opt_a = SMAHITLOption(
        option_id="use_sid",
        label="Use sid",
        description="Rename.",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=alt_c,
    )
    opt_de = SMAHITLOption(
        option_id="direct_edit",
        label="Direct",
        description="Edit.",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
    )
    item = SMAHITLItem(
        item_id="x",
        institution_id="u1",
        entity_type="cohort",
        target_field="learner_id",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="Q",
        current_field_mapping=cur_c,
        options=[opt_a, opt_de],
        choice=1,
    )
    hitl = InstitutionSMAHITLItems(institution_id="u1", entity_type="cohort", items=[item])
    hitl_path = write_sma_hitl_artifact(tmp_path, hitl)

    course_fm = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            _fmr(
                target_field="course_id",
                source_column="cid",
                source_table="course",
                confidence=1.0,
            )
        ],
    )
    cohort_fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[cur_c],
    )
    envelope = MappingManifestEnvelope(
        institution_id="u1",
        manifests={
            EntityType.cohort: cohort_fm,
            EntityType.course: course_fm,
        },
    )
    manifest_path = tmp_path / "mapping_manifest.json"
    manifest_path.write_text(envelope.model_dump_json(indent=2))

    resolve_sma_items(hitl_path, manifest_path, resolved_by=None, run_log_path=None)

    raw = json.loads(manifest_path.read_text())
    env2 = MappingManifestEnvelope.model_validate(raw)
    assert env2.manifests[EntityType.cohort].mappings[0].source_column == "sid_renamed"
    assert env2.manifests[EntityType.course].mappings[0].source_column == "cid"


def test_resolve_sma_items_institution_mismatch(tmp_path):
    from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import (
        write_sma_hitl_artifact,
    )

    cur = _fmr()
    opt_a = SMAHITLOption(
        option_id="a",
        label="a",
        description="a",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=cur,
    )
    opt_de = SMAHITLOption(
        option_id="direct_edit",
        label="d",
        description="d",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
    )
    item = SMAHITLItem(
        item_id="x",
        institution_id="u1",
        entity_type="cohort",
        target_field="learner_id",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="Q",
        current_field_mapping=cur,
        options=[opt_a, opt_de],
        choice=1,
    )
    hitl = InstitutionSMAHITLItems(institution_id="u1", entity_type="cohort", items=[item])
    hitl_path = write_sma_hitl_artifact(tmp_path, hitl)

    cohort_fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[cur],
    )
    envelope = MappingManifestEnvelope(
        institution_id="other",
        manifests={EntityType.cohort: cohort_fm},
    )
    mp = tmp_path / "m.json"
    mp.write_text(envelope.model_dump_json(indent=2))

    try:
        resolve_sma_items(hitl_path, mp)
    except SMAHITLResolverError as e:
        assert "institution_id" in str(e).lower()
    else:
        raise AssertionError("expected SMAHITLResolverError")
