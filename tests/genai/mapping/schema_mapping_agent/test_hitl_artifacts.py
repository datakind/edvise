"""SMA HITL on-disk artifact helpers."""

from __future__ import annotations

import json

from edvise.genai.mapping.schema_mapping_agent.hitl.artifacts import (
    SMA_HITL_BASENAME,
    SMA_MANIFEST_OUTPUT_BASENAME,
    load_sma_hitl,
    load_sma_manifest_output,
    unique_sma_hitl_items_by_item_id,
    write_sma_hitl_artifact,
    write_sma_manifest_artifact,
    write_sma_hitl_and_manifest_artifacts,
)
from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    InstitutionSMAHITLItems,
    SMAFailureMode,
    SMAHITLItem,
    SMAHITLOption,
    SMAReentryDepth,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
)


def _fmr(**overrides) -> FieldMappingRecord:
    base = {
        "target_field": "learner_id",
        "source_column": "student_id",
        "source_table": "cohort",
        "confidence": 1.0,
        "row_selection": {"strategy": "any_row"},
    }
    base.update(overrides)
    return FieldMappingRecord.model_validate(base)


def _minimal_manifest() -> FieldMappingManifest:
    return FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[_fmr()],
    )


def test_write_and_load_sma_hitl_roundtrip(tmp_path):
    env = InstitutionSMAHITLItems(
        institution_id="u1",
        entity_type="cohort",
        items=[],
    )
    p = write_sma_hitl_artifact(tmp_path, env)
    assert p.name == SMA_HITL_BASENAME
    loaded = load_sma_hitl(p)
    assert loaded.institution_id == "u1"
    assert loaded.items == []


def test_write_and_load_sma_manifest_roundtrip(tmp_path):
    m = _minimal_manifest()
    p = write_sma_manifest_artifact(tmp_path, m)
    assert p.name == SMA_MANIFEST_OUTPUT_BASENAME
    loaded = load_sma_manifest_output(p)
    assert loaded.entity_type == "cohort"
    assert len(loaded.mappings) == 1


def test_write_sma_hitl_and_manifest_artifacts(tmp_path):
    env = InstitutionSMAHITLItems(institution_id="u1", entity_type="course", items=[])
    m = FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            _fmr(
                target_field="course_id",
                source_column="cid",
                source_table="course",
            )
        ],
    )
    hp, mp = write_sma_hitl_and_manifest_artifacts(tmp_path, hitl=env, manifest=m)
    assert hp.exists() and mp.exists()
    assert load_sma_hitl(hp).entity_type == "course"
    assert load_sma_manifest_output(mp).target_schema == "RawEdviseCourseDataSchema"


def test_load_sma_hitl_missing_file(tmp_path):
    missing = tmp_path / "nope.json"
    try:
        load_sma_hitl(missing)
    except FileNotFoundError as e:
        assert "SMA HITL file not found" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_unique_sma_hitl_items_by_item_id():
    cur = _fmr()
    opt_terminal = SMAHITLOption(
        option_id="a",
        label="a",
        description="a",
        reentry=SMAReentryDepth.TERMINAL,
        field_mapping=cur,
    )
    opt_de = SMAHITLOption(
        option_id="direct_edit",
        label="de",
        description="de",
        reentry=SMAReentryDepth.DIRECT_EDIT,
        field_mapping=None,
    )
    i1 = SMAHITLItem(
        item_id="x",
        institution_id="u",
        entity_type="cohort",
        target_field="learner_id",
        failure_mode=SMAFailureMode.LOW_CONFIDENCE,
        hitl_question="q?",
        current_field_mapping=cur,
        options=[opt_terminal, opt_de],
    )
    i2 = SMAHITLItem.model_validate(json.loads(i1.model_dump_json()))
    i2.item_id = "x"
    out = unique_sma_hitl_items_by_item_id([i1, i2])
    assert len(out) == 1 and out[0].item_id == "x"


def test_manifest_lazy_import_from_manifest_package():
    """``manifest.hitl`` still resolves to SMA HITL (backward-compatible lazy attr)."""
    from edvise.genai.mapping.schema_mapping_agent.manifest import hitl as sma_hitl

    assert sma_hitl.SMA_HITL_BASENAME == "sma_hitl.json"
    assert sma_hitl.check_sma_hitl_gate is not None
