"""Batched Step 2a prompt: token audit vs two-pass (cohort + course) and content checks."""

from __future__ import annotations

from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    audit_step2a_batched_prompt,
    audit_step2a_prompt,
    build_step2a_batched_prompt,
)


def _minimal_institution_schema_contract() -> dict:
    return {
        "school_id": "test_cc",
        "school_name": "Test College",
        "datasets": {
            "student": {
                "normalized_columns": {"student_id": "student_id"},
                "dtypes": {"student_id": "int64"},
                "non_null_columns": ["student_id"],
                "unique_keys": ["student_id"],
                "null_tokens": [],
                "boolean_map": {},
                "training": {
                    "file_path": "x.csv",
                    "num_rows": 1,
                    "num_columns": 1,
                    "column_normalization": {},
                    "column_details": [
                        {
                            "original_name": "student_id",
                            "normalized_name": "student_id",
                            "null_count": 0,
                            "null_percentage": 0.0,
                            "unique_count": 1,
                            "sample_values": ["1"],
                        }
                    ],
                },
            }
        },
    }


def _minimal_reference_manifest() -> dict:
    return {
        "pipeline_version": "0.2.0",
        "institution_id": "ref",
        "manifests": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "RawEdviseStudentDataSchema",
                "mappings": [],
            },
            "course": {
                "entity_type": "course",
                "target_schema": "RawEdviseCourseDataSchema",
                "mappings": [],
            },
        },
    }


def test_step2a_batched_token_total_below_two_entity_passes():
    """Shared blocks are duplicated across cohort_pass + course_pass; batched sends them once."""
    kwargs = dict(
        institution_id="test_cc",
        institution_name="Test College",
        output_path="s3://bucket/manifest.json",
        institution_schema_contract=_minimal_institution_schema_contract(),
        reference_manifests=[_minimal_reference_manifest()],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
        log=False,
    )
    cohort_t = audit_step2a_prompt(**kwargs, variant="cohort_pass")[
        "total_estimated_tokens"
    ]
    course_t = audit_step2a_prompt(**kwargs, variant="course_pass")[
        "total_estimated_tokens"
    ]
    batched = audit_step2a_batched_prompt(**kwargs)["total_estimated_tokens"]
    single_t = audit_step2a_prompt(**kwargs, variant="single")["total_estimated_tokens"]

    assert batched < cohort_t + course_t
    # Batched aligns with single-pass shape (both targets + shared sections once); allow small preamble delta.
    assert abs(batched - single_t) < 500


def test_build_step2a_batched_prompt_includes_envelope_and_part_labels():
    text = build_step2a_batched_prompt(
        institution_id="test_cc",
        institution_name="Test College",
        output_path="s3://bucket/manifest.json",
        institution_schema_contract=_minimal_institution_schema_contract(),
        reference_manifests=[_minimal_reference_manifest()],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
    )
    assert "MappingManifestEnvelope" in text
    assert "PART 1: Cohort (RawEdviseStudentDataSchema)" in text
    assert "PART 2: Course (RawEdviseCourseDataSchema)" in text
    assert "omit release/institution envelope fields" in text


def test_audit_step2a_batched_prompt_section_breakdown():
    out = audit_step2a_batched_prompt(
        institution_id="test_cc",
        institution_name="Test College",
        output_path="s3://bucket/manifest.json",
        institution_schema_contract=_minimal_institution_schema_contract(),
        reference_manifests=[_minimal_reference_manifest()],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
        log=False,
    )
    assert out["builder"] == "schema_mapping_agent.step2a.batched"
    assert set(out["sections"].keys()) == {
        "preamble",
        "reference_manifests",
        "schema_contract",
        "target_schema_cohort",
        "target_schema_course",
        "manifest_schema_reference",
        "rules",
    }
    assert out["total_estimated_tokens"] == sum(out["sections"].values())
