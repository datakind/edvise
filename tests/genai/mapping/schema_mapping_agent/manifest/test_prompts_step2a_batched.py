"""Batched Step 2a prompt: token audit vs two-pass (cohort + course) and content checks."""

from __future__ import annotations

from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    audit_step2a_batched_prompt,
    audit_step2a_prompt,
    build_refinement_pass1_system_prompt,
    build_refinement_pass2_system_prompt,
    build_step2a_batched_prompt,
    build_step2a_prompt_cohort_pass,
    build_step2a_prompt_course_pass,
    extract_schema_descriptor,
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
    assert "institution_id=test_cc" in text


def test_audit_step2a_batched_prompt_section_breakdown():
    out = audit_step2a_batched_prompt(
        institution_id="test_cc",
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


def _shared_prompt_kwargs() -> dict:
    return dict(
        institution_id="test_cc",
        output_path="s3://bucket/manifest.json",
        institution_schema_contract=_minimal_institution_schema_contract(),
        reference_manifests=[_minimal_reference_manifest()],
    )


def test_cohort_pass_includes_type_vs_major_rules_not_course_degree_rules():
    text = build_step2a_prompt_cohort_pass(
        **_shared_prompt_kwargs(),
        cohort_schema_class=RawEdviseStudentDataSchema,
    )
    assert "Credential TYPE vs major / plan" in text
    assert "intended_program_type" in text
    assert "declared_major_at_entry" in text
    assert "COURSE term_degree AND term_declared_major" not in text


def test_course_pass_includes_term_degree_vs_major_rules_not_cohort_program_rules():
    text = build_step2a_prompt_course_pass(
        **_shared_prompt_kwargs(),
        course_schema_class=RawEdviseCourseDataSchema,
    )
    assert "COURSE term_degree AND term_declared_major" in text
    assert "COHORT intended_program_type AND declared_major_at_entry" not in text


def test_schema_descriptor_notes_separate_type_from_major():
    student = extract_schema_descriptor(RawEdviseStudentDataSchema)
    course = extract_schema_descriptor(RawEdviseCourseDataSchema)
    ipt = student["fields"]["intended_program_type"]["semantic_note"]
    major = student["fields"]["declared_major_at_entry"]["semantic_note"]
    term_deg = course["fields"]["term_degree"]["semantic_note"]
    term_maj = course["fields"]["term_declared_major"]["semantic_note"]
    assert "credential / program TYPE" in ipt
    assert "NOT an academic plan" in ipt
    assert "NOT a degree/credential type code" in major
    assert "degree / credential LEVEL or TYPE" in term_deg
    assert "NOT a degree/credential type code" in term_maj


def test_refinement_prompts_include_cohort_and_course_type_vs_major_rules():
    pass1 = build_refinement_pass1_system_prompt()
    pass2 = build_refinement_pass2_system_prompt()
    assert "Column kind:" in pass1
    assert "term_degree" in pass1 and "term_declared_major" in pass1
    assert "prefer degree/credential-type columns" in pass2
    assert "Course term_degree vs term_declared_major" in pass2
