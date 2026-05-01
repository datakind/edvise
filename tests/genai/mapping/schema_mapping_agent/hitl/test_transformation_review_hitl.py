"""Tests for Step 2b transformation review HITL (``transformation.hitl.review``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import ReviewStatus
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.review import (
    TransformationReviewHITLFile,
    apply_transformation_review_resolutions,
    build_transformation_review_hitl_file_for_entity,
    check_transformation_review_hitl_gate,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    MapValuesStep,
    TransformationHITLOption,
    TransformationMap,
)


def _hitl_opts() -> list[TransformationHITLOption]:
    return [
        TransformationHITLOption(
            option_id="approve",
            label="Approve",
            description="Ok",
            resolution={"approved": True},
        ),
        TransformationHITLOption(
            option_id="correct",
            label="Correct",
            description="Fix",
            resolution=None,
        ),
        TransformationHITLOption(
            option_id="unmappable",
            label="Unmappable",
            description="Skip",
            resolution={"steps": [], "output_dtype": None},
        ),
    ]


def _reviewable_wrapper(*, institution_id: str = "test_u") -> dict:
    return {
        "institution_id": institution_id,
        "transformation_maps": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "RawEdviseStudentDataSchema",
                "plans": [
                    {
                        "target_field": "risk_flag",
                        "output_dtype": "string",
                        "confidence": 0.65,
                        "review_required": True,
                        "steps": [
                            {
                                "function_name": "cast_string",
                                "column": "raw_flag",
                            }
                        ],
                        "flagged_steps": [
                            {
                                "step_index": 0,
                                "function_name": "cast_string",
                                "reason": "low_confidence_utility_chain",
                                "context": {},
                            }
                        ],
                        "hitl_options": [o.model_dump(mode="json") for o in _hitl_opts()],
                    }
                ],
            },
            "course": {
                "entity_type": "course",
                "target_schema": "RawEdviseCourseDataSchema",
                "plans": [],
            },
        },
    }


def test_build_transformation_review_hitl_extracts_items():
    data = _reviewable_wrapper()
    env = build_transformation_review_hitl_file_for_entity(
        data,
        institution_id="test_u",
        entity_type="cohort",
        pipeline_version="0.0.1",
    )
    assert env.domain == "transformation_review"
    assert len(env.items) == 1
    assert env.items[0].target_field == "risk_flag"


def test_transformation_review_hitl_file_accepts_hitl_items_key():
    raw = {
        "institution_id": "x",
        "entity_type": "cohort",
        "domain": "transformation_review",
        "pipeline_version": "1",
        "reviewed": False,
        "hitl_items": [],
    }
    env = TransformationReviewHITLFile.model_validate(raw)
    assert env.items == []


def test_apply_transformation_review_approve_strips_metadata(tmp_path: Path):
    data = _reviewable_wrapper()
    cohort_path = tmp_path / "cohort.json"
    item = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    ).items[0]
    resolved = item.model_copy(update={"status": "approved"})
    file_model = TransformationReviewHITLFile(
        institution_id="test_u",
        entity_type="cohort",
        pipeline_version="1",
        items=[resolved],
    )
    cohort_path.write_text(json.dumps(file_model.model_dump(mode="json"), indent=2))

    out = apply_transformation_review_resolutions(
        data,
        cohort_review_path=cohort_path,
        course_review_path=None,
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["steps"][0]["function_name"] == "cast_string"
    assert "review_required" not in plan
    assert "flagged_steps" not in plan
    assert plan["confidence"] == pytest.approx(0.65)
    assert plan["review_status"] == ReviewStatus.corrected_by_hitl.value
    tm_dict = {
        **out["transformation_maps"]["cohort"],
        "institution_id": "test_u",
        "pipeline_version": "1",
    }
    TransformationMap.model_validate(tm_dict)


def test_apply_transformation_review_unmappable(tmp_path: Path):
    data = _reviewable_wrapper()
    cohort_path = tmp_path / "cohort.json"
    item = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    ).items[0]
    resolved = item.model_copy(update={"status": "unmappable"})
    cohort_path.write_text(
        json.dumps(
            TransformationReviewHITLFile(
                institution_id="test_u",
                entity_type="cohort",
                pipeline_version="1",
                items=[resolved],
            ).model_dump(mode="json"),
            indent=2,
        )
    )
    out = apply_transformation_review_resolutions(
        data, cohort_review_path=cohort_path, course_review_path=None
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["steps"] == []
    assert plan.get("output_dtype") is None
    assert plan["review_status"] == ReviewStatus.corrected_by_hitl.value


def test_apply_transformation_review_corrected_uses_item_steps(tmp_path: Path):
    data = _reviewable_wrapper()
    cohort_path = tmp_path / "cohort.json"
    base_item = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    ).items[0]
    new_steps = [
        MapValuesStep(
            function_name="map_values",
            column="raw_flag",
            mapping={"Y": "yes"},
        )
    ]
    resolved = base_item.model_copy(
        update={"status": "corrected", "steps": new_steps}
    )
    cohort_path.write_text(
        json.dumps(
            TransformationReviewHITLFile(
                institution_id="test_u",
                entity_type="cohort",
                pipeline_version="1",
                items=[resolved],
            ).model_dump(mode="json"),
            indent=2,
        )
    )
    out = apply_transformation_review_resolutions(
        data, cohort_review_path=cohort_path, course_review_path=None
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["steps"][0]["function_name"] == "map_values"
    assert "review_required" not in plan
    assert plan["confidence"] == pytest.approx(0.65)
    assert plan["review_status"] == ReviewStatus.corrected_by_hitl.value


def test_effective_status_from_choice_only():
    from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.review import (
        _effective_item_status,
    )

    item = build_transformation_review_hitl_file_for_entity(
        _reviewable_wrapper(),
        institution_id="test_u",
        entity_type="cohort",
        pipeline_version="1",
    ).items[0]
    item2 = item.model_copy(update={"status": "pending", "choice": "3"})
    assert _effective_item_status(item2) == "unmappable"


def test_check_gate_raises_when_pending(tmp_path: Path):
    from edvise.genai.mapping.shared.hitl.exceptions import HITLBlockingError

    data = _reviewable_wrapper()
    env = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    )
    p = tmp_path / "c.json"
    p.write_text(json.dumps(env.model_dump(mode="json"), indent=2))
    with pytest.raises(HITLBlockingError):
        check_transformation_review_hitl_gate(p)


def test_transformation_review_hitl_file_accepts_integer_choice():
    """Streamlit ``set_item_choice`` writes JSON numbers; model must accept them."""
    data = _reviewable_wrapper()
    item = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    ).items[0]
    item_int = item.model_copy(update={"choice": 1})
    raw = TransformationReviewHITLFile(
        institution_id="test_u",
        entity_type="cohort",
        pipeline_version="1",
        items=[item_int],
    ).model_dump(mode="json")
    env = TransformationReviewHITLFile.model_validate(raw)
    assert env.items[0].choice == 1


def test_check_gate_clear_when_choice_set(tmp_path: Path):
    data = _reviewable_wrapper()
    item = build_transformation_review_hitl_file_for_entity(
        data, institution_id="test_u", entity_type="cohort", pipeline_version="1"
    ).items[0]
    item = item.model_copy(update={"choice": "1"})
    p = tmp_path / "c.json"
    p.write_text(
        json.dumps(
            TransformationReviewHITLFile(
                institution_id="test_u",
                entity_type="cohort",
                pipeline_version="1",
                items=[item],
            ).model_dump(mode="json"),
            indent=2,
        )
    )
    check_transformation_review_hitl_gate(p)
