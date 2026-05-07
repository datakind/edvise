"""Step 2b hook_required HITL envelopes and resolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook import (
    InstitutionSMATransformationHookHITLItems,
    apply_transformation_hook_hitl_resolutions,
    build_transformation_hook_hitl_envelope_for_entity,
    check_transformation_hook_hitl_gate,
    write_transformation_hook_hitl_envelope,
)


def _sample_wrapper(*, hook_on_x: bool = True) -> dict:
    plans = [
        {
            "target_field": "x",
            "hook_required": hook_on_x,
            "reviewer_notes": "needs hook",
            "steps": [
                {
                    "function_name": "cast_string",
                    "column": "c",
                }
            ],
        },
        {"target_field": "y", "hook_required": False, "steps": []},
    ]
    return {
        "institution_id": "u_test",
        "transformation_maps": {
            "cohort": {"plans": plans},
            "course": {"plans": []},
        },
    }


def test_build_envelope_skips_non_hook_plans():
    data = _sample_wrapper(hook_on_x=True)
    env = build_transformation_hook_hitl_envelope_for_entity(
        data, institution_id="u_test", entity_type="cohort"
    )
    assert len(env.items) == 1
    assert env.items[0].target_field == "x"
    assert env.items[0].current_field_mapping == {"target_field": "x"}


def test_build_envelope_empty_when_no_hook_required():
    data = _sample_wrapper(hook_on_x=False)
    env = build_transformation_hook_hitl_envelope_for_entity(
        data, institution_id="u_test", entity_type="cohort"
    )
    assert env.items == []


def test_apply_resolution_accept_partial_preserves_steps(tmp_path: Path):
    data = _sample_wrapper()
    env = build_transformation_hook_hitl_envelope_for_entity(
        data, institution_id="u_test", entity_type="cohort"
    )
    path = tmp_path / "cohort_transformation_hook_hitl.json"
    write_transformation_hook_hitl_envelope(path, env)
    raw = json.loads(path.read_text())
    raw["items"][0]["choice"] = 1  # accept_partial_chain
    path.write_text(json.dumps(raw))
    out = apply_transformation_hook_hitl_resolutions(
        data, cohort_hitl_path=path, course_hitl_path=None
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["hook_required"] is False
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["function_name"] == "cast_string"


def test_apply_resolution_defer_clears_steps(tmp_path: Path):
    data = _sample_wrapper()
    env = build_transformation_hook_hitl_envelope_for_entity(
        data, institution_id="u_test", entity_type="cohort"
    )
    path = tmp_path / "cohort_transformation_hook_hitl.json"
    write_transformation_hook_hitl_envelope(path, env)
    raw = json.loads(path.read_text())
    raw["items"][0]["choice"] = 2  # defer_field_empty_steps
    path.write_text(json.dumps(raw))
    out = apply_transformation_hook_hitl_resolutions(
        data, cohort_hitl_path=path, course_hitl_path=None
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["hook_required"] is False
    assert plan["steps"] == []


def test_check_gate_passes_empty_envelope(tmp_path: Path):
    path = tmp_path / "empty.json"
    write_transformation_hook_hitl_envelope(
        path,
        InstitutionSMATransformationHookHITLItems(
            institution_id="u", entity_type="course", items=[]
        ),
    )
    check_transformation_hook_hitl_gate(path)


def test_check_gate_blocks_pending(tmp_path: Path):
    data = _sample_wrapper()
    env = build_transformation_hook_hitl_envelope_for_entity(
        data, institution_id="u_test", entity_type="cohort"
    )
    path = tmp_path / "pending.json"
    write_transformation_hook_hitl_envelope(path, env)
    from edvise.genai.mapping.shared.hitl import HITLBlockingError

    with pytest.raises(HITLBlockingError):
        check_transformation_hook_hitl_gate(path)
