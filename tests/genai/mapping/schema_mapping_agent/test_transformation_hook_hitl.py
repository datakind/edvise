"""Step 2b hook_required HITL envelopes and resolver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_required_hitl import (
    InstitutionSMATransformationHookHITLItems,
    apply_transformation_hook_hitl_resolutions,
    attach_materialized_hook_specs_to_plans,
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


def _hook_preview_row(*, target_field: str, entity_type: str) -> dict:
    return {
        "item_id": f"u_test_{entity_type}_{target_field}_hook_required",
        "hook_spec": {
            "file": "transform_hooks.py",
            "functions": [
                {
                    "name": f"transform_{entity_type}_{target_field}",
                    "description": "generated",
                    "draft": f'def transform_{entity_type}_{target_field}(s):\n    return s\n',
                }
            ],
        },
        "review_context": {"entity_type": entity_type, "target_field": target_field},
    }


def test_attach_materialized_hook_specs_sets_hook_spec_on_matching_plan():
    """Closes the gap PR #155 introduced: materialize wrote transform_hooks.py, but nothing
    ever pointed the plan at it, so hook_required plans stayed unresolved gaps forever."""
    data = _sample_wrapper(hook_on_x=True)
    rows = [_hook_preview_row(target_field="x", entity_type="cohort")]
    out = attach_materialized_hook_specs_to_plans(
        data, entity_type="cohort", preview_rows=rows
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert plan["target_field"] == "x"
    # hook_required stays true — it still means "no built-in utility chain covers this"
    assert plan["hook_required"] is True
    assert plan["hook_spec"]["file"] == "transform_hooks.py"
    assert plan["hook_spec"]["functions"][0]["name"] == "transform_cohort_x"
    # original dict is untouched
    assert "hook_spec" not in data["transformation_maps"]["cohort"]["plans"][0]


def test_attach_materialized_hook_specs_ignores_wrong_entity_type():
    data = _sample_wrapper(hook_on_x=True)
    rows = [_hook_preview_row(target_field="x", entity_type="course")]
    out = attach_materialized_hook_specs_to_plans(
        data, entity_type="cohort", preview_rows=rows
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert "hook_spec" not in plan


def test_attach_materialized_hook_specs_ignores_non_hook_required_plan():
    data = _sample_wrapper(hook_on_x=False)
    rows = [_hook_preview_row(target_field="x", entity_type="cohort")]
    out = attach_materialized_hook_specs_to_plans(
        data, entity_type="cohort", preview_rows=rows
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert "hook_spec" not in plan


def test_attach_materialized_hook_specs_ignores_unknown_target_field():
    data = _sample_wrapper(hook_on_x=True)
    rows = [_hook_preview_row(target_field="does_not_exist", entity_type="cohort")]
    out = attach_materialized_hook_specs_to_plans(
        data, entity_type="cohort", preview_rows=rows
    )
    plan = out["transformation_maps"]["cohort"]["plans"][0]
    assert "hook_spec" not in plan
