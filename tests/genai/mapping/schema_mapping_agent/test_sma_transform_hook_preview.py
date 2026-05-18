"""Tests for SMA transform hook preview generation and artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from edvise.genai.mapping.shared.hitl.hook_spec.schemas import (
    HITLDomain,
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.shared.hitl.hook_spec.paths import ensure_hook_spec_file
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_generation import (
    generate_sma_transform_hook_spec,
    load_hook_specs_from_sma_preview_path,
    manifest_mapping_for_target,
    sma_transform_hook_item_id,
    write_sma_transform_hook_preview_json,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.hitl.hook_required_hitl import (
    iter_hook_required_plans,
)


def test_iter_hook_required_plans_filters() -> None:
    td = {
        "transformation_maps": {
            "cohort": {
                "plans": [
                    {"target_field": "gpa", "hook_required": True},
                    {"target_field": "ok", "hook_required": False},
                ]
            }
        }
    }
    got = iter_hook_required_plans(td, "cohort")
    assert len(got) == 1 and got[0]["target_field"] == "gpa"


def test_manifest_mapping_for_target() -> None:
    mm = {
        "manifests": {
            "course": {
                "mappings": [
                    {"target_field": "credits", "source_column": "cr"},
                ]
            }
        }
    }
    hit = manifest_mapping_for_target(mm, "course", "credits")
    assert hit == {"target_field": "credits", "source_column": "cr"}
    assert manifest_mapping_for_target(mm, "course", "missing") is None


def test_sma_transform_hook_item_id_stable() -> None:
    assert (
        sma_transform_hook_item_id("u1", "cohort", "Student GPA")
        == "u1_cohort_student_gpa_hook_required"
    )


def test_ensure_hook_spec_file_transform_paths() -> None:
    spec = HookSpec(
        functions=[
            HookFunctionSpec(
                name="transform_x",
                description="test",
                draft="def transform_x(s):\n    return s",
            )
        ]
    )
    out = ensure_hook_spec_file(
        spec, institution_id="inst_a", domain=HITLDomain.TRANSFORM
    )
    assert out.file == "transform_hooks.py"


def test_write_preview_roundtrip_load_specs(tmp_path: Path) -> None:
    spec = HookSpec(
        file="transform_hooks.py",
        functions=[
            HookFunctionSpec(
                name="transform_gpa",
                description="gpa",
                draft="def transform_gpa(s):\n    return s",
            )
        ],
    )
    rows = [
        {
            "item_id": "x",
            "hook_spec": spec.model_dump(mode="json"),
            "review_context": {"target_field": "gpa"},
        }
    ]
    p = tmp_path / "cohort_transformation_hook_preview.json"
    write_sma_transform_hook_preview_json(
        output_path=p,
        institution_id="u",
        domain="schema_mapping_transform_cohort",
        spec_rows=rows,
    )
    loaded = load_hook_specs_from_sma_preview_path(p)
    assert len(loaded) == 1
    assert loaded[0].functions[0].name == "transform_gpa"


def test_generate_sma_transform_hook_spec_mock_llm() -> None:
    plan = {
        "target_field": "gpa",
        "hook_required": True,
        "reviewer_notes": "custom conversion",
        "steps": [],
    }
    model_json = json.dumps(
        {
            "functions": [
                {
                    "name": "transform_cohort_gpa",
                    "description": "normalize gpa",
                    "draft": "def transform_cohort_gpa(s):\n    return s",
                }
            ]
        }
    )

    def llm(_sys: str, _user: str) -> str:
        return model_json

    spec = generate_sma_transform_hook_spec(
        item_id="u_cohort_gpa_hook_required",
        institution_id="u",
        entity_type="cohort",
        target_field="gpa",
        plan=plan,
        manifest_record=None,
        llm_complete=llm,
    )
    assert spec.file == "transform_hooks.py"
    assert spec.functions[0].name == "transform_cohort_gpa"
