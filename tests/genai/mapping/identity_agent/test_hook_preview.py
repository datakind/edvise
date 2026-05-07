"""Tests for hook preview JSON written before UC ``ia_gate_1_hooks``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookSpec,
    HookFunctionSpec,
)
from edvise.genai.mapping.identity_agent.hitl.hook_preview import (
    apply_term_hook_spec_names_from_item_id,
    assemble_hook_spec_drafts_as_module_text,
    hook_slug_from_item_id,
    write_identity_hook_preview_json,
)


def test_write_identity_hook_preview_json_roundtrip(tmp_path: Path) -> None:
    spec = HookSpec(
        file="hooks/x.py",
        functions=[
            HookFunctionSpec(
                name="f",
                description="test",
                draft="def f(x):\n    return x\n",
            ),
        ],
    )
    out = tmp_path / "p.json"
    write_identity_hook_preview_json(
        output_path=out,
        institution_id="school_a",
        domain="identity_grain",
        specs=[("item_1", spec)],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["institution_id"] == "school_a"
    assert data["domain"] == "identity_grain"
    assert len(data["specs"]) == 1
    assert data["specs"][0]["item_id"] == "item_1"
    assert data["specs"][0]["hook_spec"]["file"] == "hooks/x.py"
    assert "review_context" not in data["specs"][0]


def test_write_identity_hook_preview_json_requires_both_paths(tmp_path: Path) -> None:
    spec = HookSpec(
        file="hooks/x.py",
        functions=[
            HookFunctionSpec(
                name="f",
                description="test",
                draft="return 1\n",
            ),
        ],
    )
    out = tmp_path / "p.json"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"institution_id": "u", "datasets": {}}))
    with pytest.raises(ValueError, match="hitl_path and config_path"):
        write_identity_hook_preview_json(
            output_path=out,
            institution_id="school_a",
            domain="identity_term",
            specs=[("id1", spec)],
            config_path=cfg,
        )


def test_write_identity_hook_preview_json_includes_review_context(
    tmp_path: Path,
) -> None:
    hitl_path = tmp_path / "identity_term_hitl.json"
    config_path = tmp_path / "identity_term_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "term",
                "items": [
                    {
                        "item_id": "term_gh_1",
                        "institution_id": "u",
                        "table": "student",
                        "domain": "identity_term",
                        "hitl_question": "Confirm encoding?",
                        "hitl_context": "samples: a,b",
                        "options": [
                            {
                                "option_id": "confirm",
                                "label": "Confirm",
                                "description": "d",
                                "resolution": {
                                    "season_map_replace": [
                                        {"raw": "9", "canonical": "FALL"},
                                    ],
                                },
                                "reentry": "generate_hook",
                            },
                            {
                                "option_id": "custom",
                                "label": "Custom",
                                "description": "d",
                                "resolution": None,
                                "reentry": "terminal",
                            },
                        ],
                        "target": {
                            "institution_id": "u",
                            "table": "student",
                            "config": "term_config",
                            "field": "hook_spec",
                        },
                        "choice": 1,
                        "reviewer_note": "prefer ISO dates",
                    }
                ],
            }
        )
    )
    config_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "datasets": {
                    "student": {
                        "term_config": {
                            "term_col": "term",
                            "year_col": None,
                            "season_col": None,
                            "season_map": [],
                            "exclude_tokens": [],
                            "term_extraction": "hook_required",
                            "hook_spec": None,
                        },
                        "confidence": 0.9,
                        "hitl_flag": True,
                        "reasoning": "r",
                    }
                },
            }
        )
    )
    spec = HookSpec(
        file="identity_hooks/u/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_extractor",
                description="y",
                draft="return 2024\n",
            ),
        ],
    )
    out = tmp_path / "preview.json"
    write_identity_hook_preview_json(
        output_path=out,
        institution_id="u",
        domain="identity_term",
        specs=[("term_gh_1", spec)],
        hitl_path=hitl_path,
        config_path=config_path,
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    rc = data["specs"][0]["review_context"]
    assert rc["hitl_question"] == "Confirm encoding?"
    assert rc["hitl_context"] == "samples: a,b"
    assert rc["reviewer_note"] == "prefer ISO dates"
    assert rc["season_map_replace"] == [{"raw": "9", "canonical": "FALL"}]
    assert rc["table"] == "student"
    assert rc["config_snippet"]["term_config"]["term_col"] == "term"
    assert rc["target"]["field"] == "hook_spec"


def test_hook_slug_from_item_id_strips_institution_and_hook_suffix() -> None:
    assert (
        hook_slug_from_item_id(
            "uni_of_central_florida_student_deg_comp_term_bachelors_hook",
            institution_id="uni_of_central_florida",
        )
        == "student_deg_comp_term_bachelors"
    )
    assert (
        hook_slug_from_item_id("student_deg_comp_term_bachelors_hook")
        == "student_deg_comp_term_bachelors"
    )


def test_term_preview_json_matches_slugged_hook_spec(tmp_path: Path) -> None:
    raw = HookSpec(
        file="identity_hooks/uni_of_central_florida/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_extractor",
                description="y",
                draft=(
                    "def year_extractor(term: str) -> int:\n    return int(term[:4])\n"
                ),
            ),
            HookFunctionSpec(
                name="season_extractor",
                description="s",
                draft=(
                    "def season_extractor(term: str) -> str:\n    return term[-2:]\n"
                ),
            ),
        ],
    )
    item_id = "uni_of_central_florida_student_deg_comp_term_bachelors_hook"
    spec = apply_term_hook_spec_names_from_item_id(
        raw, item_id, institution_id="uni_of_central_florida"
    )
    out = tmp_path / "term_preview.json"
    write_identity_hook_preview_json(
        output_path=out,
        institution_id="uni_of_central_florida",
        domain="identity_term",
        specs=[(item_id, spec)],
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    funcs = data["specs"][0]["hook_spec"]["functions"]
    by_name = {f["name"]: f for f in funcs}
    y = by_name["year_extractor_student_deg_comp_term_bachelors"]
    s = by_name["season_extractor_student_deg_comp_term_bachelors"]
    assert "def year_extractor_student_deg_comp_term_bachelors(" in (
        y.get("draft") or ""
    )
    assert "def season_extractor_student_deg_comp_term_bachelors(" in (
        s.get("draft") or ""
    )


def test_assemble_hook_spec_drafts_as_module_text_joins_defs() -> None:
    text = assemble_hook_spec_drafts_as_module_text(
        {
            "functions": [
                {"name": "a", "draft": "def a():\n    return 1"},
                {"name": "b", "draft": "def b():\n    return 2"},
            ]
        }
    )
    assert "def a():" in text
    assert "def b():" in text
    assert "\n\ndef b()" in text
