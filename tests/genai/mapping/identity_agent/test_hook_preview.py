"""Tests for hook preview JSON written before UC ``ia_gate_1_hooks``."""

from __future__ import annotations

import json
from pathlib import Path

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec, HookFunctionSpec
from edvise.genai.mapping.identity_agent.hitl.hook_preview import (
    assemble_hook_spec_drafts_as_module_text,
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
