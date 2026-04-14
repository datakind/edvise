"""Canonical hook module paths."""

from pathlib import Path

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookFunctionSpec, HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    default_hook_module_relpath,
    ensure_hook_spec_file,
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain


def test_default_hook_module_relpath_grain_and_term():
    assert default_hook_module_relpath("u1", HITLDomain.IDENTITY_GRAIN) == (
        "identity_hooks/u1/dedup_hooks.py"
    )
    assert default_hook_module_relpath("u1", HITLDomain.IDENTITY_TERM) == (
        "identity_hooks/u1/term_hooks.py"
    )


def test_ensure_hook_spec_file_overwrites_model_path():
    spec = HookSpec(
        file="wrong/legacy.py",
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f():",
                description="d",
                draft="def f():\n    return 1\n",
            )
        ],
    )
    out = ensure_hook_spec_file(
        spec, institution_id="u1", domain=HITLDomain.IDENTITY_TERM
    )
    assert out.file == "identity_hooks/u1/term_hooks.py"


def test_resolve_hook_module_path(tmp_path: Path) -> None:
    p = resolve_hook_module_path(
        "identity_hooks/u1/dedup_hooks.py", root=tmp_path
    )
    assert p == tmp_path / "identity_hooks" / "u1" / "dedup_hooks.py"


def test_resolve_hook_module_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        resolve_hook_module_path("../evil.py", root=tmp_path)
