"""Canonical hook module paths."""

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookFunctionSpec, HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    default_hook_module_relpath,
    ensure_hook_spec_file,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain


def test_default_hook_module_relpath_grain_and_term():
    assert default_hook_module_relpath("u1", HITLDomain.IDENTITY_GRAIN) == (
        "pipelines/u1/helpers/dedup_hooks.py"
    )
    assert default_hook_module_relpath("u1", HITLDomain.IDENTITY_TERM) == (
        "pipelines/u1/helpers/term_hooks.py"
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
    assert out.file == "pipelines/u1/helpers/term_hooks.py"
