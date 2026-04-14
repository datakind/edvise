"""HookSpec → .py materialization."""

import ast
from pathlib import Path

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookFunctionSpec, HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
    materialize_hook_spec_to_file,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import HITLValidationError
from edvise.genai.mapping.identity_agent.hitl.schemas import HITLDomain


def _grain_draft() -> str:
    return """def f(df):
    import pandas as pd
    return df.drop_duplicates()
"""


def _term_draft() -> str:
    return """def year_extractor(term: str) -> int:
    return int(term[:4])
"""


def test_materialize_writes_full_def_verbatim(tmp_path: Path) -> None:
    spec = HookSpec(
        file="helpers/dedup_hooks.py",
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f(df)",
                description="d",
                draft=_grain_draft(),
            )
        ],
    )
    out = materialize_hook_spec_to_file(
        spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_GRAIN
    )
    assert out == tmp_path / "helpers" / "dedup_hooks.py"
    text = out.read_text()
    assert "# HITL domain: identity_grain" in text
    assert "def f(df):" in text
    assert "import pandas as pd" in text
    assert "return df.drop_duplicates()" in text
    ast.parse(text)


def test_materialize_term_full_def_no_auto_wrap(tmp_path: Path) -> None:
    spec = HookSpec(
        file="helpers/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_extractor",
                signature="def year_extractor(term: str) -> int",
                description="d",
                draft=_term_draft(),
            )
        ],
    )
    out = materialize_hook_spec_to_file(
        spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_TERM
    )
    text = out.read_text()
    assert "def year_extractor(term: str) -> int:" in text
    assert "return int(term[:4])" in text
    ast.parse(text)


def test_materialize_rejects_path_traversal(tmp_path: Path) -> None:
    spec = HookSpec(
        file="../evil.py",
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f()",
                description="d",
                draft="def f():\n    pass\n",
            )
        ],
    )
    with pytest.raises(HITLValidationError, match="escapes root"):
        materialize_hook_spec_to_file(
            spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_GRAIN
        )


def test_materialize_rejects_null_file(tmp_path: Path) -> None:
    spec = HookSpec(
        file=None,
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f()",
                description="d",
                draft="def f():\n    pass\n",
            )
        ],
    )
    with pytest.raises(HITLValidationError, match="hook_spec.file is null"):
        materialize_hook_spec_to_file(
            spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_GRAIN
        )


def test_materialize_rejects_null_draft(tmp_path: Path) -> None:
    spec = HookSpec(
        file="helpers/x.py",
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f()",
                description="d",
                draft=None,
            )
        ],
    )
    with pytest.raises(HITLValidationError, match="empty draft"):
        materialize_hook_spec_to_file(
            spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_GRAIN
        )


def test_ast_parse_failure_raises(tmp_path: Path) -> None:
    spec = HookSpec(
        file="helpers/bad.py",
        functions=[
            HookFunctionSpec(
                name="f",
                description="d",
                draft="def f(  # broken",
            )
        ],
    )
    with pytest.raises(HITLValidationError, match="ast.parse"):
        materialize_hook_spec_to_file(
            spec, repo_root=tmp_path, domain=HITLDomain.IDENTITY_GRAIN
        )
