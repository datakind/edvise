"""HookSpec → .py materialization."""

import ast
from pathlib import Path

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookFunctionSpec, HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
    materialize_hook_spec_to_file,
    materialize_hook_specs_to_file,
    merge_hook_specs,
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


def test_merge_hook_specs_same_file(tmp_path: Path) -> None:
    a = HookSpec(
        file="helpers/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="year_a",
                description="d",
                draft="def year_a(term: str) -> int:\n    return 1\n",
            )
        ],
    )
    b = HookSpec(
        file="helpers/term_hooks.py",
        functions=[
            HookFunctionSpec(
                name="season_b",
                description="d",
                draft="def season_b(term: str) -> str:\n    return 'x'\n",
            )
        ],
    )
    merged = merge_hook_specs(a, b, repo_root=tmp_path)
    assert merged.file == "helpers/term_hooks.py"
    assert [f.name for f in merged.functions] == ["year_a", "season_b"]


def test_merge_hook_specs_rejects_different_paths(tmp_path: Path) -> None:
    a = HookSpec(
        file="helpers/a.py",
        functions=[
            HookFunctionSpec(
                name="f",
                description="d",
                draft="def f():\n    pass\n",
            )
        ],
    )
    b = HookSpec(
        file="helpers/b.py",
        functions=[
            HookFunctionSpec(
                name="g",
                description="d",
                draft="def g():\n    pass\n",
            )
        ],
    )
    with pytest.raises(HITLValidationError, match="same location"):
        merge_hook_specs(a, b, repo_root=tmp_path)


def test_merge_hook_specs_rejects_duplicate_names(tmp_path: Path) -> None:
    dup = HookFunctionSpec(
        name="f",
        description="d",
        draft="def f():\n    pass\n",
    )
    a = HookSpec(file="helpers/x.py", functions=[dup])
    b = HookSpec(file="helpers/x.py", functions=[dup])
    with pytest.raises(HITLValidationError, match="duplicate function name"):
        merge_hook_specs(a, b, repo_root=tmp_path)


def test_materialize_hook_specs_to_file_writes_merged(tmp_path: Path) -> None:
    specs = [
        HookSpec(
            file="helpers/term_hooks.py",
            functions=[
                HookFunctionSpec(
                    name="year_extractor",
                    description="d",
                    draft=_term_draft(),
                )
            ],
        ),
        HookSpec(
            file="helpers/term_hooks.py",
            functions=[
                HookFunctionSpec(
                    name="season_extractor",
                    description="d",
                    draft="def season_extractor(term: str) -> str:\n    return 'FALL'\n",
                )
            ],
        ),
    ]
    out = materialize_hook_specs_to_file(
        specs, repo_root=tmp_path, domain=HITLDomain.IDENTITY_TERM
    )
    text = out.read_text()
    assert "def year_extractor" in text
    assert "def season_extractor" in text
    ast.parse(text)


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
