"""Tests for draft vs runtime signature comparison."""

import pytest

from edvise.genai.mapping.identity_agent.hitl.hook_generation.signature_check import (
    signature_mismatches,
)


def _fn_ok(a: str, b: int) -> str:
    return a + str(b)


def test_signature_mismatches_empty_draft():
    errs = signature_mismatches(_fn_ok, expected_name="_fn_ok", draft=None)
    assert errs and "empty" in errs[0].lower()


def test_signature_mismatches_param_order():
    draft = """def _fn_ok(b: int, a: str) -> str:
    return a + str(b)
"""
    errs = signature_mismatches(_fn_ok, expected_name="_fn_ok", draft=draft)
    assert any("parameter" in e.lower() for e in errs)


def test_signature_mismatches_return_ann():
    draft = """def _fn_ok(a: str, b: int) -> int:
    return 0
"""
    errs = signature_mismatches(_fn_ok, expected_name="_fn_ok", draft=draft)
    assert any("return annotation" in e.lower() for e in errs)


def test_signature_mismatches_ok_when_draft_matches():
    draft = """def _fn_ok(a: str, b: int) -> str:
    return a + str(b)
"""
    assert signature_mismatches(_fn_ok, expected_name="_fn_ok", draft=draft) == []


def _fn_quoted_pd(group: str) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame()


def test_signature_mismatches_ok_quoted_pd_dataframe_vs_str_runtime():
    """Draft uses -> \"pd.DataFrame\"; runtime annotation is str (forward ref)."""
    draft = """def _fn_quoted_pd(group: str) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame()
"""
    assert (
        signature_mismatches(_fn_quoted_pd, expected_name="_fn_quoted_pd", draft=draft)
        == []
    )


def test_signature_mismatches_ok_quoted_pd_dataframe_vs_resolved_type():
    """Draft -> \"pd.DataFrame\"; runtime is pandas.DataFrame class (evaluated hints)."""
    pd = pytest.importorskip("pandas")

    def _fn_pd_resolved(group: str) -> pd.DataFrame:
        return pd.DataFrame()

    draft = """def _fn_pd_resolved(group: str) -> "pd.DataFrame":
    return pd.DataFrame()
"""
    assert (
        signature_mismatches(
            _fn_pd_resolved, expected_name="_fn_pd_resolved", draft=draft
        )
        == []
    )
