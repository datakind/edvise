"""Unit tests for hitl.resolver config mutations (GrainContract / TermContract shape)."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    _apply_grain_hook_spec_dict,
    _apply_grain_resolution,
    _apply_term_hook_spec_dict,
    _apply_term_resolution,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.hitl.schemas import GrainResolution, TermResolution
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermOrderConfig


def _hook() -> HookSpec:
    return HookSpec(
        file="inst/hooks.py",
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f(x: str) -> str",
                description="t",
            )
        ],
    )


def _term_item(table: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(item_id="item_a", target=SimpleNamespace(table=table))


def _grain_item(table: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(item_id="item_g", target=SimpleNamespace(table=table))


def test_term_col_override_clears_split_columns():
    cfg = {
        "datasets": {
            "t1": {
                "term_config": {
                    "term_col": None,
                    "year_col": "y",
                    "season_col": "s",
                    "season_map": [{"raw": "FA", "canonical": "FALL"}],
                    "term_extraction": "standard",
                }
            }
        }
    }
    res = TermResolution(term_col_override="TERM_COMBINED")
    _apply_term_resolution(cfg, _term_item(), res)
    tc = cfg["datasets"]["t1"]["term_config"]
    assert tc["term_col"] == "TERM_COMBINED"
    assert tc["year_col"] is None
    assert tc["season_col"] is None


def test_season_map_append_validates_entries():
    cfg = {
        "datasets": {
            "t1": {
                "term_config": {
                    "year_col": "y",
                    "season_col": "s",
                    "season_map": [{"raw": "FA", "canonical": "FALL"}],
                    "term_extraction": "standard",
                }
            }
        }
    }
    bad = TermResolution(season_map_append=[{"raw": "X", "canonical": "NOT_A_SEASON"}])
    with pytest.raises(ValidationError):
        _apply_term_resolution(cfg, _term_item(), bad)


def test_apply_grain_hook_spec_dict_sets_policy_required():
    grain_cfg = {
        "dedup_policy": {
            "strategy": "temporal_collapse",
            "sort_by": "dt",
            "sort_ascending": True,
            "keep": "first",
            "notes": "",
        }
    }
    _apply_grain_hook_spec_dict(grain_cfg, _hook())
    dp = grain_cfg["dedup_policy"]
    assert dp["strategy"] == "policy_required"
    assert dp["hook_spec"] is not None
    assert dp["sort_by"] is None
    assert dp["sort_ascending"] is None
    assert dp["keep"] is None
    GrainContract.model_validate(
        {
            "institution_id": "u",
            "table": "t",
            "post_clean_primary_key": ["k"],
            "dedup_policy": dp,
            "row_selection_required": False,
            "join_keys_for_2a": ["k"],
            "confidence": 0.9,
            "hitl_flag": False,
            "reasoning": "r",
        }
    )


def test_apply_term_hook_spec_dict_clears_split_when_term_col_present():
    term_cfg = {
        "term_col": "term",
        "year_col": "y",
        "season_col": "s",
        "season_map": [{"raw": "FA", "canonical": "FALL"}],
        "term_extraction": "standard",
    }
    _apply_term_hook_spec_dict(term_cfg, _hook(), item_id="x")
    assert term_cfg["year_col"] is None
    assert term_cfg["season_col"] is None
    assert term_cfg["term_extraction"] == "hook_required"
    assert term_cfg["hook_spec"] is not None
    TermOrderConfig.model_validate(term_cfg)


def test_apply_term_hook_spec_dict_raises_when_split_only():
    term_cfg = {
        "year_col": "y",
        "season_col": "s",
        "season_map": [{"raw": "FA", "canonical": "FALL"}],
        "term_extraction": "standard",
    }
    with pytest.raises(HITLValidationError, match="Cannot write term hook_spec"):
        _apply_term_hook_spec_dict(term_cfg, _hook(), item_id="x")


def test_grain_resolution_applies_terminal_hook_spec():
    cfg = {
        "datasets": {
            "t1": {
                "grain_contract": {
                    "institution_id": "u",
                    "table": "t1",
                    "post_clean_primary_key": ["k"],
                    "dedup_policy": {
                        "strategy": "policy_required",
                        "sort_by": None,
                        "sort_ascending": None,
                        "keep": None,
                        "notes": "",
                        "hook_spec": None,
                    },
                    "row_selection_required": False,
                    "join_keys_for_2a": ["k"],
                    "confidence": 0.9,
                    "hitl_flag": True,
                    "reasoning": "r",
                }
            }
        }
    }
    res = GrainResolution(hook_spec=_hook())
    _apply_grain_resolution(cfg, _grain_item(), res)
    dp = cfg["datasets"]["t1"]["grain_contract"]["dedup_policy"]
    assert dp["strategy"] == "policy_required"
    assert dp["hook_spec"] is not None


def test_term_resolution_applies_terminal_hook_spec_after_override():
    cfg = {
        "datasets": {
            "t1": {
                "term_config": {
                    "year_col": "y",
                    "season_col": "s",
                    "season_map": [{"raw": "FA", "canonical": "FALL"}],
                    "term_extraction": "standard",
                }
            }
        }
    }
    res = TermResolution(
        term_col_override="term_raw",
        hook_spec=_hook(),
    )
    _apply_term_resolution(cfg, _term_item(), res)
    tc = cfg["datasets"]["t1"]["term_config"]
    assert tc["term_col"] == "term_raw"
    assert tc["year_col"] is None
    assert tc["season_col"] is None
    assert tc["term_extraction"] == "hook_required"
    TermOrderConfig.model_validate(tc)
