"""Unit tests for hitl.resolver config mutations (GrainContract / TermContract shape)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    ReentryDepth,
    TermResolution,
)

from edvise.genai.mapping.identity_agent.grain_inference.schemas import (
    HookFunctionSpec,
    HookSpec,
)
from edvise.genai.mapping.identity_agent.hitl.resolver import (
    HITLValidationError,
    apply_hook_spec,
    _apply_grain_hook_spec_dict,
    _apply_grain_resolution,
    _apply_term_hook_spec_dict,
    _apply_term_resolution,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermOrderConfig


def _hook() -> HookSpec:
    return HookSpec(
        functions=[
            HookFunctionSpec(
                name="f",
                signature="def f(x: str) -> str",
                description="t",
                draft="def f(x: str) -> str:\n    return x\n",
            )
        ],
    )


def _term_item(table: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(
        item_id="item_a",
        institution_id="u",
        target=SimpleNamespace(table=table),
    )


def _grain_item(table: str = "t1") -> SimpleNamespace:
    return SimpleNamespace(
        item_id="item_g",
        institution_id="u",
        target=SimpleNamespace(table=table),
    )


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
    _apply_grain_hook_spec_dict(grain_cfg, _hook(), institution_id="u")
    dp = grain_cfg["dedup_policy"]
    assert dp["strategy"] == "policy_required"
    assert dp["hook_spec"] is not None
    assert dp["hook_spec"]["file"] == "identity_hooks/u/dedup_hooks.py"
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
    _apply_term_hook_spec_dict(term_cfg, _hook(), item_id="x", institution_id="u")
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
        _apply_term_hook_spec_dict(term_cfg, _hook(), item_id="x", institution_id="u")


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
        season_map_replace=[{"raw": "FA", "canonical": "FALL"}],
    )
    _apply_term_resolution(cfg, _term_item(), res)
    tc = cfg["datasets"]["t1"]["term_config"]
    assert tc["term_col"] == "term_raw"
    assert tc["year_col"] is None
    assert tc["season_col"] is None
    assert tc["term_extraction"] == "hook_required"
    TermOrderConfig.model_validate(tc)


def test_apply_hook_spec_term_fanout_uses_hook_group_tables(tmp_path):
    """
    apply_hook_spec(apply_to_group=True) writes the same hook_spec to every dataset listed in
    HITLItem.hook_group_tables, not only the HITL anchor table.
    """
    hitl_path = tmp_path / "identity_term_hitl.json"
    config_path = tmp_path / "identity_term_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "term",
                "items": [
                    {
                        "item_id": "item1",
                        "institution_id": "u",
                        "table": "student",
                        "domain": "identity_term",
                        "hook_group_id": "shared_g",
                        "hook_group_tables": ["student", "course"],
                        "hitl_question": "Confirm encoding?",
                        "hitl_context": "1192",
                        "options": [
                            {
                                "option_id": "confirm",
                                "label": "Confirm",
                                "description": "d",
                                "resolution": None,
                                "reentry": "generate_hook",
                            },
                            {
                                "option_id": "custom",
                                "label": "Custom",
                                "description": "d",
                                "resolution": None,
                                "reentry": "generate_hook",
                            },
                        ],
                        "target": {
                            "institution_id": "u",
                            "table": "student",
                            "config": "term_config",
                            "field": "hook_spec",
                        },
                        "choice": 1,
                    }
                ],
            }
        )
    )
    stale = {
        "functions": [
            {
                "name": "year_extractor_wrong",
                "description": "d",
                "draft": "def year_extractor_wrong(term: str) -> int:\n    return 1\n",
            },
            {
                "name": "season_extractor_wrong",
                "description": "d",
                "draft": "def season_extractor_wrong(term: str) -> str:\n    return \"9\"\n",
            },
        ]
    }
    config_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "datasets": {
                    "student": {
                        "term_config": {
                            "term_col": "term",
                            "season_map": [],
                            "exclude_tokens": [],
                            "term_extraction": "hook_required",
                            "hook_spec": stale,
                        }
                    },
                    "course": {
                        "term_config": {
                            "term_col": "term",
                            "season_map": [],
                            "exclude_tokens": [],
                            "term_extraction": "hook_required",
                            "hook_spec": stale,
                        }
                    },
                },
            }
        )
    )
    new_spec = HookSpec(
        file=None,
        functions=[
            HookFunctionSpec(
                name="year_extractor_shared",
                description="y",
                draft="def year_extractor_shared(term: str) -> int:\n    return 2019\n",
            ),
            HookFunctionSpec(
                name="season_extractor_shared",
                description="s",
                draft="def season_extractor_shared(term: str) -> str:\n    return \"9\"\n",
            ),
        ],
    )
    apply_hook_spec(
        hitl_path=hitl_path,
        config_path=config_path,
        item_id="item1",
        hook_spec=new_spec,
        apply_to_group=True,
        materialize=False,
    )
    out = json.loads(config_path.read_text())
    for ds in ("student", "course"):
        fnames = [f["name"] for f in out["datasets"][ds]["term_config"]["hook_spec"]["functions"]]
        assert fnames == ["year_extractor_shared", "season_extractor_shared"]
        assert (
            out["datasets"][ds]["term_config"]["hook_spec"]["file"]
            == "identity_hooks/u/term_hooks.py"
        )
    TermOrderConfig.model_validate(out["datasets"]["course"]["term_config"])


def test_hitl_item_hook_group_tables_requires_hook_group_id():
    with pytest.raises(ValidationError, match="hook_group_tables requires hook_group_id"):
        HITLItem(
            item_id="x",
            institution_id="u",
            table="student",
            domain=HITLDomain.IDENTITY_TERM,
            hook_group_id=None,
            hook_group_tables=["student", "course"],
            hitl_question="q",
            hitl_context=None,
            options=[
                HITLOption(
                    option_id="a",
                    label="l",
                    description="d",
                    resolution=TermResolution(exclude_tokens=["x"]).model_dump(mode="json"),
                    reentry=ReentryDepth.TERMINAL,
                ),
                HITLOption(
                    option_id="custom",
                    label="c",
                    description="d",
                    resolution=None,
                    reentry=ReentryDepth.GENERATE_HOOK,
                ),
            ],
            target=HITLTarget(
                institution_id="u",
                table="student",
                config="term_config",
                field="hook_spec",
            ),
        )
