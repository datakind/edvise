"""Unit tests for hitl.resolver config mutations (GrainContract / TermContract shape)."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainAmbiguityHITLContext,
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    InstitutionHITLItems,
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
    resolve_items,
    _apply_grain_hook_spec_dict,
    _apply_grain_resolution,
    _apply_term_hook_spec_dict,
    _apply_term_resolution,
)
from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    TermOrderConfig,
)


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
        season_map_replace=[
            {"raw": "2", "canonical": "SPRING"},
            {"raw": "9", "canonical": "FALL"},
        ],
    )
    _apply_term_resolution(cfg, _term_item(), res)
    tc = cfg["datasets"]["t1"]["term_config"]
    assert tc["term_col"] == "term_raw"
    assert tc["year_col"] is None
    assert tc["season_col"] is None
    assert tc["term_extraction"] == "hook_required"
    assert tc["season_map"] == [
        {"raw": "2", "canonical": "SPRING"},
        {"raw": "9", "canonical": "FALL"},
    ]
    TermOrderConfig.model_validate(tc)


def test_season_map_replace_can_clear_season_map():
    cfg = {
        "datasets": {
            "t1": {
                "term_config": {
                    "term_col": "term",
                    "season_map": [{"raw": "FA", "canonical": "FALL"}],
                    "term_extraction": "hook_required",
                    "hook_spec": None,
                }
            }
        }
    }
    res = TermResolution(season_map_replace=[])
    _apply_term_resolution(cfg, _term_item(), res)
    assert cfg["datasets"]["t1"]["term_config"]["season_map"] == []


def test_resolve_items_fans_out_term_resolution_to_hook_group_tables(tmp_path):
    """Same season_map must land on every dataset in hook_group_tables."""
    hitl_path = tmp_path / "identity_term_hitl.json"
    config_path = tmp_path / "identity_term_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "term",
                "items": [
                    {
                        "item_id": "shared_term",
                        "institution_id": "u",
                        "table": "student",
                        "domain": "identity_term",
                        "hook_group_id": "g1",
                        "hook_group_tables": ["student", "course"],
                        "hitl_question": "q",
                        "hitl_context": "c",
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
                                "reentry": "terminal",
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
                            "season_map": [],
                            "term_extraction": "standard",
                        },
                        "confidence": 0.9,
                        "hitl_flag": True,
                        "reasoning": "r",
                    },
                    "course": {
                        "term_config": {
                            "term_col": "term",
                            "season_map": [],
                            "term_extraction": "standard",
                        },
                        "confidence": 0.9,
                        "hitl_flag": True,
                        "reasoning": "r",
                    },
                },
            }
        )
    )
    resolve_items(hitl_path, config_path)
    out = json.loads(config_path.read_text())
    sm = [{"raw": "9", "canonical": "FALL"}]
    assert out["datasets"]["student"]["term_config"]["season_map"] == sm
    assert out["datasets"]["course"]["term_config"]["season_map"] == sm


def test_resolve_items_coerces_dict_resolution_for_term(tmp_path):
    """JSON stores option.resolution as dict; resolve_items must apply term mutations."""
    hitl_path = tmp_path / "identity_term_hitl.json"
    config_path = tmp_path / "identity_term_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "term",
                "items": [
                    {
                        "item_id": "term_item_1",
                        "institution_id": "u",
                        "table": "student",
                        "domain": "identity_term",
                        "hitl_question": "Confirm term encoding?",
                        "hitl_context": "1192",
                        "options": [
                            {
                                "option_id": "confirm",
                                "label": "Confirm",
                                "description": "Apply resolution",
                                "resolution": {
                                    "season_map_replace": [
                                        {"raw": "2", "canonical": "SPRING"},
                                        {"raw": "9", "canonical": "FALL"},
                                    ],
                                    "hook_spec": {
                                        "functions": [
                                            {
                                                "name": "year_extractor_shared",
                                                "description": "y",
                                                "draft": (
                                                    "def year_extractor_shared(term: str) -> int:\n"
                                                    "    return 2020\n"
                                                ),
                                            },
                                            {
                                                "name": "season_extractor_shared",
                                                "description": "s",
                                                "draft": (
                                                    "def season_extractor_shared(term: str) -> str:\n"
                                                    '    return "9"\n'
                                                ),
                                            },
                                        ],
                                    },
                                },
                                "reentry": "terminal",
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
                        "institution_id": "u",
                        "table": "student",
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
    resolve_items(hitl_path, config_path)
    out = json.loads(config_path.read_text())
    sm = out["datasets"]["student"]["term_config"]["season_map"]
    assert sm == [
        {"raw": "2", "canonical": "SPRING"},
        {"raw": "9", "canonical": "FALL"},
    ]
    assert (
        out["datasets"]["student"]["term_config"]["term_extraction"] == "hook_required"
    )
    assert out["datasets"]["student"]["term_config"]["hook_spec"] is not None


def test_resolve_items_generate_hook_still_applies_season_map_replace(tmp_path):
    """reentry=generate_hook defers hook_spec; season_map_replace must still apply."""
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
                        "hitl_question": "q",
                        "hitl_context": "c",
                        "options": [
                            {
                                "option_id": "confirm",
                                "label": "Confirm",
                                "description": "d",
                                "resolution": {
                                    "season_map_replace": [
                                        {"raw": "9", "canonical": "FALL"},
                                    ],
                                    "hook_spec": {
                                        "functions": [
                                            {
                                                "name": "y",
                                                "description": "y",
                                                "draft": "def y(t: str) -> int:\n    return 1\n",
                                            },
                                            {
                                                "name": "s",
                                                "description": "s",
                                                "draft": 'def s(t: str) -> str:\n    return "9"\n',
                                            },
                                        ],
                                    },
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
    resolve_items(hitl_path, config_path)
    out = json.loads(config_path.read_text())
    assert out["datasets"]["student"]["term_config"]["season_map"] == [
        {"raw": "9", "canonical": "FALL"},
    ]
    assert out["datasets"]["student"]["term_config"]["hook_spec"] is None


def test_resolve_items_custom_generate_hook_applies_partial_season_map_replace(
    tmp_path,
):
    """custom + generate_hook may carry season_map_replace only; reviewer_note drives hook code."""
    hitl_path = tmp_path / "identity_term_hitl.json"
    config_path = tmp_path / "identity_term_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "term",
                "items": [
                    {
                        "item_id": "term_custom_map",
                        "institution_id": "u",
                        "table": "student",
                        "domain": "identity_term",
                        "hook_group_id": "g",
                        "hook_group_tables": ["student"],
                        "hitl_question": "q",
                        "hitl_context": "c",
                        "options": [
                            {
                                "option_id": "confirm",
                                "label": "Confirm",
                                "description": "d",
                                "resolution": {
                                    "season_map_replace": [
                                        {"raw": "9", "canonical": "FALL"},
                                    ],
                                    "hook_spec": {
                                        "functions": [
                                            {
                                                "name": "y",
                                                "description": "y",
                                                "draft": "def y(t: str) -> int:\n    return 1\n",
                                            },
                                            {
                                                "name": "s",
                                                "description": "s",
                                                "draft": 'def s(t: str) -> str:\n    return "9"\n',
                                            },
                                        ],
                                    },
                                },
                                "reentry": "generate_hook",
                            },
                            {
                                "option_id": "custom",
                                "label": "Custom",
                                "description": "d",
                                "resolution": {
                                    "season_map_replace": [
                                        {"raw": "2", "canonical": "SPRING"},
                                        {"raw": "6", "canonical": "SUMMER"},
                                        {"raw": "9", "canonical": "FALL"},
                                    ],
                                },
                                "reentry": "generate_hook",
                            },
                        ],
                        "target": {
                            "institution_id": "u",
                            "table": "student",
                            "config": "term_config",
                            "field": "hook_spec",
                        },
                        "choice": 2,
                        "reviewer_note": "Last digit encodes season; 2/6/9 only.",
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
    resolve_items(hitl_path, config_path)
    out = json.loads(config_path.read_text())
    assert out["datasets"]["student"]["term_config"]["season_map"] == [
        {"raw": "2", "canonical": "SPRING"},
        {"raw": "6", "canonical": "SUMMER"},
        {"raw": "9", "canonical": "FALL"},
    ]
    assert out["datasets"]["student"]["term_config"]["hook_spec"] is None


def test_resolve_items_coerces_dict_resolution_for_grain(tmp_path):
    hitl_path = tmp_path / "identity_grain_hitl.json"
    config_path = tmp_path / "identity_grain_output.json"
    hitl_path.write_text(
        json.dumps(
            {
                "institution_id": "u",
                "domain": "grain",
                "items": [
                    {
                        "item_id": "grain_item_1",
                        "institution_id": "u",
                        "table": "t1",
                        "domain": "identity_grain",
                        "hitl_question": "PK?",
                        "hitl_context": "k",
                        "options": [
                            {
                                "option_id": "fix",
                                "label": "Fix",
                                "description": "d",
                                "resolution": {
                                    "candidate_key_override": ["learner_id", "term"],
                                },
                                "reentry": "terminal",
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
                            "table": "t1",
                            "config": "grain_contract",
                            "field": "dedup_policy",
                        },
                        "choice": 1,
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
                    "t1": {
                        "grain_contract": {
                            "institution_id": "u",
                            "table": "t1",
                            "post_clean_primary_key": ["id"],
                            "dedup_policy": {
                                "strategy": "policy_required",
                                "sort_by": None,
                                "sort_ascending": None,
                                "keep": None,
                                "notes": "",
                                "hook_spec": None,
                            },
                            "row_selection_required": False,
                            "join_keys_for_2a": ["id"],
                            "confidence": 0.9,
                            "hitl_flag": True,
                            "reasoning": "r",
                        }
                    }
                },
            }
        )
    )
    resolve_items(hitl_path, config_path)
    out = json.loads(config_path.read_text())
    assert out["datasets"]["t1"]["grain_contract"]["post_clean_primary_key"] == [
        "learner_id",
        "term",
    ]


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
                "draft": 'def season_extractor_wrong(term: str) -> str:\n    return "9"\n',
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
                draft='def season_extractor_shared(term: str) -> str:\n    return "9"\n',
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
        fnames = [
            f["name"]
            for f in out["datasets"][ds]["term_config"]["hook_spec"]["functions"]
        ]
        assert fnames == ["year_extractor_shared", "season_extractor_shared"]
        assert (
            out["datasets"][ds]["term_config"]["hook_spec"]["file"]
            == "identity_hooks/u/term_hooks.py"
        )
    TermOrderConfig.model_validate(out["datasets"]["course"]["term_config"])


def test_hitl_item_hook_group_tables_requires_hook_group_id():
    with pytest.raises(
        ValidationError, match="hook_group_tables requires hook_group_id"
    ):
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
                    resolution=TermResolution(exclude_tokens=["x"]).model_dump(
                        mode="json"
                    ),
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


def test_institution_hitl_items_accepts_structured_grain_hitl_context():
    raw = {
        "institution_id": "synthetic_metro_research_uni",
        "domain": "grain",
        "items": [
            {
                "item_id": "synthetic_metro_research_uni_student_grain_ambiguity",
                "institution_id": "synthetic_metro_research_uni",
                "table": "student",
                "domain": "identity_grain",
                "hook_group_id": None,
                "hook_group_tables": None,
                "hitl_question": "Which grain defines a row?",
                "hitl_context": {
                    "candidate_keys": [
                        {
                            "rank": 1,
                            "columns": [
                                "STUDENT_ID",
                                "CREDITS_EARNED_BOT",
                                "CUM_GPA_END_TERM",
                            ],
                            "uniqueness_score": 0.9953,
                            "notes": "includes measure columns",
                        },
                        {
                            "rank": 2,
                            "columns": ["STUDENT_ID", "TERM_DESC"],
                            "uniqueness_score": 0.85,
                            "notes": "natural semantic grain",
                        },
                        {
                            "rank": 3,
                            "columns": ["STUDENT_ID"],
                            "uniqueness_score": 0.42,
                            "notes": "high duplicate rate",
                        },
                    ],
                    "variance_profile": {
                        "COHORT_YEAR": "25%–58.8% within groups",
                        "TERM_DESC": "41.2%–62% within groups",
                    },
                },
                "options": [
                    {
                        "option_id": "student_term",
                        "label": "Student-term grain",
                        "description": "d",
                        "resolution": {
                            "candidate_key_override": ["STUDENT_ID", "TERM_DESC"],
                            "dedup_strategy": "true_duplicate",
                        },
                        "reentry": "terminal",
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
                    "institution_id": "synthetic_metro_research_uni",
                    "table": "student",
                    "config": "grain_contract",
                    "field": "dedup_policy",
                },
            }
        ],
    }
    env = InstitutionHITLItems.model_validate(raw)
    assert isinstance(env.items[0].hitl_context, GrainAmbiguityHITLContext)
    assert env.items[0].hitl_context.candidate_keys[2].uniqueness_score == 0.42
