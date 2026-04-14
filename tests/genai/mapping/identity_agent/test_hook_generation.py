"""Hook generation prompts and HookSpec parsing."""

import json

import pytest

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation import (
    build_hook_generation_system_prompt,
    build_hook_generation_user_message,
    extract_config_snippet_for_hook_item,
    normalized_column_names_from_raw_headers,
    parse_hook_spec,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLTarget,
    ReentryDepth,
    TermResolution,
)


def _minimal_grain_item(table: str = "t1") -> HITLItem:
    return HITLItem(
        item_id="x",
        institution_id="u1",
        table=table,
        domain=HITLDomain.IDENTITY_GRAIN,
        hitl_question="q",
        hitl_context="ctx",
        options=[
            HITLOption(
                option_id="no_dedup",
                label="No dedup",
                description="d",
                resolution=GrainResolution(dedup_strategy="no_dedup").model_dump(
                    mode="json"
                ),
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="c",
                resolution=None,
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id="u1",
            table=table,
            config="grain_contract",
            field="dedup_policy",
        ),
    )


def _minimal_term_item(table: str = "c1") -> HITLItem:
    return HITLItem(
        item_id="y",
        institution_id="u1",
        table=table,
        domain=HITLDomain.IDENTITY_TERM,
        hitl_question="q",
        hitl_context=None,
        options=[
            HITLOption(
                option_id="map",
                label="Map",
                description="d",
                resolution=TermResolution(exclude_tokens=["Med"]).model_dump(
                    mode="json"
                ),
                reentry=ReentryDepth.TERMINAL,
            ),
            HITLOption(
                option_id="custom",
                label="Custom",
                description="c",
                resolution=None,
                reentry=ReentryDepth.GENERATE_HOOK,
            ),
        ],
        target=HITLTarget(
            institution_id="u1",
            table=table,
            config="term_config",
            field="hook_spec",
        ),
    )


def test_build_hook_generation_system_prompt_grain_term():
    g = build_hook_generation_system_prompt(HITLDomain.IDENTITY_GRAIN)
    assert "HookSpec" in g and "dedup" in g.lower()
    assert "machine-round-trip" not in g.lower()
    assert "ast.parse" in g
    assert "normalized_columns" in g
    t = build_hook_generation_system_prompt(HITLDomain.IDENTITY_TERM)
    assert "year_extractor" in t and "season_extractor" in t
    assert "validate_hook" in t
    assert "normalized_columns" in t


def test_normalized_column_names_from_raw_headers_snake_case():
    assert normalized_column_names_from_raw_headers(["Major", "MAJOR_CODE"]) == [
        "major",
        "major_code",
    ]


def test_build_hook_generation_user_message_includes_normalized_columns():
    item = _minimal_grain_item("programs")
    cfg = {
        "institution_id": "u1",
        "datasets": {
            "programs": {
                "grain_contract": {"table": "programs", "institution_id": "u1"},
            }
        },
    }
    sn = extract_config_snippet_for_hook_item(cfg, item)
    raw = build_hook_generation_user_message(
        item,
        sn,
        normalized_columns=["major", "major_code", "learner_id"],
    )
    payload = json.loads(raw)
    assert payload["normalized_columns"] == ["major", "major_code", "learner_id"]
    payload2 = json.loads(build_hook_generation_user_message(item, sn))
    assert "normalized_columns" not in payload2


def test_build_hook_generation_system_prompt_rejects_unknown():
    with pytest.raises(ValueError, match="not supported"):
        build_hook_generation_system_prompt(HITLDomain.SCHEMA_MAPPING)


def test_parse_hook_spec_dict_and_fenced_json():
    spec = {
        "functions": [
            {
                "name": "f",
                "signature": "def f(x: str) -> int",
                "description": "test",
                "draft": "def f(x: str) -> int:\n    return 1\n",
            }
        ],
    }
    h1 = parse_hook_spec(spec)
    assert isinstance(h1, HookSpec)
    assert h1.file is None
    text = "```json\n" + json.dumps(spec) + "\n```"
    h2 = parse_hook_spec(text)
    assert h2.functions[0].name == h1.functions[0].name


def test_extract_config_snippet_grain():
    item = _minimal_grain_item("enroll")
    cfg = {
        "institution_id": "u1",
        "datasets": {
            "enroll": {
                "grain_contract": {"table": "enroll", "institution_id": "u1"},
            }
        },
    }
    sn = extract_config_snippet_for_hook_item(cfg, item)
    assert "grain_contract" in sn
    assert sn["grain_contract"]["table"] == "enroll"


def test_extract_config_snippet_term_null():
    item = _minimal_term_item("course")
    cfg = {
        "datasets": {
            "course": {"term_config": None},
        },
    }
    sn = extract_config_snippet_for_hook_item(cfg, item)
    assert sn["term_config"] is None
