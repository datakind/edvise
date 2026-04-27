import copy

import pytest
from pydantic import BaseModel

from edvise.genai.mapping.shared.schema_utils import (
    _inline_refs,
    to_gateway_schema,
    to_gateway_schema_from_dict,
)


class Simple(BaseModel):
    name: str
    count: int


class Inner(BaseModel):
    v: str


class Outer(BaseModel):
    inner: Inner


class L2(BaseModel):
    z: int


class L1(BaseModel):
    child: L2


class Root(BaseModel):
    item: L1


def _assert_no_defs_or_refs(obj) -> None:
    if isinstance(obj, dict):
        assert "$ref" not in obj
        assert "$defs" not in obj
        for v in obj.values():
            _assert_no_defs_or_refs(v)
    elif isinstance(obj, list):
        for i in obj:
            _assert_no_defs_or_refs(i)


def test_simple_model_schema_matches_raw_minus_wrapper():
    raw = Simple.model_json_schema()
    expected = copy.deepcopy(raw)
    expected.pop("$defs", None)
    out = to_gateway_schema(Simple, "SimpleDoc")
    assert out["json_schema"]["schema"] == expected


def test_nested_defs_and_refs_fully_inlined():
    out = to_gateway_schema(Outer, "OuterDoc")["json_schema"]["schema"]
    _assert_no_defs_or_refs(out)
    assert out["properties"]["inner"]["type"] == "object"
    assert out["properties"]["inner"]["properties"]["v"]["type"] == "string"


def test_gateway_wrapper_shape():
    out = to_gateway_schema(Simple, "MySchema", strict=True)
    assert out["type"] == "json_schema"
    assert set(out["json_schema"].keys()) == {"name", "strict", "schema"}
    assert out["json_schema"]["name"] == "MySchema"
    assert out["json_schema"]["strict"] is True
    assert isinstance(out["json_schema"]["schema"], dict)


def test_two_level_nesting_inlines_and_terminates():
    """Deep $ref chains (e.g. Root -> L1 -> L2) resolve without looping."""
    out = to_gateway_schema(Root, "RootDoc")["json_schema"]["schema"]
    _assert_no_defs_or_refs(out)
    item = out["properties"]["item"]
    assert item["title"] == "L1"
    assert item["properties"]["child"]["title"] == "L2"
    assert item["properties"]["child"]["properties"]["z"]["type"] == "integer"


def test_inline_refs_private_helper_matches_to_gateway_inner_schema():
    raw = Outer.model_json_schema()
    inlined = _inline_refs(copy.deepcopy(raw))
    assert inlined == to_gateway_schema(Outer, "x")["json_schema"]["schema"]


@pytest.mark.parametrize("strict", [False, True])
def test_strict_flag_passed_through(strict: bool):
    out = to_gateway_schema(Simple, "S", strict=strict)
    assert out["json_schema"]["strict"] is strict


def test_to_gateway_schema_delegates_to_from_dict():
    raw = Simple.model_json_schema()
    a = to_gateway_schema(Simple, "Same", strict=True)
    b = to_gateway_schema_from_dict("Same", raw, strict=True)
    assert a == b


def test_to_gateway_schema_from_dict_does_not_mutate_input():
    schema = {
        "$defs": {"Inner": {"type": "object", "properties": {"v": {"type": "string"}}}},
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "required": ["inner"],
        "type": "object",
    }
    snapshot = copy.deepcopy(schema)
    to_gateway_schema_from_dict("doc", schema)
    assert schema == snapshot
