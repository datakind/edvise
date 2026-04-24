"""Tests for :mod:`edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans`."""

import copy
import logging

import pytest

from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
    dedupe_plans_in_section,
    dedupe_transformation_plans_in_wrapper,
)

_LOG = logging.getLogger("test_dedupe_plans")


def _step(name: str = "cast_string", col: str = "c") -> dict:
    return {"function_name": name, "column": col, "rationale": "r"}


def test_merges_identical_duplicates_and_joins_distinct_notes() -> None:
    plans = [
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step()],
            "validation_notes": "one",
            "reviewer_notes": "r1",
        },
        {
            "target_field": "x",
            "output_dtype": "string",
            "steps": [_step()],
            "validation_notes": "two",
            "reviewer_notes": "r1",
        },
    ]
    out, n = dedupe_plans_in_section(
        copy.deepcopy(plans), entity="course", log=_LOG
    )
    assert n == 1
    assert len(out) == 1
    p = out[0]
    assert p["output_dtype"] == "string"
    assert p["steps"] == [_step()]
    # First plan's note first, then the distinct second
    assert p["validation_notes"] == "one; two"
    assert p["reviewer_notes"] == "r1"


def test_raises_on_conflicting_duplicates() -> None:
    plans = [
        {"target_field": "x", "output_dtype": "string", "steps": [_step("cast_string", "a")]},
        {"target_field": "x", "output_dtype": "string", "steps": [_step("cast_string", "b")]},
    ]
    with pytest.raises(ValueError) as e:
        dedupe_plans_in_section(plans, entity="cohort", log=_LOG)
    assert "conflicting" in str(e.value).lower() or "Duplicate" in str(e.value)


def test_dedupe_wrapper_both_entities() -> None:
    w = {
        "transformation_maps": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "S",
                "plans": [
                    {
                        "target_field": "a",
                        "output_dtype": None,
                        "steps": [],
                        "validation_notes": "m1",
                    },
                    {
                        "target_field": "a",
                        "output_dtype": None,
                        "steps": [],
                        "validation_notes": "m2",
                    },
                ],
            },
            "course": {"entity_type": "course", "target_schema": "C", "plans": []},
        }
    }
    n = dedupe_transformation_plans_in_wrapper(w, log=_LOG)
    assert n == 1
    assert len(w["transformation_maps"]["cohort"]["plans"]) == 1
    assert "m1" in w["transformation_maps"]["cohort"]["plans"][0]["validation_notes"]
    assert "m2" in w["transformation_maps"]["cohort"]["plans"][0]["validation_notes"]
