"""Tests for :mod:`edvise.genai.mapping.shared.grain.dedup_execution`."""

from __future__ import annotations

import logging

import pandas as pd

from edvise.genai.mapping.shared.grain.dedup_execution import apply_sma_grain_resolution_payload


def test_true_duplicate_dedupes_on_entity_keys_not_full_row() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["A", "A"],
            "extra": [10, 20],
        }
    )
    keys = ["sid", "term"]
    payload = {"grain_resolution": {"dedup_strategy": "true_duplicate"}}
    out = apply_sma_grain_resolution_payload(df, keys, payload, log=logging.getLogger("test"))
    assert len(out) == 1
    assert list(out.columns) == list(df.columns)


def test_temporal_collapse_sort_then_keep_first() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1, 1],
            "term": ["A", "A", "A"],
            "seq": [3, 1, 2],
        }
    )
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "temporal_collapse",
            "dedup_sort_by": "seq",
            "dedup_sort_ascending": True,
            "dedup_keep": "first",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 1
    assert int(out["seq"].iloc[0]) == 1


def test_categorical_priority_keeps_best_row() -> None:
    df = pd.DataFrame(
        {
            "sid": [1, 1],
            "term": ["A", "A"],
            "honors": ["Cum Laude", "Summa Cum Laude"],
        }
    )
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "categorical_priority",
            "priority_column": "honors",
            "priority_order": ["Summa Cum Laude", "Magna Cum Laude", "Cum Laude"],
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 1
    assert "Summa" in str(out["honors"].iloc[0])


def test_suffix_identifier_requires_suffix_in_entity_keys() -> None:
    df = pd.DataFrame({"sid": [1, 1], "course": ["X", "X"], "term": ["A", "A"]})
    keys = ["sid", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "suffix_identifier",
            "suffix_column": "course",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    # course not in entity_keys — skip mutation
    assert len(out) == 2


def test_suffix_identifier_applies_when_suffix_in_keys() -> None:
    df = pd.DataFrame({"sid": [1, 1], "course": ["X", "X"], "term": ["A", "A"]})
    keys = ["sid", "course", "term"]
    payload = {
        "grain_resolution": {
            "dedup_strategy": "suffix_identifier",
            "suffix_column": "course",
        }
    }
    out = apply_sma_grain_resolution_payload(df, keys, payload)
    assert len(out) == 2
    vals = set(out["course"].astype(str))
    assert any("-" in v for v in vals)


def test_no_dedup_returns_copy() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    out = apply_sma_grain_resolution_payload(
        df, ["a"], {"grain_resolution": {"dedup_strategy": "no_dedup"}}
    )
    assert len(out) == 2
    assert out is not df
