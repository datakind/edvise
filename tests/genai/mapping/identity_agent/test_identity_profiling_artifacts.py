"""Tests for :func:`write_identity_profiling_artifacts`."""

from __future__ import annotations

import json

from edvise.genai.mapping.identity_agent.grain_inference.run_by_dataset import (
    identity_profiling_run_to_jsonable,
    write_identity_profiling_artifacts,
)


def test_identity_profiling_run_to_jsonable_empty():
    d = identity_profiling_run_to_jsonable("test_inst", {})
    assert d == {"institution_id": "test_inst", "datasets": {}}


def test_write_identity_profiling_artifacts_empty_round_trip(tmp_path):
    path = write_identity_profiling_artifacts(tmp_path, "test_inst", {})
    assert path.name == "identity_profiling_run.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["institution_id"] == "test_inst"
    assert loaded["datasets"] == {}
