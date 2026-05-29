"""Tests for :mod:`edvise.genai.mapping.shared.silver_run_paths`."""

from __future__ import annotations

from pathlib import Path

from edvise.genai.mapping.shared.silver_run_paths import sma_pipeline_input_root


def test_sma_pipeline_input_root() -> None:
    genai = Path("/Volumes/cat/uni_silver/silver_volume/genai_mapping")
    assert sma_pipeline_input_root(genai, mode="onboard", run_id="r1") == Path(
        "/Volumes/cat/uni_silver/silver_volume/genai_mapping/runs/onboard/r1/pipeline_input"
    )
    assert sma_pipeline_input_root(genai, mode="execute", run_id="x") == Path(
        "/Volumes/cat/uni_silver/silver_volume/genai_mapping/runs/execute/x/pipeline_input"
    )
