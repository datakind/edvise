"""Unity volume path segments for GenAI mapping runs on institution silver."""

from __future__ import annotations

from pathlib import Path


def sma_pipeline_input_root(
    genai_mapping_root: Path | str, *, mode: str, run_id: str
) -> Path:
    """
    Directory for SMA **materialized pipeline outputs** (cohort/course parquet).

    Layout: ``{genai_mapping}/runs/{mode}/{run_id}/pipeline_input/`` — sibling of
    ``schema_mapping_agent/`` under the same run id.

    JSON manifests, HITL, and maps stay under
    ``{genai_mapping}/runs/{mode}/{run_id}/schema_mapping_agent/``.
    """
    root = Path(genai_mapping_root)
    m = str(mode).strip()
    r = str(run_id).strip()
    return root / "runs" / m / r / "pipeline_input"
