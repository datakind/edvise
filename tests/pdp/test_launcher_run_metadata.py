"""Tests for parent launcher run id resolution and metadata payload."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipelines.pdp.launchers.launcher_run_metadata import (
    get_databricks_run_id,
    resolve_launcher_run_id,
)


def test_resolve_launcher_run_id_prefers_job_parameter() -> None:
    assert resolve_launcher_run_id("439619245566927") == "439619245566927"


def test_resolve_launcher_run_id_ignores_unresolved_template(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DATABRICKS_RUN_ID", "111222333")
    assert resolve_launcher_run_id("{{job.run_id}}") == "111222333"
    assert resolve_launcher_run_id("") == "111222333"


def test_resolve_launcher_run_id_none_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("DATABRICKS_RUN_ID", raising=False)
    assert resolve_launcher_run_id("") is None
    assert resolve_launcher_run_id("{{job.run_id}}") is None
    assert get_databricks_run_id() is None
