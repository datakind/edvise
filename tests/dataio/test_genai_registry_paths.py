"""Tests for :mod:`edvise.dataio.genai_registry_paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.dataio import genai_registry_paths as grp


def test_resolve_genai_pipeline_input_dir_prefers_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text(
        '{"onboard_run_id": "ob1", "execute_run_id": "ex1"}\n'
    )
    onboard_dir = silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    execute_dir = silver / "genai_mapping" / "runs" / "execute" / "ex1" / "pipeline_input"
    onboard_dir.mkdir(parents=True)
    execute_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")
    (execute_dir / "cohort.parquet").write_text("execute")

    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    resolved = grp.resolve_genai_pipeline_input_dir(str(silver))
    assert resolved == str(execute_dir)


def test_resolve_genai_pipeline_input_dir_falls_back_to_onboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text(
        '{"onboard_run_id": "ob1", "execute_run_id": "missing_ex"}\n'
    )
    onboard_dir = silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    onboard_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")

    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    resolved = grp.resolve_genai_pipeline_input_dir(str(silver))
    assert resolved == str(onboard_dir)


def test_resolve_genai_pipeline_input_dir_onboard_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text('{"onboard_run_id": "ob1"}\n')
    onboard_dir = silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    onboard_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")

    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    resolved = grp.resolve_genai_pipeline_input_dir(str(silver))
    assert resolved == str(onboard_dir)
