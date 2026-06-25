"""Tests for :mod:`edvise.dataio.genai_registry_paths`."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.dataio import genai_registry_paths as grp


def _layout_with_both_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text(
        '{"onboard_run_id": "ob1", "execute_run_id": "ex1"}\n'
    )
    onboard_dir = (
        silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    )
    execute_dir = (
        silver / "genai_mapping" / "runs" / "execute" / "ex1" / "pipeline_input"
    )
    onboard_dir.mkdir(parents=True)
    execute_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")
    (execute_dir / "cohort.parquet").write_text("execute")
    return silver, onboard_dir, execute_dir


def test_resolve_genai_pipeline_input_dir_training_uses_onboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver, onboard_dir, _execute_dir = _layout_with_both_runs(tmp_path)
    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    resolved = grp.resolve_genai_pipeline_input_dir(str(silver), job_type="training")
    assert resolved == str(onboard_dir)


def test_resolve_genai_pipeline_input_dir_inference_uses_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver, _onboard_dir, execute_dir = _layout_with_both_runs(tmp_path)
    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    resolved = grp.resolve_genai_pipeline_input_dir(str(silver), job_type="inference")
    assert resolved == str(execute_dir)


def test_resolve_genai_pipeline_input_dir_inference_requires_execute_run_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text('{"onboard_run_id": "ob1"}\n')
    onboard_dir = (
        silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    )
    onboard_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")

    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    with pytest.raises(FileNotFoundError, match="no execute_run_id"):
        grp.resolve_genai_pipeline_input_dir(str(silver), job_type="inference")


def test_resolve_genai_pipeline_input_dir_inference_requires_execute_pipeline_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    silver = tmp_path / "silver"
    active = silver / "genai_mapping" / "active"
    active.mkdir(parents=True)
    (active / "genai_active_registry.json").write_text(
        '{"onboard_run_id": "ob1", "execute_run_id": "missing_ex"}\n'
    )
    onboard_dir = (
        silver / "genai_mapping" / "runs" / "onboard" / "ob1" / "pipeline_input"
    )
    onboard_dir.mkdir(parents=True)
    (onboard_dir / "cohort.parquet").write_text("onboard")

    monkeypatch.setattr(grp, "path_exists", lambda p: Path(p).exists())

    with pytest.raises(FileNotFoundError, match="Run genai mapping execute"):
        grp.resolve_genai_pipeline_input_dir(str(silver), job_type="inference")
