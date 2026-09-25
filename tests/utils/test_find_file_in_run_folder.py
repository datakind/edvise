"""Inference must read the training config, never an inference/ copy."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("google.auth")

from edvise.utils.databricks import find_file_in_run_folder


def _layout(tmp_path: Path) -> Path:
    run_root = tmp_path / "run-1"
    (run_root / "training").mkdir(parents=True)
    (run_root / "inference").mkdir()
    (run_root / "training" / "config.toml").write_text("trained\n", encoding="utf-8")
    (run_root / "inference" / "config.toml").write_text("stale\n", encoding="utf-8")
    return run_root


def test_prefers_training_config_over_inference_copy(tmp_path: Path) -> None:
    run_root = _layout(tmp_path)
    found = find_file_in_run_folder(str(run_root), keyword="config")
    assert found.endswith("/training/config.toml")
    assert "inference" not in found


def test_ignores_config_that_exists_only_under_inference(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    (run_root / "inference").mkdir(parents=True)
    (run_root / "inference" / "config.toml").write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Did not search inference/"):
        find_file_in_run_folder(str(run_root), keyword="config")


def test_finds_config_at_run_root_when_training_has_none(tmp_path: Path) -> None:
    run_root = tmp_path / "run-1"
    run_root.mkdir()
    (run_root / "inference").mkdir()
    (run_root / "config.toml").write_text("legacy\n", encoding="utf-8")
    (run_root / "inference" / "config.toml").write_text("stale\n", encoding="utf-8")

    found = find_file_in_run_folder(str(run_root))
    assert found.endswith("/run-1/config.toml")
    assert not found.endswith("/inference/config.toml")
