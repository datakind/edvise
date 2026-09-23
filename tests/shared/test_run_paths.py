"""Tests for silver training/inference run path layout."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from edvise.shared.logger import _archive_prior_inference_run, resolve_run_path


def _inf_args(run_id: str | None) -> argparse.Namespace:
    return argparse.Namespace(job_type="inference", db_run_id=run_id)


def _cfg(model_run_id: str) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(run_id=model_run_id))


def test_resolve_run_path_matches_develop_layout(tmp_path) -> None:
    silver = str(tmp_path)
    assert (
        resolve_run_path(
            argparse.Namespace(job_type="training", db_run_id="train-9"),
            SimpleNamespace(),
            silver,
        )
        == f"{silver}/train-9/training"
    )
    assert resolve_run_path(_inf_args("inf-2"), _cfg("model-1"), silver) == (
        f"{silver}/model-1/inference"
    )
    assert resolve_run_path(_inf_args(None), _cfg("model-1"), silver) == (
        f"{silver}/model-1/inference"
    )


def test_resolve_run_path_inference_requires_model_run_id() -> None:
    with pytest.raises(ValueError, match="cfg.model.run_id"):
        resolve_run_path(_inf_args("inf-1"), SimpleNamespace(model=None), "/s")


def test_resolve_run_path_archives_prior_inference_files(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"
    inference.mkdir(parents=True)
    (inference / "student_terms.parquet").write_text("old")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("new")
    path2 = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    path3 = resolve_run_path(_inf_args("inf-newer"), _cfg("model-1"), silver)

    assert path == path2 == path3 == str(inference)
    assert (inference / "archive" / "student_terms.parquet").read_text() == "old"
    assert (
        inference / "archive" / "inf-new" / "student_terms.parquet"
    ).read_text() == ("new")
    assert not (inference / "student_terms.parquet").exists()


def test_latest_run_stays_in_inference_until_next_job(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"

    resolve_run_path(_inf_args("run-1"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("first")

    assert not (inference / "archive" / "run-1").exists()
    assert (inference / "student_terms.parquet").read_text() == "first"

    resolve_run_path(_inf_args("run-2"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("second")

    assert (inference / "archive" / "run-1" / "student_terms.parquet").read_text() == (
        "first"
    )
    assert (inference / "student_terms.parquet").read_text() == "second"
    assert not (inference / "archive" / "run-2").exists()


def test_inference_archive_does_not_move_config_toml(tmp_path) -> None:
    silver = str(tmp_path)
    run_root = tmp_path / "model-1"
    training = run_root / "training"
    inference = run_root / "inference"
    training.mkdir(parents=True)
    inference.mkdir(parents=True)
    (training / "config.toml").write_text("trained")
    (inference / "config.toml").write_text("do-not-move")
    (inference / "student_terms.parquet").write_text("old")

    resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert (training / "config.toml").read_text() == "trained"
    assert (inference / "config.toml").read_text() == "do-not-move"
    assert not (inference / "student_terms.parquet").exists()
    assert (inference / "archive" / "student_terms.parquet").read_text() == "old"
    assert not (inference / "archive" / "config.toml").exists()


def test_archive_refuses_training_directory(tmp_path) -> None:
    training = tmp_path / "model-1" / "training"
    training.mkdir(parents=True)
    (training / "config.toml").write_text("trained")

    with pytest.raises(ValueError, match="Training config must stay in training/"):
        _archive_prior_inference_run(str(training), "inf-new")

    assert (training / "config.toml").read_text() == "trained"
