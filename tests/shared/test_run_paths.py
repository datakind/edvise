"""Tests for silver training/inference run path layout."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from edvise.shared.logger import resolve_run_path


def _inf_args(run_id: str | None) -> argparse.Namespace:
    return argparse.Namespace(job_type="inference", db_run_id=run_id)


def _cfg(model_run_id: str) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(run_id=model_run_id))


def test_resolve_run_path_training() -> None:
    args = argparse.Namespace(job_type="training", db_run_id="train-9")
    assert resolve_run_path(args, SimpleNamespace(), "/silver") == (
        "/silver/train-9/training"
    )


def test_resolve_run_path_inference_uses_current_run_id(tmp_path) -> None:
    silver = str(tmp_path)
    path = resolve_run_path(_inf_args("inf-2"), _cfg("model-1"), silver)
    assert path == f"{silver}/model-1/inference/current/inf-2"


def test_resolve_run_path_inference_without_db_run_id_keeps_legacy_folder() -> None:
    path = resolve_run_path(_inf_args(None), _cfg("model-1"), "/silver")
    assert path == "/silver/model-1/inference"


def test_resolve_run_path_inference_requires_model_run_id() -> None:
    with pytest.raises(ValueError, match="cfg.model.run_id"):
        resolve_run_path(_inf_args("inf-1"), SimpleNamespace(model=None), "/s")


def test_resolve_run_path_archives_stale_current_then_points_at_new(tmp_path) -> None:
    silver = str(tmp_path)
    old = tmp_path / "model-1" / "inference" / "current" / "inf-old"
    old.mkdir(parents=True)
    (old / "preprocessed.parquet").write_text("prev")
    loose = tmp_path / "model-1" / "inference" / "preprocessed.parquet"
    loose.write_text("keep")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert path == f"{silver}/model-1/inference/current/inf-new"
    assert (
        tmp_path
        / "model-1"
        / "inference"
        / "archive"
        / "inf-old"
        / "preprocessed.parquet"
    ).read_text() == "prev"
    assert not old.exists()
    assert loose.read_text() == "keep"
    path2 = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    assert path2 == path
    assert not (tmp_path / "model-1" / "inference" / "archive" / "inf-new").exists()
