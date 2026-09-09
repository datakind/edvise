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


def test_resolve_run_path_inference_stays_in_existing_folder(tmp_path) -> None:
    silver = str(tmp_path)
    path = resolve_run_path(_inf_args("inf-2"), _cfg("model-1"), silver)
    assert path == f"{silver}/model-1/inference"


def test_resolve_run_path_inference_without_db_run_id_keeps_legacy_folder() -> None:
    path = resolve_run_path(_inf_args(None), _cfg("model-1"), "/silver")
    assert path == "/silver/model-1/inference"


def test_resolve_run_path_inference_requires_model_run_id() -> None:
    with pytest.raises(ValueError, match="cfg.model.run_id"):
        resolve_run_path(_inf_args("inf-1"), SimpleNamespace(model=None), "/s")


def test_resolve_run_path_archives_existing_inference_files(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"
    inference.mkdir(parents=True)
    (inference / "student_terms.parquet").write_text("old")
    (inference / "preprocessed.parquet").write_text("old")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert path == str(inference)
    archived = inference / "archive" / "legacy"
    assert (archived / "student_terms.parquet").read_text() == "old"
    assert (archived / "preprocessed.parquet").read_text() == "old"
    assert not (inference / "student_terms.parquet").exists()
    assert (inference / "run_id").read_text() == "inf-new"


def test_resolve_run_path_same_job_does_not_rearchive(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"
    resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("new")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert path == str(inference)
    assert (inference / "student_terms.parquet").read_text() == "new"
    assert not (inference / "archive").exists()


def test_resolve_run_path_archives_previous_run_then_keeps_new_in_inference(
    tmp_path,
) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"
    inference.mkdir(parents=True)
    (inference / "student_terms.parquet").write_text("keep")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert path == str(inference)
    assert (inference / "archive" / "legacy" / "student_terms.parquet").read_text() == (
        "keep"
    )
    assert not (inference / "student_terms.parquet").exists()
    path2 = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    assert path2 == path
    (inference / "student_terms.parquet").write_text("new")
    path3 = resolve_run_path(_inf_args("inf-newer"), _cfg("model-1"), silver)
    assert path3 == path
    assert (
        inference / "archive" / "inf-new" / "student_terms.parquet"
    ).read_text() == ("new")
    assert not (inference / "current").exists()
