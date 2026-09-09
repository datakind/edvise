"""Tests for silver training/inference run path layout."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from edvise.shared.logger import resolve_run_path, snapshot_inference_run


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
    assert not (inference / "archive" / "legacy").exists()


def test_resolve_run_path_flattens_existing_archive_legacy(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"
    legacy = inference / "archive" / "legacy"
    legacy.mkdir(parents=True)
    (legacy / "student_terms.parquet").write_text("old")
    (inference / "run_id").write_text("inf-new")

    resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert (inference / "archive" / "student_terms.parquet").read_text() == "old"
    assert not legacy.exists()


def test_two_inference_runs_leave_two_archive_folders(tmp_path) -> None:
    silver = str(tmp_path)
    inference = tmp_path / "model-1" / "inference"

    resolve_run_path(_inf_args("run-1"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("first")
    snapshot_inference_run(str(inference), "run-1")

    resolve_run_path(_inf_args("run-2"), _cfg("model-1"), silver)
    (inference / "student_terms.parquet").write_text("second")
    snapshot_inference_run(str(inference), "run-2")

    assert (inference / "student_terms.parquet").read_text() == "second"
    assert (inference / "archive" / "run-1" / "student_terms.parquet").read_text() == (
        "first"
    )
    assert (inference / "archive" / "run-2" / "student_terms.parquet").read_text() == (
        "second"
    )
