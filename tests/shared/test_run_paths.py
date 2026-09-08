"""Tests for silver training/inference run path layout."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from edvise.shared.logger import (
    archive_previous_inference_runs,
    inference_archive_root,
    inference_current_root,
    resolve_run_path,
)


def _inf_args(run_id: str) -> argparse.Namespace:
    return argparse.Namespace(job_type="inference", db_run_id=run_id)


def _cfg(model_run_id: str) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(run_id=model_run_id))


def test_inference_current_and_archive_roots() -> None:
    silver = "/Volumes/cat/inst_silver/silver_volume"
    assert inference_current_root(silver, "model-1") == (
        "/Volumes/cat/inst_silver/silver_volume/model-1/inf/current"
    )
    assert inference_archive_root(silver, "model-1") == (
        "/Volumes/cat/inst_silver/silver_volume/model-1/inf/archive"
    )


def test_resolve_run_path_training() -> None:
    args = argparse.Namespace(job_type="training", db_run_id="train-9")
    assert resolve_run_path(args, SimpleNamespace(), "/silver") == (
        "/silver/train-9/training"
    )


def test_resolve_run_path_inference_uses_current_run_id(tmp_path) -> None:
    silver = str(tmp_path)
    path = resolve_run_path(_inf_args("inf-2"), _cfg("model-1"), silver)
    assert path == f"{silver}/model-1/inf/current/inf-2"


def test_resolve_run_path_inference_requires_ids() -> None:
    with pytest.raises(ValueError, match="cfg.model.run_id"):
        resolve_run_path(_inf_args("inf-1"), SimpleNamespace(model=None), "/s")
    with pytest.raises(ValueError, match="db_run_id"):
        resolve_run_path(
            argparse.Namespace(job_type="inference", db_run_id=None),
            _cfg("model-1"),
            "/s",
        )


def test_archive_copies_previous_current_run_and_keeps_new(tmp_path) -> None:
    silver = str(tmp_path)
    model_id = "model-1"
    old = tmp_path / "model-1" / "inf" / "current" / "inf-old"
    old.mkdir(parents=True)
    (old / "features_with_most_impact.parquet").write_text("old")

    archived = archive_previous_inference_runs(silver, model_id, "inf-new")

    assert archived == ["inf-old"]
    dest = tmp_path / "model-1" / "inf" / "archive" / "inf-old"
    assert (dest / "features_with_most_impact.parquet").read_text() == "old"
    assert not old.exists()


def test_resolve_run_path_archives_stale_current_then_points_at_new(tmp_path) -> None:
    silver = str(tmp_path)
    old = tmp_path / "model-1" / "inf" / "current" / "inf-old"
    old.mkdir(parents=True)
    (old / "support_overview.parquet").write_text("prev")

    path = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)

    assert path == f"{silver}/model-1/inf/current/inf-new"
    assert (
        tmp_path
        / "model-1"
        / "inf"
        / "archive"
        / "inf-old"
        / "support_overview.parquet"
    ).read_text() == "prev"
    assert not old.exists()
    # same run again must not archive itself
    path2 = resolve_run_path(_inf_args("inf-new"), _cfg("model-1"), silver)
    assert path2 == path
    assert not (tmp_path / "model-1" / "inf" / "archive" / "inf-new").exists()


def test_archive_replaces_existing_archive_folder(tmp_path) -> None:
    silver = str(tmp_path)
    current = tmp_path / "model-1" / "inf" / "current" / "inf-old"
    current.mkdir(parents=True)
    (current / "a.parquet").write_text("new-copy")
    dest = tmp_path / "model-1" / "inf" / "archive" / "inf-old"
    dest.mkdir(parents=True)
    (dest / "stale.parquet").write_text("stale")

    archive_previous_inference_runs(silver, "model-1", "inf-new")

    assert (dest / "a.parquet").read_text() == "new-copy"
    assert not (dest / "stale.parquet").exists()
