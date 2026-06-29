"""Tests for batch dataset path resolution under gcs_uploads/{batch_id}/."""

from __future__ import annotations

from pathlib import Path

from edvise.dataio import batch_dataset_paths as m


def test_resolve_dataset_file_in_batch_dir_exact_and_substring(tmp_path: Path) -> None:
    (tmp_path / "cohort.csv").write_text("x", encoding="utf-8")
    (tmp_path / "other_course.csv").write_text("y", encoding="utf-8")

    assert m.resolve_dataset_file_in_batch_dir(str(tmp_path), "cohort.csv") == str(
        tmp_path / "cohort.csv"
    )
    assert m.resolve_dataset_file_in_batch_dir(str(tmp_path), "course") == str(
        tmp_path / "other_course.csv"
    )
