"""Tests for batch dataset path resolution under gcs_uploads/{batch_id}/."""

from __future__ import annotations

import os
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


def test_resolve_dataset_file_in_batch_dir_file_kind_suffix(tmp_path: Path) -> None:
    student = tmp_path / "1782516108693_2026_01_20_Edvise Student File.csv"
    course = tmp_path / "1782516108691_2026_01_20_Edvise Course File.csv"
    semester = tmp_path / "1782516108692_2026_01_20_Edvise Semester File.csv"
    student.write_text("a\n", encoding="utf-8")
    course.write_text("b\n", encoding="utf-8")
    semester.write_text("c\n", encoding="utf-8")

    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "2025-09-19_CCC Student File.csv",
        dataset_key="student",
    ) == str(student)
    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "2025-09-19_CCC Course File.csv",
        dataset_key="course",
    ) == str(course)
    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "2025-09-19_CCC Semester File.csv",
        dataset_key="semester",
    ) == str(semester)


def test_resolve_dataset_file_in_batch_dir_timestamped_reports(tmp_path: Path) -> None:
    learner = tmp_path / "Datakind - Learner Report_20260916_095850.csv"
    course = tmp_path / "Datakind - Course Report_20260916_110342.csv"
    learner.write_text("student\n", encoding="utf-8")
    course.write_text("course\n", encoding="utf-8")

    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "Datakind - Learner Report_20260910_142024.csv",
        dataset_key="student",
    ) == str(learner)
    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "Datakind - Course Report_20260910_143123.csv",
        dataset_key="course",
    ) == str(course)


def test_resolve_dataset_file_in_batch_dir_uses_newest_best_match(
    tmp_path: Path,
) -> None:
    older = tmp_path / "Datakind - Learner Report_20260910_142024.csv"
    newer = tmp_path / "Datakind - Learner Report_20260916_095850.csv"
    older.write_text("old\n", encoding="utf-8")
    newer.write_text("new\n", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "Datakind - Learner Report_20260901_120000.csv",
        dataset_key="student",
    ) == str(newer)
