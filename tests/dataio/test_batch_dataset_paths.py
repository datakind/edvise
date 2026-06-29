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


def test_extract_file_kind_suffix() -> None:
    assert m._extract_file_kind_suffix("2025-09-19_CCC Student File.csv") == (
        "student file"
    )
    assert m._extract_file_kind_suffix("fixture_students.csv") is None
