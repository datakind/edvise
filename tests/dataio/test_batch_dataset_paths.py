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


def test_resolve_dataset_file_in_batch_dir_prefers_closer_name_over_newer(
    tmp_path: Path,
) -> None:
    """Shared tokens alone must not let a newer Advising file beat Learner Report."""
    learner = tmp_path / "Datakind - Learner Report_20260916_095850.csv"
    advising = tmp_path / "Datakind - Student Advising_20260916_110000.csv"
    learner.write_text("learner\n", encoding="utf-8")
    advising.write_text("advising\n", encoding="utf-8")
    os.utime(learner, (1, 1))
    os.utime(advising, (2, 2))

    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "Datakind - Learner Report_20260910_142024.csv",
        dataset_key="raw_student",
    ) == str(learner)


def test_resolve_dataset_file_in_batch_dir_accepts_one_edit_typo(
    tmp_path: Path,
) -> None:
    typo = tmp_path / "Datakind - Learnr Report_20260916_095850.csv"
    typo.write_text("learner\n", encoding="utf-8")

    assert m.resolve_dataset_file_in_batch_dir(
        str(tmp_path),
        "Datakind - Learner Report_20260910_142024.csv",
        dataset_key="raw_student",
    ) == str(typo)


def test_resolve_es_raw_dataset_paths_substring_match(tmp_path: Path) -> None:
    student = tmp_path / "1782424164337_2025-09-19_CCC Student File.csv"
    course = tmp_path / "1782424164335_2025-09-19_CCC Course File.csv"
    student.write_text("a\n", encoding="utf-8")
    course.write_text("b\n", encoding="utf-8")

    cohort_path, course_path = m.resolve_es_raw_dataset_paths(
        str(tmp_path),
        raw_cohort_name="2025-09-19_CCC Student File.csv",
        raw_course_name="2025-09-19_CCC Course File.csv",
    )
    assert cohort_path == str(student)
    assert course_path == str(course)
