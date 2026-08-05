import pytest

from edvise.ingestion.nsc_sftp.file_selection import (
    classify_pdp_file_role,
    discover_file_pairs,
    extract_file_stamp,
    select_file_pair,
)


def _row(name: str, size: int = 10) -> dict:
    return {
        "source_system": "NSC",
        "sftp_path": "./receive",
        "file_name": name,
        "file_size": size,
        "file_modified_time": None,
    }


def test_extract_file_stamp():
    assert extract_file_stamp("PDP_Cohort_File_20240115123045.csv") == "20240115123045"


def test_classify_pdp_file_role():
    assert classify_pdp_file_role("School_Cohort_20240115123045.csv") == "cohort"
    assert classify_pdp_file_role("School_Course_20240115123045.csv") == "course"
    assert classify_pdp_file_role("readme.txt") is None


def test_discover_file_pairs_requires_both_roles():
    rows = [
        _row("A_Cohort_20240115123045.csv"),
        _row("A_Course_20240115123045.csv"),
        _row("B_Cohort_20240201101010.csv"),  # incomplete pair
        _row("noise_20240301111111.csv"),
    ]
    pairs = discover_file_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0].stamp == "20240115123045"
    assert pairs[0].cohort_file_name.endswith("Cohort_20240115123045.csv")
    assert pairs[0].course_file_name.endswith("Course_20240115123045.csv")


def test_select_file_pair_manual():
    cohort = "A_Cohort_20240115123045.csv"
    course = "A_Course_20240115123045.csv"
    c, o, mode = select_file_pair(
        [],
        mode="uningested",
        cohort_file_name=cohort,
        course_file_name=course,
    )
    assert (c, o, mode) == (cohort, course, "manual")


def test_select_file_pair_latest():
    rows = [
        _row("A_Cohort_20240115123045.csv"),
        _row("A_Course_20240115123045.csv"),
        _row("B_Cohort_20240201101010.csv"),
        _row("B_Course_20240201101010.csv"),
    ]
    c, o, mode = select_file_pair(rows, mode="latest")
    assert mode == "latest"
    assert c == "B_Cohort_20240201101010.csv"
    assert o == "B_Course_20240201101010.csv"


def test_select_file_pair_uningested_skips_bronze_written():
    rows = [
        _row("A_Cohort_20240115123045.csv"),
        _row("A_Course_20240115123045.csv"),
        _row("B_Cohort_20240201101010.csv"),
        _row("B_Course_20240201101010.csv"),
    ]
    fingerprint_by_name = {
        "B_Cohort_20240201101010.csv": "fp_b_cohort",
        "B_Course_20240201101010.csv": "fp_b_course",
        "A_Cohort_20240115123045.csv": "fp_a_cohort",
        "A_Course_20240115123045.csv": "fp_a_course",
    }
    status_by_fingerprint = {
        "fp_b_cohort": "BRONZE_WRITTEN",
        "fp_b_course": "BRONZE_WRITTEN",
    }
    c, o, mode = select_file_pair(
        rows,
        mode="uningested",
        fingerprint_by_name=fingerprint_by_name,
        status_by_fingerprint=status_by_fingerprint,
    )
    assert mode == "uningested"
    assert c == "A_Cohort_20240115123045.csv"
    assert o == "A_Course_20240115123045.csv"


def test_select_file_pair_uningested_all_done_raises():
    rows = [
        _row("A_Cohort_20240115123045.csv"),
        _row("A_Course_20240115123045.csv"),
    ]
    fingerprint_by_name = {
        "A_Cohort_20240115123045.csv": "fp_a_cohort",
        "A_Course_20240115123045.csv": "fp_a_course",
    }
    status_by_fingerprint = {
        "fp_a_cohort": "BRONZE_WRITTEN",
        "fp_a_course": "BRONZE_WRITTEN",
    }
    with pytest.raises(FileNotFoundError, match="already BRONZE_WRITTEN"):
        select_file_pair(
            rows,
            mode="uningested",
            fingerprint_by_name=fingerprint_by_name,
            status_by_fingerprint=status_by_fingerprint,
        )
