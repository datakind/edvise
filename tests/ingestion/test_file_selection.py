import pytest

from edvise.ingestion.nsc_sftp.file_selection import (
    classify_pdp_file_role,
    discover_file_pairs,
    extract_file_stamp,
    select_file_pairs,
)

COHORT_A = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20240115123045.csv"
COURSE_A = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20240115123045.csv"
COHORT_B = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20240201101010.csv"
COURSE_B = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20240201101010.csv"
COHORT_C = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20260724030759.csv"
COURSE_C = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20260724030759.csv"
COHORT_D = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20260724040738.csv"
COURSE_D = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20260724040738.csv"
# Same calendar day, three deliveries (matches production AO1600 pattern).
COHORT_E1 = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20260824090803.csv"
COURSE_E1 = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20260824090803.csv"
COHORT_E2 = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20260824090817.csv"
COURSE_E2 = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20260824090817.csv"
COHORT_E3 = "AO1600pdp_AO1600_AR_DEIDENTIFIED_STUDYID_20260824090841.csv"
COURSE_E3 = "AO1600pdp_AO1600_COURSE_LEVEL_AR_DEIDENTIFIED_STUDYID_20260824090841.csv"


def _row(name: str, size: int = 10) -> dict:
    return {
        "source_system": "NSC",
        "sftp_path": "./receive",
        "file_name": name,
        "file_size": size,
        "file_modified_time": None,
    }


def test_extract_file_stamp():
    assert extract_file_stamp(COHORT_A) == "20240115123045"


def test_classify_pdp_file_role():
    assert classify_pdp_file_role(COHORT_C) == "cohort"
    assert classify_pdp_file_role(COURSE_C) == "course"
    assert classify_pdp_file_role("readme.txt") is None


def test_discover_file_pairs_requires_both_roles():
    rows = [
        _row(COHORT_A),
        _row(COURSE_A),
        _row(COHORT_B),  # incomplete pair
        _row("noise_20240301111111.csv"),
    ]
    pairs = discover_file_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0].stamp == "20240115123045"
    assert pairs[0].cohort_file_name == COHORT_A
    assert pairs[0].course_file_name == COURSE_A


def test_discover_file_pairs_and_latest_same_day_returns_all():
    rows = [
        _row(COHORT_C),
        _row(COURSE_C),
        _row(COHORT_D),
        _row(COURSE_D),
        _row(COHORT_E1),
        _row(COURSE_E1),
        _row(COHORT_E2),
        _row(COURSE_E2),
        _row(COHORT_E3),
        _row(COURSE_E3),
    ]
    pairs = discover_file_pairs(rows)
    assert [p.stamp for p in pairs] == [
        "20260724030759",
        "20260724040738",
        "20260824090803",
        "20260824090817",
        "20260824090841",
    ]
    chosen, mode = select_file_pairs(rows, mode="latest")
    assert mode == "latest"
    assert [p.stamp for p in chosen] == [
        "20260824090803",
        "20260824090817",
        "20260824090841",
    ]
    assert {p.date for p in chosen} == {"20260824"}


def test_select_file_pairs_manual():
    chosen, mode = select_file_pairs(
        [],
        mode="skip_ingested",
        cohort_file_name=COHORT_A,
        course_file_name=COURSE_A,
    )
    assert mode == "manual"
    assert len(chosen) == 1
    assert chosen[0].cohort_file_name == COHORT_A
    assert chosen[0].course_file_name == COURSE_A


def test_select_file_pairs_skip_ingested_same_day_pending():
    rows = [
        _row(COHORT_E1),
        _row(COURSE_E1),
        _row(COHORT_E2),
        _row(COURSE_E2),
        _row(COHORT_E3),
        _row(COURSE_E3),
    ]
    # One pair already bronze; remaining same-day pairs should both be selected.
    chosen, mode = select_file_pairs(
        rows,
        mode="skip_ingested",
        ingested_file_names={COHORT_E1, COURSE_E1},
    )
    assert mode == "skip_ingested"
    assert [p.stamp for p in chosen] == ["20260824090817", "20260824090841"]


def test_select_file_pairs_skip_ingested_prefers_newer_date():
    rows = [
        _row(COHORT_A),
        _row(COURSE_A),
        _row(COHORT_B),
        _row(COURSE_B),
    ]
    chosen, mode = select_file_pairs(
        rows,
        mode="skip_ingested",
        ingested_file_names={COHORT_B, COURSE_B},
    )
    assert mode == "skip_ingested"
    assert [p.stamp for p in chosen] == ["20240115123045"]


def test_select_file_pairs_skip_ingested_all_done_raises():
    rows = [_row(COHORT_A), _row(COURSE_A)]
    with pytest.raises(FileNotFoundError, match="already BRONZE_WRITTEN"):
        select_file_pairs(
            rows,
            mode="skip_ingested",
            ingested_file_names={COHORT_A, COURSE_A},
        )
