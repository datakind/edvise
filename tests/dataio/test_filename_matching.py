"""Tests for shared institutional extract filename matching."""

from edvise.dataio.filename_matching import (
    filename_match_score,
    filename_match_tokens,
)


def test_filename_match_tokens_drops_volatile_and_generic_tokens() -> None:
    assert filename_match_tokens(
        "Datakind - Learner Report_20260910_142024.csv"
    ) == frozenset({"datakind", "student"})


def test_filename_match_tokens_canonicalizes_dataset_aliases() -> None:
    assert filename_match_tokens("Learner Export.csv") == filename_match_tokens(
        "Student File.parquet"
    )


def test_filename_match_score_matches_timestamped_extracts_by_stable_tokens() -> None:
    assert filename_match_score(
        "Datakind - Learner Report_20260910_142024.csv",
        "Datakind - Learner Report_20260916_095850.csv",
        dataset_key="student",
    )


def test_filename_match_score_rejects_wrong_dataset_semantics() -> None:
    assert (
        filename_match_score(
            "Datakind - Learner Report_20260910_142024.csv",
            "Datakind - Course Report_20260916_095850.csv",
            dataset_key="student",
        )
        is None
    )


def test_filename_match_score_allows_dataset_alias_with_changed_prefix() -> None:
    assert filename_match_score(
        "2025-09-19_CCC Student File.csv",
        "1782516108693_2026_01_20_Edvise Learner Report.csv",
        dataset_key="student",
    ) == 100


def test_filename_match_score_tokenizes_compound_dataset_key() -> None:
    assert filename_match_score(
        "2025-09-19_CCC Student File.csv",
        "1782516108693_2026_01_20_Edvise Learner Report.csv",
        dataset_key="raw_student",
    ) == 100


def test_filename_match_score_dataset_key_requires_token_in_both_names() -> None:
    assert (
        filename_match_score(
            "financial aid.csv",
            "Edvise Student File.csv",
            dataset_key="raw_student",
        )
        is None
    )


def test_filename_match_score_matches_multiword_keyword_tokens() -> None:
    assert filename_match_score(
        "transfer advisement",
        "DEIDENTIFIED_ir_trn_transfer-advisement_Fall_26.csv",
    )
