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


def test_filename_match_score_prefers_closer_name_over_shared_tokens() -> None:
    """Dates/generics drop out, but extra content like Advising must not tie Learner."""
    keyword = "Datakind - Learner Report_20260910_142024.csv"
    learner = "Datakind - Learner Report_20260916_095850.csv"
    advising = "Datakind - Student Advising_20260916_110000.csv"

    learner_score = filename_match_score(keyword, learner, dataset_key="raw_student")
    advising_score = filename_match_score(keyword, advising, dataset_key="raw_student")

    assert learner_score is not None
    assert advising_score is not None
    assert learner_score > advising_score


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
    assert (
        filename_match_score(
            "2025-09-19_CCC Student File.csv",
            "1782516108693_2026_01_20_Edvise Learner Report.csv",
            dataset_key="student",
        )
        == 100
    )


def test_filename_match_score_tokenizes_compound_dataset_key() -> None:
    assert (
        filename_match_score(
            "2025-09-19_CCC Student File.csv",
            "1782516108693_2026_01_20_Edvise Learner Report.csv",
            dataset_key="raw_student",
        )
        == 100
    )


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


def test_filename_match_score_allows_one_edit_on_long_tokens() -> None:
    keyword = "Datakind - Learner Report_20260910_142024.csv"
    exact = "Datakind - Learner Report_20260916_095850.csv"
    typo = "Datakind - Learnr Report_20260916_095850.csv"
    swapped = "Studnet File.csv"

    exact_score = filename_match_score(keyword, exact)
    typo_score = filename_match_score(keyword, typo)
    assert exact_score == 202
    assert typo_score == 152
    assert exact_score > typo_score
    assert filename_match_score("Student File.csv", swapped) == 151


def test_filename_match_score_rejects_short_token_and_unrelated_edits() -> None:
    assert filename_match_score("term file.csv", "team file.csv") is None
    assert (
        filename_match_score(
            "Datakind - Learner Report.csv",
            "Datakind - Course Report.csv",
            dataset_key="student",
        )
        is None
    )
