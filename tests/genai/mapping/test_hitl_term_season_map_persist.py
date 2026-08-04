"""Persistence helpers for term ``season_map_replace`` (Streamlit bundle + core schema)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_HITL_APP = _REPO / "src/edvise/genai/mapping/streamlit-genai-hitl-app"
if str(_HITL_APP) not in sys.path:
    sys.path.insert(0, str(_HITL_APP))

from hitl_reviewer.persistence.silver_hitl_paths import (
    merge_season_map_replace_into_selected_option,
    season_map_rows_chronology_error,
    sort_season_map_dataframe_chronologically,
    validated_season_map_replace_from_dataframe,
)


def test_validated_season_map_replace_from_dataframe_ok() -> None:
    df = pd.DataFrame([{"raw": "01", "canonical": "spring"}])
    rows, err = validated_season_map_replace_from_dataframe(df)
    assert err is None
    assert rows == [{"raw": "01", "canonical": "SPRING"}]


def test_validated_season_map_replace_from_dataframe_flags_out_of_order() -> None:
    df = pd.DataFrame(
        [
            {"raw": "15", "canonical": "FALL"},
            {"raw": "20", "canonical": "FALL"},
            {"raw": "25", "canonical": "SPRING"},
        ]
    )
    rows, err = validated_season_map_replace_from_dataframe(df)
    assert rows is None
    assert err is not None
    assert "calendar-chronological" in err
    assert "FALL (row 2) precedes SPRING (row 3)" in err


def test_season_map_rows_chronology_error_ok_when_sorted() -> None:
    rows = [
        {"raw": "25", "canonical": "SPRING"},
        {"raw": "35", "canonical": "SUMMER"},
        {"raw": "15", "canonical": "FALL"},
    ]
    assert season_map_rows_chronology_error(rows) is None


def test_season_map_rows_chronology_error_allows_duplicates() -> None:
    rows = [
        {"raw": "25", "canonical": "SPRING"},
        {"raw": "30", "canonical": "SPRING"},
        {"raw": "35", "canonical": "SUMMER"},
    ]
    assert season_map_rows_chronology_error(rows) is None


def test_sort_season_map_dataframe_chronologically_reorders_and_is_stable() -> None:
    df = pd.DataFrame(
        [
            {"raw": "15", "canonical": "FALL"},
            {"raw": "20", "canonical": "FALL"},
            {"raw": "25", "canonical": "SPRING"},
            {"raw": "30", "canonical": "SPRING"},
            {"raw": "35", "canonical": "SUMMER"},
            {"raw": "40", "canonical": "SUMMER"},
        ]
    )
    sorted_df = sort_season_map_dataframe_chronologically(df)
    assert sorted_df["raw"].tolist() == ["25", "30", "35", "40", "15", "20"]
    assert sorted_df["canonical"].tolist() == [
        "SPRING",
        "SPRING",
        "SUMMER",
        "SUMMER",
        "FALL",
        "FALL",
    ]
    assert season_map_rows_chronology_error(sorted_df.to_dict("records")) is None


def test_sort_season_map_dataframe_chronologically_sends_blank_rows_last() -> None:
    df = pd.DataFrame(
        [
            {"raw": "15", "canonical": "FALL"},
            {"raw": "", "canonical": ""},
            {"raw": "25", "canonical": "SPRING"},
        ]
    )
    sorted_df = sort_season_map_dataframe_chronologically(df)
    assert sorted_df["raw"].tolist() == ["25", "15", ""]


def test_sort_season_map_dataframe_chronologically_empty_df() -> None:
    df = pd.DataFrame(columns=["raw", "canonical"])
    sorted_df = sort_season_map_dataframe_chronologically(df)
    assert sorted_df.empty


def test_merge_season_map_replace_into_selected_option() -> None:
    data = {
        "items": [
            {
                "options": [
                    {
                        "option_id": "c",
                        "resolution": {"season_map_replace": []},
                        "reentry": "generate_hook",
                    },
                ],
            },
        ],
    }
    merge_season_map_replace_into_selected_option(
        data,
        0,
        1,
        [{"raw": "08", "canonical": "FALL"}],
    )
    assert data["items"][0]["options"][0]["resolution"]["season_map_replace"] == [
        {"raw": "08", "canonical": "FALL"},
    ]
