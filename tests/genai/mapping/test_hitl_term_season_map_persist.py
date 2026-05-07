"""Persistence helpers for term ``season_map_replace`` (Streamlit bundle + core schema)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_HITL_APP = _REPO / "src/edvise/genai/mapping/streamlit-genai-hitl-app"
if str(_HITL_APP) not in sys.path:
    sys.path.insert(0, str(_HITL_APP))

from hitl_reviewer.persistence.hitl_json_batch_commit import (
    _validated_season_map_replace_from_dataframe,
)
from hitl_reviewer.persistence.silver_hitl_paths import (
    merge_season_map_replace_into_selected_option,
)


def test_validated_season_map_replace_from_dataframe_ok() -> None:
    df = pd.DataFrame([{"raw": "01", "canonical": "spring"}])
    rows, err = _validated_season_map_replace_from_dataframe(df)
    assert err is None
    assert rows == [{"raw": "01", "canonical": "SPRING"}]


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
