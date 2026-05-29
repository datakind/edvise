"""Heuristic season_map seeds for the Streamlit IA term reviewer."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_HITL_APP = _REPO / "src/edvise/genai/mapping/streamlit-genai-hitl-app"
if str(_HITL_APP) not in sys.path:
    sys.path.insert(0, str(_HITL_APP))

from hitl_reviewer.ui.ia.term_season_map_guess import (
    build_season_map_seed_dataframe,
    guess_month_code_rows_from_hitl_text,
)


def test_guess_month_codes_from_quoted_tokens() -> None:
    rows = guess_month_code_rows_from_hitl_text(
        "Confirm mapping",
        "month codes '01', '05', '08' observed",
    )
    assert {r["raw"] for r in rows} == {"01", "05", "08"}
    assert all(r["canonical"] in ("SPRING", "SUMMER", "FALL") for r in rows)


def test_build_seed_prefers_json_then_guess() -> None:
    df = build_season_map_seed_dataframe(
        resolution_season_map_replace=[],
        hitl_question="q",
        hitl_context="sample 201308.0",
    )
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    raws = set(df["raw"].astype(str).str.strip())
    assert "08" in raws
