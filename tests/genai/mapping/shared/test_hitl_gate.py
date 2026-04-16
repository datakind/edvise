"""Tests for :mod:`edvise.genai.mapping.shared.hitl`."""

from __future__ import annotations

from pathlib import Path

import pytest

from edvise.genai.mapping.shared.hitl import HITLBlockingError, raise_if_hitl_pending


def test_raise_if_hitl_pending_noop_when_empty(tmp_path: Path) -> None:
    raise_if_hitl_pending(
        [],
        hitl_path=tmp_path / "hitl.json",
        format_item=lambda x: str(x),
        instructions="  • do thing\n",
    )


def test_raise_if_hitl_pending_raises(tmp_path: Path) -> None:
    p = tmp_path / "grain_hitl.json"
    with pytest.raises(HITLBlockingError) as excinfo:
        raise_if_hitl_pending(
            ["a"],
            hitl_path=p,
            format_item=lambda x: f"[{x}] blocked",
            instructions="  • fix\n",
        )
    msg = str(excinfo.value)
    assert "1 unreviewed" in msg
    assert "grain_hitl.json" in msg
    assert "[a] blocked" in msg
