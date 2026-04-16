"""Shared gate-check helpers for human-in-the-loop review across agents."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypeVar

from .exceptions import HITLBlockingError

T = TypeVar("T")


def raise_if_hitl_pending(
    pending: Sequence[T],
    *,
    hitl_path: Path,
    format_item: Callable[[T], str],
    instructions: str,
) -> None:
    """
    Raise :class:`HITLBlockingError` when ``pending`` is non-empty.

    Typical usage: pass ``envelope.pending`` (choice not set), or a merged list
    that also includes rows that need follow-up (e.g. incomplete direct edit).
    """
    if not pending:
        return
    n = len(pending)
    summary = "\n".join(format_item(i) for i in pending)
    # Indent each line so the block lines up with IdentityAgent gate messages.
    indented = "\n".join(f"  {line}" for line in summary.splitlines())
    raise HITLBlockingError(
        f"\n{n} unreviewed HITL item(s) blocking pipeline:\n{indented}\n\n"
        f"To resolve, edit {hitl_path.name}:\n{instructions}"
    )
