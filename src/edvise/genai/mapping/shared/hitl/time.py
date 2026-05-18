"""Time helpers for HITL audit trails."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string (same shape as ``datetime.now(timezone.utc).isoformat()``)."""
    return datetime.now(timezone.utc).isoformat()
