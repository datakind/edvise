"""Cross-agent HITL utilities (gate checks, shared exceptions)."""

from __future__ import annotations

from .exceptions import HITLBlockingError
from .gate import raise_if_hitl_pending

__all__ = [
    "HITLBlockingError",
    "raise_if_hitl_pending",
]
