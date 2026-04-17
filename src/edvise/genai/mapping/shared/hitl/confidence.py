"""Single default for HITL routing by model-reported confidence (IdentityAgent + SMA)."""

from __future__ import annotations

# Compared with ``<=`` in validators: at this score or lower, HITL / hitl_flag rules apply.
PIPELINE_HITL_CONFIDENCE_THRESHOLD: float = 0.7
