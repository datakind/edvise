"""Rate helpers shared by Step 2a and Step 2b scoring."""

from __future__ import annotations


def safe_div(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or (precision + recall) == 0:
        return None
    return round(2 * precision * recall / (precision + recall), 3)
