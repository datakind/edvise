# modeling/training_types.py (or similar)
from __future__ import annotations
import typing as t
from typing import Protocol, runtime_checkable


@runtime_checkable
class _ExperimentType(Protocol):
    experiment_id: str


@runtime_checkable
class _BestTrialType(Protocol):
    mlflow_run_id: str


@runtime_checkable
class AutoMLSummaryType(Protocol):
    experiment: _ExperimentType
    best_trial: _BestTrialType
    # Keep this flexible; adjust if you know the exact shape
    metric_distribution: t.Mapping[str, float] | t.Any
