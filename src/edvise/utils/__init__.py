"""Edvise utility submodules.

Submodules are loaded lazily so lightweight imports (e.g.
``edvise.utils.institution_naming`` for the GenAI HITL Streamlit app) do not
pull cluster runtime dependencies such as ``databricks-connect`` or ``mlflow``.
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_SUBMODULES = frozenset(
    {
        "api_requests",
        "automate_releases",
        "data_cleaning",
        "databricks",
        "drop_columns_safely",
        "infer_data_terms",
        "types",
        "update_config",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(_LAZY_SUBMODULES)
