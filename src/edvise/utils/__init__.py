"""Edvise utility submodules.

``update_config`` is loaded lazily (see :func:`__getattr__`) so imports like
``edvise.utils.databricks`` work on Databricks clusters that omit optional deps
(e.g. ``tomlkit`` used only for TOML editing).
"""

from __future__ import annotations

import importlib
from typing import Any

from . import (
    api_requests,
    automate_releases,
    data_cleaning,
    databricks,
    drop_columns_safely,
    infer_data_terms,
    types,
)


def __getattr__(name: str) -> Any:
    if name == "update_config":
        return importlib.import_module(f"{__name__}.update_config")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
