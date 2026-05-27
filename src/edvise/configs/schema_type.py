"""Map ``--schema_type`` CLI values to PDP vs Edvise project config classes."""

from __future__ import annotations

import typing as t

from edvise.configs.es import ESProjectConfig
from edvise.configs.pdp import PDPProjectConfig


def normalize_schema_type(raw: str) -> str:
    return raw.strip().lower()


def project_config_class(
    schema_type: str,
) -> t.Type[PDPProjectConfig | ESProjectConfig]:
    s = normalize_schema_type(schema_type)
    if s == "pdp":
        return PDPProjectConfig
    if s in ("edvise", "es"):
        return ESProjectConfig
    raise ValueError(
        f"Unknown --schema_type {schema_type!r}; expected 'pdp', 'edvise', or 'es'."
    )


def is_edvise_schema(schema_type: str) -> bool:
    """True when ``schema_type`` selects :class:`ESProjectConfig` (``edvise`` or ``es``)."""
    return normalize_schema_type(schema_type) in ("edvise", "es")
