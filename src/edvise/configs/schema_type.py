"""Map ``--schema_type`` CLI values to project config classes."""

from __future__ import annotations

import typing as t

from edvise.configs.es import ESProjectConfig
from edvise.configs.legacy import LegacyProjectConfig
from edvise.configs.pdp import PDPProjectConfig

ProjectConfig = PDPProjectConfig | ESProjectConfig | LegacyProjectConfig

_DEFAULT_FEATURES_TABLE_PATH = "shared/assets/pdp_features_table.toml"


def normalize_schema_type(raw: str) -> str:
    return raw.strip().lower()


def project_config_class(schema_type: str) -> t.Type[ProjectConfig]:
    s = normalize_schema_type(schema_type)
    if s == "pdp":
        return PDPProjectConfig
    if s in ("edvise", "es"):
        return ESProjectConfig
    if s == "legacy":
        return LegacyProjectConfig
    raise ValueError(
        f"Unknown --schema_type {schema_type!r}; "
        "expected 'pdp', 'edvise', 'es', or 'legacy'."
    )


def is_edvise_schema(schema_type: str) -> bool:
    """True when ``schema_type`` selects :class:`ESProjectConfig` (``edvise`` or ``es``)."""
    return normalize_schema_type(schema_type) in ("edvise", "es")


def is_legacy_schema(schema_type: str) -> bool:
    """True when ``schema_type`` selects :class:`LegacyProjectConfig`."""
    return normalize_schema_type(schema_type) == "legacy"


def resolve_features_table_path(
    schema_type: str,
    features_table_path: str | None,
) -> str:
    """Return the features-table TOML path for the given schema type."""
    if features_table_path:
        return features_table_path
    if is_legacy_schema(schema_type):
        raise ValueError("--features_table_path required when --schema_type=legacy")
    return _DEFAULT_FEATURES_TABLE_PATH
