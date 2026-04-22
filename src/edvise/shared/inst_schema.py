"""
Shared ``--inst_schema`` → project config class mapping for pipeline scripts.
"""

from __future__ import annotations

import argparse

from edvise.configs.es import ESProjectConfig
from edvise.configs.pdp import PDPProjectConfig

CONFIG_SCHEMA_BY_KEY: dict[str, type] = {
    "pdp": PDPProjectConfig,
    "es": ESProjectConfig,
}


def project_config_schema(inst_schema: str) -> type:
    key = inst_schema.strip().lower()
    if key not in CONFIG_SCHEMA_BY_KEY:
        raise ValueError(
            f"Unknown inst_schema {inst_schema!r}; "
            f"expected one of {sorted(CONFIG_SCHEMA_BY_KEY)}"
        )
    return CONFIG_SCHEMA_BY_KEY[key]


def add_inst_schema_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--inst_schema",
        type=str,
        choices=sorted(CONFIG_SCHEMA_BY_KEY),
        default="pdp",
        help="Which project config class to use when parsing config.toml (pdp or es).",
    )
