"""
Edvise Schema (ES) project configuration.

Currently subclasses :class:`~edvise.configs.pdp.PDPProjectConfig` so ES pipeline
steps share the same TOML shape until ES-specific fields are required.
"""

from edvise.configs.pdp import PDPProjectConfig


class ESProjectConfig(PDPProjectConfig):
    """ES pipeline config; extend when ES diverges from PDP."""
