from .. import (
    bias_sections as base_bias_sections,
)


def register_bias_sections(card, registry):
    # Use base bias_groups_section (lists groups from MLflow bias_test artifacts).
    base_bias_sections.register_bias_sections(card, registry)
