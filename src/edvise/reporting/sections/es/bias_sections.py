from .. import (
    bias_sections as base_bias_sections,
)


def register_bias_sections(card, registry):
    base_bias_sections.register_bias_sections(card, registry)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        Returns bias groups for Edvise models, derived from configured
        student_group_cols.
        """
        intro = f"{card.format.indent_level(1)}- Our assessment for FNR Parity was conducted across the following student groups.\n"
        group_cols = card.cfg.student_group_cols or []
        groups = [card.format.friendly_case(col) for col in group_cols]
        nested = [f"{card.format.indent_level(2)}- {group}\n" for group in groups]
        return intro + "".join(nested)
