from .. import (
    bias_sections as base_bias_sections,
)
from ...utils import utils

# Stable display labels for PDP group columns (friendly_case alone shortens
# first_gen → "First Gen").
_PDP_GROUP_LABELS = {
    "ethnicity": "Ethnicity",
    "first_gen": "First Generation Status",
    "gender": "Gender",
    "race": "Race",
    "student_age": "Student Age",
}


def register_bias_sections(card, registry):
    base_bias_sections.register_bias_sections(card, registry)

    @registry.register("bias_groups_section")
    def bias_groups_section():
        """
        List student groups actually bias-assessed for this PDP run, using
        PDP-specific display labels when available.
        """
        intro = (
            f"{card.format.indent_level(1)}- Our assessment for FNR Parity was "
            "conducted across the following student groups.\n"
        )
        group_cols = utils.get_assessed_bias_group_cols(card.run_id)
        if not group_cols:
            return (
                intro
                + f"{card.format.indent_level(2)}- Unable to determine assessed student groups\n"
            )
        labels = [
            _PDP_GROUP_LABELS.get(col, card.format.friendly_case(col))
            for col in group_cols
        ]
        nested = [f"{card.format.indent_level(2)}- {label}\n" for label in labels]
        return intro + "".join(nested)
