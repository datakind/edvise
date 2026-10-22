from .. import (
    bias_sections as base_bias_sections,
)

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
    def group_label_fn(col: str) -> str:
        return str(_PDP_GROUP_LABELS.get(col, card.format.friendly_case(col)))

    base_bias_sections.register_bias_sections(
        card, registry, group_label_fn=group_label_fn
    )
