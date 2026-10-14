import pytest
from unittest.mock import MagicMock, patch
from edvise.reporting.sections.registry import SectionRegistry
from edvise.reporting.sections.pdp import (
    bias_sections as pdp_bias_sections,
)


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.run_id = "run123"
    card.format.indent_level = lambda level: "  " * level
    card.format.friendly_case = lambda s: s.replace("_", " ").title()
    return card


@patch(
    "edvise.reporting.sections.pdp.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=["ethnicity", "first_gen", "gender", "race", "student_age"],
)
def test_pdp_bias_groups_lists_assessed_with_labels(mock_groups, mock_card):
    registry = SectionRegistry()
    pdp_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    bias_groups_output = rendered["bias_groups_section"]

    assert "Ethnicity" in bias_groups_output
    assert "First Generation Status" in bias_groups_output
    assert "Gender" in bias_groups_output
    assert "Race" in bias_groups_output
    assert "Student Age" in bias_groups_output
    assert "bias_summary_section" in rendered
    mock_groups.assert_called_with("run123")


@patch(
    "edvise.reporting.sections.pdp.bias_sections.utils.get_assessed_bias_group_cols",
    return_value=["gender", "race"],
)
def test_pdp_bias_groups_omits_unassessed(mock_groups, mock_card):
    registry = SectionRegistry()
    pdp_bias_sections.register_bias_sections(mock_card, registry)

    rendered = registry.render_all()
    bias_groups_output = rendered["bias_groups_section"]

    assert "Gender" in bias_groups_output
    assert "Race" in bias_groups_output
    assert "Ethnicity" not in bias_groups_output
    assert "First Generation Status" not in bias_groups_output
    assert "Student Age" not in bias_groups_output
