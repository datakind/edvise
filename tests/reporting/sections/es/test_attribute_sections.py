import pytest
from unittest.mock import MagicMock

from edvise.reporting.sections.es import (
    attribute_sections as es_attribute_sections,
)
from edvise.reporting.sections.registry import SectionRegistry
from edvise.reporting.utils.formatting import Formatting


@pytest.fixture
def mock_card():
    card = MagicMock()
    card.format = Formatting()
    # Stub sibling sections so render_all() can exercise outcome only.
    card.cfg.preprocessing.checkpoint.type_ = "first"
    card.cfg.preprocessing.selection.student_criteria = {}
    return card


@pytest.mark.parametrize(
    "outcome_type, time_limits, extra_config, expected_snippet",
    [
        (
            "retention",
            {},
            {},
            "This model predicts the likelihood of non-retention into the student's second academic year based on student, course, and academic data.",
        ),
        (
            "graduation",
            {"FULL-TIME": (2.0, "year"), "PART-TIME": (3.0, "year")},
            {},
            "This model predicts the likelihood of not graduating on time within 2 years for full-time students, and within 3 years for part-time students, based on student, course, and academic data.",
        ),
        (
            "credits_earned",
            {"FULL-TIME": (1.5, "year")},
            {"min_num_credits": 45},
            "This model predicts the likelihood of not earning 45 credits within 1.5 years for full-time students, based on student, course, and academic data.",
        ),
    ],
)
def test_outcome_variants(
    mock_card, outcome_type, time_limits, extra_config, expected_snippet
):
    mock_card.cfg.preprocessing.target.type_ = outcome_type
    mock_card.cfg.preprocessing.selection.intensity_time_limits = time_limits
    mock_card.cfg.preprocessing.target.intensity_time_limits = None

    if outcome_type == "credits_earned" and "min_num_credits" in extra_config:
        mock_card.cfg.preprocessing.target.min_num_credits = extra_config[
            "min_num_credits"
        ]

    registry = SectionRegistry()
    es_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert expected_snippet in rendered["outcome_section"]


def test_outcome_falls_back_to_target_intensity_time_limits(mock_card):
    """Selection intensity_time_limits is optional; use target when omitted."""
    mock_card.cfg.preprocessing.target.type_ = "graduation"
    mock_card.cfg.preprocessing.selection.intensity_time_limits = None
    mock_card.cfg.preprocessing.target.intensity_time_limits = {
        "FULL-TIME": (2.0, "year"),
        "PART-TIME": (3.0, "year"),
    }

    registry = SectionRegistry()
    es_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert (
        "not graduating on time within 2 years for full-time students, "
        "and within 3 years for part-time students" in rendered["outcome_section"]
    )


def test_outcome_missing_intensity_time_limits(mock_card):
    mock_card.cfg.preprocessing.target.type_ = "graduation"
    mock_card.cfg.preprocessing.selection.intensity_time_limits = None
    mock_card.cfg.preprocessing.target.intensity_time_limits = None

    registry = SectionRegistry()
    es_attribute_sections.register_attribute_sections(mock_card, registry)
    rendered = registry.render_all()

    assert "Timeframe for Outcome Variable Not Found" in rendered["outcome_section"]
