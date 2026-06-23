import pytest
from unittest.mock import MagicMock

from edvise.configs.es import ESProjectConfig
from edvise.reporting.sections.es import register_sections
from edvise.reporting.sections.registry import SectionRegistry


def make_es_config() -> ESProjectConfig:
    return ESProjectConfig(
        institution_id="inst_id",
        institution_name="Inst Name",
        model={"experiment_id": "exp123", "run_id": "abc"},
        datasets={
            "raw_course": "dummy.csv",
            "raw_cohort": "dummy.csv",
        },
        preprocessing={
            "selection": {"student_criteria": {"status": "active"}},
            "checkpoint": {
                "name": "checkpoint_nth",
                "type_": "nth",
                "n": 4,
            },
            "target": {
                "name": "retention",
                "type_": "retention",
                "max_academic_year": "2025-26",
            },
            "features": {},
        },
        modeling={
            "feature_selection": {
                "collinear_threshold": 10.0,
                "low_variance_threshold": 0.0,
                "incomplete_threshold": 0.5,
            },
            "training": {"primary_metric": "logloss", "timeout_minutes": 10},
        },
        split_col=None,
    )


def test_bias_groups_section_uses_student_group_cols():
    card = MagicMock()
    card.cfg = make_es_config()
    card.format.friendly_case.side_effect = lambda s: s.replace("_", " ").title()
    card.format.indent_level.side_effect = lambda n: "  " * n

    registry = SectionRegistry()
    register_sections(card, registry)
    sections = registry.render_all()

    bias_section = sections["bias_groups_section"]
    assert "Learner Age" in bias_section
    assert "First Generation Status" in bias_section
    assert "Student Age" not in bias_section
