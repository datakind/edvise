"""Tests for Step 2b transformation plan vs manifest validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edvise.genai.mapping.schema_mapping_agent.transformation.validation import (
    TransformationPlanValidationErrorCode,
    is_manifest_record_unmapped,
    raise_pydantic_validation_error_if_any,
    validate_transformation_plans_against_manifest,
)


def test_is_manifest_record_unmapped() -> None:
    assert is_manifest_record_unmapped(None) is True
    assert (
        is_manifest_record_unmapped(
            {
                "target_field": "bachelors_degree_conferral_date",
                "source_column": None,
                "source_table": None,
            }
        )
        is True
    )
    assert (
        is_manifest_record_unmapped(
            {
                "target_field": "gpa",
                "source_column": "gpa_val",
                "source_table": "student",
            }
        )
        is False
    )


def test_validate_rejects_hook_required_on_unmapped() -> None:
    td = {
        "transformation_maps": {
            "cohort": {
                "plans": [
                    {
                        "target_field": "bachelors_degree_conferral_date",
                        "hook_required": True,
                        "steps": [],
                    },
                    {
                        "target_field": "gpa",
                        "hook_required": True,
                        "steps": [],
                    },
                ]
            }
        }
    }
    mm = {
        "manifests": {
            "cohort": {
                "mappings": [
                    {
                        "target_field": "bachelors_degree_conferral_date",
                        "source_column": None,
                        "source_table": None,
                    },
                    {
                        "target_field": "gpa",
                        "source_column": "gpa_val",
                        "source_table": "student",
                    },
                ]
            }
        }
    }
    errs = validate_transformation_plans_against_manifest(td, mm)
    assert len(errs) == 1
    assert errs[0].target_field == "bachelors_degree_conferral_date"
    assert (
        errs[0].error_code
        == TransformationPlanValidationErrorCode.HOOK_REQUIRED_ON_UNMAPPED
    )


def test_validate_rejects_review_required_on_unmapped() -> None:
    td = {
        "transformation_maps": {
            "course": {
                "plans": [
                    {
                        "target_field": "term_pell_recipient",
                        "review_required": True,
                        "steps": [],
                    }
                ]
            }
        }
    }
    mm = {
        "manifests": {
            "course": {
                "mappings": [
                    {
                        "target_field": "term_pell_recipient",
                        "source_column": None,
                        "source_table": None,
                    }
                ]
            }
        }
    }
    errs = validate_transformation_plans_against_manifest(td, mm)
    assert len(errs) == 1
    assert (
        errs[0].error_code
        == TransformationPlanValidationErrorCode.REVIEW_REQUIRED_ON_UNMAPPED
    )


def test_raise_pydantic_validation_error_if_any() -> None:
    td = {
        "transformation_maps": {
            "cohort": {
                "plans": [
                    {
                        "target_field": "bachelors_degree_conferral_date",
                        "hook_required": True,
                        "steps": [],
                    }
                ]
            }
        }
    }
    mm = {
        "manifests": {
            "cohort": {
                "mappings": [
                    {
                        "target_field": "bachelors_degree_conferral_date",
                        "source_column": None,
                        "source_table": None,
                    }
                ]
            }
        }
    }
    with pytest.raises(ValidationError):
        raise_pydantic_validation_error_if_any(
            validate_transformation_plans_against_manifest(td, mm)
        )
