# ruff: noqa: F821
# mypy: ignore-errors
"""Shared regex patterns and Field definitions for Edvise raw schemas (student + course)."""

import re

try:
    import pandera as pda
except ModuleNotFoundError:
    import edvise.utils as utils

    utils.databricks.mock_pandera()
    import pandera as pda

# Year format YYYY-YY (cohort_year, academic_year)
YEAR_PATTERN = re.compile(r"^\d{4}-\d{2}$")

# Term name, e.g. Fall, Fall 2023, SP (cohort_term, academic_term)
TERM_PATTERN = re.compile(
    r"(?i)^(\d{4})?\s?(Fall|Winter|Spring|Summer|FA|WI|SP|SU|SM)\s?(\d{4})?$"
)

# Credential/degree type (student + course optional fields)
CREDENTIAL_DEGREE_PATTERN = re.compile(
    r"(?i).*(bachelor|ba|bs|associate|aa|as|aas|certificate|certification).*"
)

StudentIdField = pda.Field(nullable=False, str_length={"min_value": 1})
