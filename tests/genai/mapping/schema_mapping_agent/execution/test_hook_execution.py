"""
hook_required plans stay execution gaps until a materialized hook_spec is attached; once
attached, execute_transformation_map dynamically imports the materialized module and runs the
named function instead of skipping the field.

Regression coverage for the gap identified in PR #155 (refactor: simplify and refine SMA
transformation HITL): materialize_hook_specs_to_file wrote transform_hooks.py, but nothing ever
attached the materialized function back onto its plan, and the executor had no loader for it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.genai.mapping.schema_mapping_agent.execution.field_executor import (
    ExecutionError,
    execute_transformation_map,
)
from edvise.genai.mapping.schema_mapping_agent.execution.step_dispatcher import (
    ExecutionGapError,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    RowSelectionConfig,
    RowSelectionStrategy,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
    FieldTransformationPlan,
    TransformationMap,
)
from edvise.genai.mapping.shared.hitl.hook_spec.schemas import HookFunctionSpec, HookSpec

HOOK_MODULE_RELPATH = "transform_hooks.py"

PREFIX_HOOK_DRAFT = '''def transform_course_course_prefix(s: "pd.Series") -> "pd.Series":
    import pandas as pd
    import re

    def extract_alpha_prefix(code):
        if pd.isna(code) or code == "":
            return None
        match = re.match(r"^([A-Za-z]+)", str(code).strip())
        return match.group(1).upper() if match else None

    return s.apply(extract_alpha_prefix).astype("string")
'''


def _manifest(course_number_source: str = "course_number") -> FieldMappingManifest:
    return FieldMappingManifest(
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        mappings=[
            FieldMappingRecord(
                target_field="course_id",
                source_column="course_id",
                source_table="course",
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=1.0,
                rationale="",
            ),
            FieldMappingRecord(
                target_field="course_prefix",
                source_column=course_number_source,
                source_table="course",
                row_selection=RowSelectionConfig(strategy=RowSelectionStrategy.any_row),
                confidence=0.6,
                rationale="",
            ),
        ],
        column_aliases=[],
    )


def _dataframes() -> dict[str, pd.DataFrame]:
    return {
        "course": pd.DataFrame(
            {
                "course_id": [1, 2, 3],
                "course_number": ["ENC1101", "CGS1060C", "MAC1105"],
            }
        )
    }


def _materialize(tmp_path: Path, draft: str) -> None:
    (tmp_path / HOOK_MODULE_RELPATH).write_text(draft, encoding="utf-8")


def test_hook_required_without_hook_spec_is_still_a_gap(tmp_path: Path) -> None:
    """Preserves prior behavior: hook_required alone (no materialized pointer) is a gap."""
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(
                target_field="course_prefix",
                steps=[],
                hook_required=True,
                reviewer_notes="needs alpha/digit split",
            ),
        ],
    )
    out = execute_transformation_map(
        tm,
        _manifest(),
        _dataframes(),
        RawEdviseCourseDataSchema,
        institution_id="test_inst",
        hook_modules_root=tmp_path,
    )
    assert out.gaps == ["course_prefix"]
    assert "course_prefix" not in out.df.columns


def test_hook_required_without_hook_spec_raises_on_gap(tmp_path: Path) -> None:
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(target_field="course_prefix", steps=[], hook_required=True),
        ],
    )
    with pytest.raises(ExecutionGapError):
        execute_transformation_map(
            tm,
            _manifest(),
            _dataframes(),
            RawEdviseCourseDataSchema,
            institution_id="test_inst",
            raise_on_gap=True,
            hook_modules_root=tmp_path,
        )


def test_hook_required_with_attached_hook_spec_runs_materialized_function(
    tmp_path: Path,
) -> None:
    """The exact ENC1101 -> ENC scenario: a materialized hook_spec makes the field executable."""
    _materialize(tmp_path, PREFIX_HOOK_DRAFT)
    hook_spec = HookSpec(
        file=HOOK_MODULE_RELPATH,
        functions=[
            HookFunctionSpec(
                name="transform_course_course_prefix",
                description="Extract leading alpha prefix from a combined course code.",
                draft=PREFIX_HOOK_DRAFT,
            )
        ],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(
                target_field="course_prefix",
                steps=[],
                hook_required=True,
                hook_spec=hook_spec,
                reviewer_notes="needs alpha/digit split",
            ),
        ],
    )
    out = execute_transformation_map(
        tm,
        _manifest(),
        _dataframes(),
        RawEdviseCourseDataSchema,
        institution_id="test_inst",
        hook_modules_root=tmp_path,
    )
    assert out.gaps == []
    assert "course_prefix" in out.executed
    assert out.df["course_prefix"].tolist() == ["ENC", "CGS", "MAC"]


def test_hook_required_missing_hook_modules_root_is_a_gap_even_with_hook_spec(
    tmp_path: Path,
) -> None:
    """Without hook_modules_root, hook_spec is never consulted — same as omitting it."""
    hook_spec = HookSpec(
        file=HOOK_MODULE_RELPATH,
        functions=[
            HookFunctionSpec(
                name="transform_course_course_prefix",
                description="x",
                draft=PREFIX_HOOK_DRAFT,
            )
        ],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(
                target_field="course_prefix",
                steps=[],
                hook_required=True,
                hook_spec=hook_spec,
            ),
        ],
    )
    out = execute_transformation_map(
        tm,
        _manifest(),
        _dataframes(),
        RawEdviseCourseDataSchema,
        institution_id="test_inst",
        # hook_modules_root omitted
    )
    assert out.gaps == ["course_prefix"]


def test_hook_missing_on_disk_raises_execution_error(tmp_path: Path) -> None:
    """hook_spec points at a module that was never materialized — a real bug, not a soft gap."""
    hook_spec = HookSpec(
        file=HOOK_MODULE_RELPATH,
        functions=[HookFunctionSpec(name="transform_course_course_prefix", description="x")],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(
                target_field="course_prefix",
                steps=[],
                hook_required=True,
                hook_spec=hook_spec,
            ),
        ],
    )
    with pytest.raises(ExecutionError, match="not found"):
        execute_transformation_map(
            tm,
            _manifest(),
            _dataframes(),
            RawEdviseCourseDataSchema,
            institution_id="test_inst",
            hook_modules_root=tmp_path,
        )


def test_hook_returning_non_series_raises_execution_error(tmp_path: Path) -> None:
    bad_draft = (
        'def transform_course_course_prefix(s: "pd.Series"):\n'
        "    return list(s)\n"
    )
    _materialize(tmp_path, bad_draft)
    hook_spec = HookSpec(
        file=HOOK_MODULE_RELPATH,
        functions=[
            HookFunctionSpec(
                name="transform_course_course_prefix", description="x", draft=bad_draft
            )
        ],
    )
    tm = TransformationMap(
        institution_id="test_inst",
        entity_type="course",
        target_schema="RawEdviseCourseDataSchema",
        plans=[
            FieldTransformationPlan(target_field="course_id", steps=[]),
            FieldTransformationPlan(
                target_field="course_prefix",
                steps=[],
                hook_required=True,
                hook_spec=hook_spec,
            ),
        ],
    )
    with pytest.raises(ExecutionError, match="must return a pandas Series"):
        execute_transformation_map(
            tm,
            _manifest(),
            _dataframes(),
            RawEdviseCourseDataSchema,
            institution_id="test_inst",
            hook_modules_root=tmp_path,
        )
