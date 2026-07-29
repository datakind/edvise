"""Tests for SMA gate_2 override_2a_manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edvise.genai.mapping.schema_mapping_agent.manifest.hitl.artifacts import (
    write_sma_manifest_artifact,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
    FieldMappingRecord,
    ReviewStatus,
)
from edvise.genai.mapping.scripts.edvise_genai_sma import (
    _as_bool_flag,
    apply_gate_2_manifest_overrides,
    resolve_overrides_json_path,
)
from edvise.genai.mapping.shared.hitl.json_io import read_pydantic_json
from edvise.genai.mapping.shared.hitl.run_log import MappingOverrideLog


def _fmr(**overrides: object) -> FieldMappingRecord:
    base = {
        "target_field": "learner_id",
        "source_column": "student_id",
        "source_table": "cohort",
        "confidence": 0.4,
        "rationale": "original rationale",
        "row_selection": {"strategy": "any_row"},
    }
    base.update(overrides)
    return FieldMappingRecord.model_validate(base)


def test_as_bool_flag() -> None:
    assert _as_bool_flag(True) is True
    assert _as_bool_flag(False) is False
    assert _as_bool_flag("true") is True
    assert _as_bool_flag("TRUE") is True
    assert _as_bool_flag("false") is False
    assert _as_bool_flag("") is False
    assert _as_bool_flag("0") is False


def test_resolve_overrides_json_path_absolute_and_relative(tmp_path: Path) -> None:
    abs_file = tmp_path / "abs.json"
    abs_file.write_text("{}", encoding="utf-8")
    assert resolve_overrides_json_path(str(abs_file), run_root=tmp_path) == abs_file

    run_root = tmp_path / "schema_mapping_agent"
    run_root.mkdir()
    rel = run_root / "overrides.json"
    rel.write_text("{}", encoding="utf-8")
    assert resolve_overrides_json_path("overrides.json", run_root=run_root) == rel


def test_resolve_overrides_json_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Overrides JSON not found"):
        resolve_overrides_json_path("missing.json", run_root=tmp_path)


def test_apply_gate_2_manifest_overrides(tmp_path: Path) -> None:
    run_root = tmp_path / "schema_mapping_agent"
    run_root.mkdir()
    fm = FieldMappingManifest(
        entity_type="cohort",
        target_schema="RawEdviseStudentDataSchema",
        mappings=[
            _fmr(),
            _fmr(
                target_field="entry_term",
                source_column="term_code",
                source_table="cohort",
            ),
        ],
    )
    manifest_path = write_sma_manifest_artifact(
        run_root, fm, basename="manifest_map.json"
    )
    overrides_path = run_root / "overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "entity_type": "cohort",
                        "target_field": "learner_id",
                        "correction": {
                            "target_field": "learner_id",
                            "source_column": "student_id_v2",
                            "source_table": "cohort",
                            "confidence": 1.0,
                            "row_selection": {"strategy": "any_row"},
                        },
                    },
                    {
                        "entity_type": "cohort",
                        "target_field": "entry_term",
                        "unmap": True,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        run_root=run_root,
        manifest_map=manifest_path,
        mapping_override_log=run_root / "mapping_override_log.json",
    )

    count = apply_gate_2_manifest_overrides(
        paths,  # type: ignore[arg-type]
        "overrides.json",
        institution_id="u9",
        overridden_by="pipeline",
        original_db_run_id="db-1",
    )
    assert count == 2

    out = read_pydantic_json(manifest_path, FieldMappingManifest)
    by_field = {m.target_field: m for m in out.mappings}
    assert by_field["learner_id"].source_column == "student_id_v2"
    assert by_field["learner_id"].review_status == ReviewStatus.corrected_by_override
    assert by_field["entry_term"].source_column is None

    log = read_pydantic_json(paths.mapping_override_log, MappingOverrideLog)
    assert len(log.events) == 2


def test_apply_gate_2_manifest_overrides_missing_manifest_raises(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "sma"
    run_root.mkdir()
    (run_root / "overrides.json").write_text(
        json.dumps({"overrides": []}), encoding="utf-8"
    )
    paths = SimpleNamespace(
        run_root=run_root,
        manifest_map=run_root / "manifest_map.json",
        mapping_override_log=run_root / "mapping_override_log.json",
    )
    with pytest.raises(FileNotFoundError, match="manifest_map.json missing"):
        apply_gate_2_manifest_overrides(
            paths,  # type: ignore[arg-type]
            "overrides.json",
            institution_id="u9",
            overridden_by="pipeline",
            original_db_run_id="db-1",
        )
