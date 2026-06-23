"""Persist ColumnRolesAgent output for audit."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from .runner import column_roles_by_dataset_to_jsonable
from .schemas import ColumnRolesResult


def write_column_roles_artifacts(
    output_dir: str | Path,
    institution_id: str,
    roles_by_dataset: Mapping[str, ColumnRolesResult],
    *,
    filename: str = "column_roles_run.json",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = column_roles_by_dataset_to_jsonable(institution_id, roles_by_dataset)
    path = output_dir / filename
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
