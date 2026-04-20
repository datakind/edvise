"""Read / write Pydantic models as JSON files (indent=2, UTF-8)."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_pydantic_json(path: Path, model: type[T]) -> T:
    """Load ``path`` and validate as ``model``. Raises FileNotFoundError if missing."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return model.model_validate_json(path.read_text())


def write_pydantic_json(
    path: Path, model: BaseModel, *, indent: int | None = 2
) -> None:
    """Serialize ``model`` to ``path`` (parent directories are not created)."""
    path.write_text(model.model_dump_json(indent=indent))
