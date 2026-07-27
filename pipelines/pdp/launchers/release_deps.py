"""Parse ``pyproject.toml`` at a release SHA into ``release_requirements.txt``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PYPROJECT_FILENAME = "pyproject.toml"
RELEASE_REQUIREMENTS_FILENAME = "release_requirements.txt"


def parse_pyproject_dependencies(text: str) -> list[str]:
    """Return ``[project].dependencies`` lines (PEP 508 strings)."""
    if sys.version_info >= (3, 11):
        try:
            import tomllib

            data = tomllib.loads(text)
        except Exception as exc:
            msg = f"Invalid pyproject.toml: {exc}"
            raise ValueError(msg) from exc
    else:
        try:
            import tomli  # type: ignore[import-not-found]

            data = tomli.loads(text)
        except Exception as exc:
            msg = f"Invalid pyproject.toml: {exc}"
            raise ValueError(msg) from exc

    if not isinstance(data, dict):
        msg = "pyproject.toml root must be a table"
        raise TypeError(msg)
    project = data.get("project")
    if not isinstance(project, dict):
        msg = "pyproject.toml missing [project] table"
        raise ValueError(msg)
    raw = project.get("dependencies")
    if raw is None:
        return []
    if not isinstance(raw, list):
        msg = "[project].dependencies must be an array"
        raise TypeError(msg)
    return [str(dep).strip() for dep in raw if str(dep).strip()]


def pyproject_path(release_dir: Path) -> Path:
    return release_dir / PYPROJECT_FILENAME


def release_requirements_path(release_dir: Path) -> Path:
    return release_dir / RELEASE_REQUIREMENTS_FILENAME


def write_release_requirements(
    pyproject_file: Path,
    out_path: Path,
    *,
    logger: logging.Logger = LOGGER,
) -> list[str]:
    """Write ``release_requirements.txt`` from a ``pyproject.toml`` file."""
    deps = parse_pyproject_dependencies(pyproject_file.read_text(encoding="utf-8"))
    lines = list(deps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    logger.info(
        "Wrote %s with %s dependencies from %s",
        out_path,
        len(lines),
        pyproject_file.name,
    )
    return deps
