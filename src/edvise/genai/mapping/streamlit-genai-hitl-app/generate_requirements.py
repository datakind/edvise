#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tomllib

DEPENDENCY_GROUP = "streamlit-genai-hitl-app"


def find_repo_root(start_path: Path) -> Path:
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError("Could not find pyproject.toml in this repository.")


def load_dependency_group(pyproject_path: Path, group_name: str) -> list[str]:
    pyproject = tomllib.loads(pyproject_path.read_text())
    tool_config = pyproject.get("tool", {})
    edvise_config = tool_config.get("edvise", {})
    requirements = edvise_config.get("requirements", {})
    dependencies = requirements.get(group_name)
    if not isinstance(dependencies, list):
        raise KeyError(
            f"Requirements list '{group_name}' is missing from pyproject.toml."
        )
    return [str(dependency) for dependency in dependencies]


def _edvise_requirement_line(*, app_dir: Path, repo_root: Path) -> str:
    """
    Prefer a wheel under ``./wheels/`` (CI / Databricks bundle); else editable install
    of the parent repo for local ``streamlit run``.
    """
    wheels = sorted(app_dir.glob("wheels/edvise-*.whl"))
    if wheels:
        rel = wheels[0].relative_to(app_dir).as_posix()
        return f"edvise @ file:./{rel}"

    rel_repo = Path(os.path.relpath(repo_root.resolve(), app_dir.resolve())).as_posix()
    return f"-e {rel_repo}"


def write_requirements(output_path: Path, lines: list[str]) -> None:
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate requirements.txt for the GenAI HITL Streamlit app."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. Defaults to requirements.txt in the app directory.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    app_dir = script_path.parent
    repo_root = find_repo_root(app_dir)
    pyproject_path = repo_root / "pyproject.toml"
    output_path = args.output.resolve() if args.output else app_dir / "requirements.txt"

    dependencies = load_dependency_group(pyproject_path, DEPENDENCY_GROUP)
    edvise_line = _edvise_requirement_line(app_dir=app_dir, repo_root=repo_root)
    write_requirements(output_path, [*dependencies, edvise_line])

    print(output_path)
    print(output_path.read_text(), end="")


if __name__ == "__main__":
    main()
