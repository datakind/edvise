"""
Run school-specific legacy preprocessing before ``training_h2o`` or ``inference_h2o``.

Loads ``preprocessing.py`` from the Databricks workspace under the SSI service principal’s
fixed ``student-success-intervention/pipelines`` tree (see ``SSI_PIPELINES_WORKSPACE_ROOT``).

**Training:** ``--config_file_path`` is typically the institution config on the bronze volume
(training inputs). **Inference:** run after ``legacy_inference_inputs`` so
``--config_file_path`` is the same config snapshot used for training (resolved from the
model run under the silver volume). In both cases, ``--run_type`` is forwarded to the
school ``run()`` (``train`` vs ``predict``).

Resolution under ``<root>/<databricks_institution_name>/``:

- If ``--legacy_preprocessing_model_subdir`` is set: ``<subdir>/preprocessing.py``.
- Else: ``preprocessing.py`` at the institution root, or exactly one ``*/preprocessing.py``.

Institution **data** paths in ``config.toml`` still use UC volume URIs
(e.g. ``/Volumes/<catalog>/..._bronze/bronze_volume/...``).

Disable preprocessing with job parameter ``legacy_preprocessing_enabled`` set to ``false``
when modeling tables are produced elsewhere.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)

# SSI repo ``pipelines/`` directory for the pipeline job SP (institution = next path segment).
SSI_PIPELINES_WORKSPACE_ROOT = (
    "/Workspace/Users/6c8d8d76-1399-4065-aeb5-9474d32773cf/"
    "student-success-intervention/pipelines"
)


def _normalize_fs_path(raw: str) -> Path:
    p = raw.strip()
    if p.startswith("dbfs:/Workspace"):
        p = "/Workspace" + p[len("dbfs:/Workspace") :]
    elif p.startswith("dbfs:/"):
        p = "/dbfs/" + p[6:].lstrip("/")
    return Path(p)


def resolve_workspace_preprocessing_py(
    institution_id: str,
    model_subdir: str | None,
    *,
    workspace_root: str = SSI_PIPELINES_WORKSPACE_ROOT,
) -> tuple[Path, Path]:
    """
    Returns ``(preprocessing_py, institution_pipeline_dir)`` where
    ``institution_pipeline_dir`` is ``.../pipelines/<institution_id>/``.

    ``workspace_root`` defaults to ``SSI_PIPELINES_WORKSPACE_ROOT``; override for tests only.
    """
    root = _normalize_fs_path(workspace_root)
    inst = institution_id.strip()
    if not inst:
        raise ValueError("institution_id / databricks_institution_name must be non-empty.")
    base = root / inst

    sub = (model_subdir or "").strip()
    if sub:
        candidate = base / sub / "preprocessing.py"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Expected preprocessing at {candidate} (legacy_preprocessing_model_subdir={sub!r})."
            )
        return candidate, base

    direct = base / "preprocessing.py"
    if direct.is_file():
        return direct, base

    nested = sorted(p for p in base.glob("*/preprocessing.py") if p.is_file())
    if len(nested) == 1:
        return nested[0], base
    if len(nested) > 1:
        raise FileNotFoundError(
            "Multiple preprocessing.py files under "
            f"{base}: {nested}. Set --legacy_preprocessing_model_subdir to disambiguate."
        )
    raise FileNotFoundError(
        f"No preprocessing.py under {base} (tried {direct} and one subfolder)."
    )


def _ensure_edvise_src_on_path() -> None:
    """
    Workspace ``preprocessing.py`` files often ``import edvise`` like the Git entry scripts.
    Those scripts prepend repo ``src/`` to ``sys.path``; dynamic loads do not inherit that
    unless we add it here (``cwd`` may not be the Git checkout root on Databricks).

    Databricks sometimes runs ``legacy_preprocessing`` via ``exec(compile(...))``, which
    leaves ``__file__`` undefined; fall back to ``inspect.getfile`` for this module path.
    """
    try:
        script_path = Path(__file__).resolve()
    except NameError:
        script_path = Path(inspect.getfile(_ensure_edvise_src_on_path)).resolve()
    src_dir = script_path.parents[2]
    s = str(src_dir)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_module_from_file(py_file: Path, institution_pipeline_dir: Path):
    """
    Load ``preprocessing.py``; same-folder imports work. If the file lives in a model
    subfolder, also put the institution directory on ``sys.path`` and set
    ``__package__`` so ``from .helpers import ...`` resolves.
    """
    _ensure_edvise_src_on_path()
    # SSI clone root (parent of ``.../student-success-intervention/pipelines``) for
    # ``import pipelines.<inst>.<model>.helpers`` — same as each file’s ``parents[3]`` trick.
    ssi_root = str(_normalize_fs_path(SSI_PIPELINES_WORKSPACE_ROOT).parent)
    if ssi_root not in sys.path:
        sys.path.insert(0, ssi_root)
    inst_s = str(institution_pipeline_dir.resolve())
    model_dir = py_file.resolve().parent
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))
    if model_dir.resolve() != institution_pipeline_dir.resolve():
        if inst_s not in sys.path:
            sys.path.insert(0, inst_s)

    mod_name = "_edvise_legacy_preprocessing_workspace"
    spec = importlib.util.spec_from_file_location(mod_name, py_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not build import spec for {py_file}")
    mod = importlib.util.module_from_spec(spec)
    if model_dir.resolve() != institution_pipeline_dir.resolve():
        mod.__package__ = model_dir.name
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy H2O preprocessing via SSI workspace preprocessing.py."
    )
    parser.add_argument(
        "--config_file_path",
        required=True,
        help="Path to institution config.toml (same as training_h2o).",
    )
    parser.add_argument(
        "--run_type",
        default="train",
        choices=("train", "predict"),
        help="Forwarded to the school preprocessing module.",
    )
    parser.add_argument(
        "--databricks_institution_name",
        default="",
        help="Institution folder under pipelines/ (e.g. john_jay_col). Required when preprocessing runs.",
    )
    parser.add_argument(
        "--legacy_preprocessing_model_subdir",
        default="",
        help=(
            "Optional subfolder under pipelines/<inst>/ containing preprocessing.py "
            "(e.g. transfer_model). Empty = preprocessing.py at institution root or exactly one */preprocessing.py."
        ),
    )
    parser.add_argument(
        "--legacy_preprocessing_enabled",
        default="true",
        help='If "false", exit without running (for schools with pre-built tables).',
    )
    args = parser.parse_args()

    enabled = str(args.legacy_preprocessing_enabled).strip().lower()
    if enabled in ("false", "0", "no", "off"):
        LOGGER.info(
            "legacy_preprocessing_enabled=%s — skipping preprocessing.",
            args.legacy_preprocessing_enabled,
        )
        return

    inst = (args.databricks_institution_name or "").strip()
    if not inst:
        raise SystemExit(
            "--databricks_institution_name is required when legacy_preprocessing runs."
        )
    sub = (args.legacy_preprocessing_model_subdir or "").strip() or None
    LOGGER.info(
        "Resolving preprocessing under workspace root %s for institution %r",
        SSI_PIPELINES_WORKSPACE_ROOT,
        inst,
    )
    py_file, inst_dir = resolve_workspace_preprocessing_py(inst, sub)
    LOGGER.info("Loading preprocessing from workspace file: %s", py_file)
    mod = load_module_from_file(py_file, inst_dir)
    label = str(py_file)

    run = getattr(mod, "run", None)
    if run is None:
        raise RuntimeError(f"{label!r} must define a run() function.")
    LOGGER.info(
        "Running %s.run(config_file_path=%r, run_type=%r)",
        label,
        args.config_file_path,
        args.run_type,
    )
    run(config_file_path=args.config_file_path, run_type=args.run_type)


if __name__ == "__main__":
    main()
