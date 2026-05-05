"""
Run school-specific legacy preprocessing before ``training_h2o`` or ``inference_h2o``.

**SSI workspace layout (fixed)**

Workspace root defaults to ``…/student-success-intervention/pipelines`` (override with
``--ssi_pipelines_workspace_root``).

- **Preprocessing** is always at
  ``<workspace_root>/<databricks_institution_name>/<model_name>/preprocessing.py``.

- **Training** — config and features TOMLs live under the institution folder
  ``<workspace_root>/<databricks_institution_name>/``. Relative paths (no ``..``):

  - If ``--ssi_config_toml_relative_to_institution`` is empty, default is
    ``<model_name>/<config_file_name>`` (e.g. ``john_jay_col_graduation_6years_time_cuny_transfer/config.toml``).

  - If ``--ssi_features_toml_relative_to_institution`` is empty, default is
    ``<model_name>/<features_table_name>`` (e.g. ``john_jay_col_graduation_6years_time_cuny_transfer/features_table.toml``).

  Set the two ``ssi_*_relative_to_institution`` parameters to point elsewhere under the
  institution directory (e.g. ``shared/features_table.toml``).

``--config_file_path`` is required for ``run_type=predict`` (from ``legacy_inference_inputs``);
``run_type=train`` ignores it and uses the workspace paths above.

``databricks_institution_name`` must still match the first folder under ``pipelines/`` for
Unity Catalog naming; it is the same segment used in the SSI repo layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import os
import sys
import tempfile
from pathlib import Path


def _edvise_src_from_repo_layout(script_path: Path) -> Path:
    """``.../src/edvise/scripts/<this>.py`` -> ``.../src``."""
    return script_path.resolve().parents[2]


def _walk_parents_for_edvise_src(start: Path) -> list[Path]:
    """Find ``<repo>/src`` by walking parents for ``src/edvise``."""
    found: list[Path] = []
    for base in (start, *start.parents):
        to_check = [base / "src"]
        if base.name == "src":
            to_check.append(base)
        for cand in to_check:
            if (
                cand.is_dir()
                and (cand / "edvise").is_dir()
                and (cand / "edvise" / "__init__.py").is_file()
            ):
                found.append(cand.resolve())
    return found


def _resolve_this_script_path() -> Path | None:
    """
    Path to this file.

    Databricks sometimes runs this module via ``exec(compile(...))``, so ``__file__`` may be
    missing; the code object's ``co_filename`` still carries the repo path from ``compile``.
    """
    try:
        return Path(__file__).resolve()
    except NameError:
        pass
    frame = inspect.currentframe()
    while frame is not None:
        g = frame.f_globals
        gf = g.get("__file__")
        if isinstance(gf, str) and gf.endswith("legacy_preprocessing.py"):
            return Path(gf).resolve()
        cf = frame.f_code.co_filename
        if cf and not cf.startswith("<") and cf.endswith("legacy_preprocessing.py"):
            return Path(cf).resolve()
        frame = frame.f_back
    if sys.argv and sys.argv[0].endswith("legacy_preprocessing.py"):
        return Path(sys.argv[0]).resolve()
    return None


def _ensure_edvise_src_on_sys_path() -> None:
    """Databricks GIT / notebook-style runs do not install ``edvise``; add ``<repo>/src``."""
    candidates: list[Path] = []

    sp = _resolve_this_script_path()
    if sp is not None:
        candidates.append(_edvise_src_from_repo_layout(sp))

    cwd = Path(os.getcwd()).resolve()
    candidates.extend(_walk_parents_for_edvise_src(cwd))

    if sys.argv and sys.argv[0].endswith(".py"):
        try:
            candidates.extend(
                _walk_parents_for_edvise_src(Path(sys.argv[0]).resolve().parent)
            )
        except OSError:
            pass

    cwd_src = cwd / "src"
    if (cwd_src / "edvise").is_dir():
        candidates.append(cwd_src.resolve())

    seen: set[str] = set()
    for root in candidates:
        root_s = str(root)
        if root_s in seen:
            continue
        seen.add(root_s)
        if (root / "edvise").is_dir() and root_s not in sys.path:
            sys.path.insert(0, root_s)
            return

    try:
        import edvise  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    raise RuntimeError(
        "legacy_preprocessing: could not locate ``edvise`` (add <repo>/src to sys.path). "
        f"script_path={sp!r} cwd={os.getcwd()!r} argv={sys.argv[:3]!r} "
        f"candidates_tried={candidates!r}"
    )


_ensure_edvise_src_on_sys_path()

import tomlkit

from edvise.configs.legacy import deep_substitute_uc_catalog_placeholders
from edvise.dataio.read import from_toml_file

DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT = (
    "/Workspace/Users/6c8d8d76-1399-4065-aeb5-9474d32773cf/"
    "student-success-intervention/pipelines"
)
SSI_PIPELINES_WORKSPACE_ROOT = DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def normalize_fs_path(raw: str) -> Path:
    p = raw.strip()
    if p.startswith("dbfs:/Workspace"):
        p = "/Workspace" + p[len("dbfs:/Workspace") :]
    elif p.startswith("dbfs:/"):
        p = "/dbfs/" + p[6:].lstrip("/")
    return Path(p)


def _normalize_relative_under_institution(rel: str) -> str:
    rel_norm = rel.strip().replace("\\", "/").lstrip("/")
    if not rel_norm:
        raise ValueError("Relative path under institution must not be empty.")
    if any(p == ".." for p in rel_norm.split("/")):
        raise ValueError(f"Relative path must not contain '..': {rel!r}")
    return rel_norm


def resolve_ssi_preprocessing_py(
    institution_id: str,
    model_name: str,
    *,
    workspace_root: str | None = None,
) -> tuple[Path, Path]:
    """
    ``preprocessing.py`` at ``pipelines/<inst>/<model_name>/preprocessing.py``.

    Returns ``(preprocessing_py, institution_base)`` where ``institution_base`` is
    ``pipelines/<inst>/``.
    """
    root = (workspace_root or "").strip() or DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT
    root_r = normalize_fs_path(root).resolve()
    inst = institution_id.strip()
    mn = (model_name or "").strip()
    if not inst:
        raise ValueError("databricks_institution_name must be non-empty.")
    if not mn:
        raise ValueError(
            "model_name must be non-empty (folder under pipelines/<inst>/ "
            "that contains preprocessing.py)."
        )
    institution_base = (root_r / inst).resolve()
    try:
        institution_base.relative_to(root_r)
    except ValueError as e:
        raise ValueError(
            f"Institution path {institution_base} is not under pipelines root {root_r}"
        ) from e

    py = (institution_base / mn / "preprocessing.py").resolve()
    try:
        py.relative_to(institution_base)
    except ValueError as e:
        raise ValueError(
            f"preprocessing.py path {py} escapes institution directory {institution_base}"
        ) from e
    if not py.is_file():
        raise FileNotFoundError(
            f"Expected preprocessing.py at {py} "
            f"(institution={inst!r}, model_name={mn!r})."
        )
    return py, institution_base


def resolve_ssi_training_workspace_toml_paths(
    institution_id: str,
    model_name: str,
    config_file_name: str,
    features_table_name: str,
    *,
    workspace_root: str | None = None,
    ssi_config_toml_relative_to_institution: str | None = None,
    ssi_features_toml_relative_to_institution: str | None = None,
) -> tuple[str, str]:
    """
    Absolute paths to training config and features TOMLs under ``pipelines/<inst>/``.
    """
    _, institution_base = resolve_ssi_preprocessing_py(
        institution_id,
        model_name,
        workspace_root=workspace_root,
    )
    mn = (model_name or "").strip()
    cfg_rel = (ssi_config_toml_relative_to_institution or "").strip() or (
        f"{mn}/{config_file_name.strip()}"
    )
    feat_rel = (ssi_features_toml_relative_to_institution or "").strip() or (
        f"{mn}/{features_table_name.strip()}"
    )
    cfg_norm = _normalize_relative_under_institution(cfg_rel)
    feat_norm = _normalize_relative_under_institution(feat_rel)

    inst_r = institution_base.resolve()
    cfg_p = (inst_r / cfg_norm).resolve()
    feat_p = (inst_r / feat_norm).resolve()
    for label, p in (("config", cfg_p), ("features", feat_p)):
        try:
            p.relative_to(inst_r)
        except ValueError as e:
            raise ValueError(
                f"{label} path {p} is not under institution directory {inst_r}"
            ) from e
    if not cfg_p.is_file():
        raise FileNotFoundError(f"Training config TOML not found: {cfg_p}")
    if not feat_p.is_file():
        raise FileNotFoundError(f"Features table TOML not found: {feat_p}")
    return str(cfg_p), str(feat_p)


def _ensure_edvise_src_on_path() -> None:
    """Ensure ``edvise`` is importable before loading workspace ``preprocessing.py``."""
    _ensure_edvise_src_on_sys_path()


def load_module_from_file(py_file: Path, institution_pipeline_dir: Path):
    _ensure_edvise_src_on_path()
    ssi_root = str(normalize_fs_path(SSI_PIPELINES_WORKSPACE_ROOT).parent)
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


def _legacy_toml_has_uc_placeholder(text: str) -> bool:
    return (
        "CATALOG." in text
        or "/Volumes/CATALOG/" in text
        or "dbfs:/Volumes/CATALOG/" in text
    )


def materialize_legacy_config_with_uc_catalog(
    config_file_path: str, uc_catalog: str
) -> str:
    """
    When the TOML uses ``CATALOG`` placeholders and ``uc_catalog`` is non-empty,
    write a temp file with substitutions and return its path; otherwise return the
    original path unchanged.
    """
    cat = (uc_catalog or "").strip()
    if not cat:
        return config_file_path
    path = Path(config_file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        LOGGER.warning(
            "Could not read %s for CATALOG resolution: %s", config_file_path, exc
        )
        return config_file_path
    if not _legacy_toml_has_uc_placeholder(text):
        return config_file_path
    data = from_toml_file(str(path.resolve()))
    patched = deep_substitute_uc_catalog_placeholders(data, cat)
    fd, tmp_path = tempfile.mkstemp(prefix="edvise_legacy_cfg_", suffix=".toml")
    try:
        os.write(fd, tomlkit.dumps(patched).encode("utf-8"))
    finally:
        os.close(fd)
    LOGGER.info(
        "Resolved CATALOG placeholders to uc_catalog=%r; using temp config %s",
        cat,
        tmp_path,
    )
    return tmp_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Legacy H2O preprocessing via SSI workspace preprocessing.py."
    )
    parser.add_argument(
        "--config_file_path",
        default="",
        help="Required for run_type=predict. Ignored for train.",
    )
    parser.add_argument(
        "--config_file_name",
        default="config.toml",
        help="Basename used in default config path <model_name>/<config_file_name> (train).",
    )
    parser.add_argument(
        "--features_table_name",
        default="features_table.toml",
        help="Basename used in default features path <model_name>/<features_table_name> (train).",
    )
    parser.add_argument(
        "--ssi_config_toml_relative_to_institution",
        default="",
        help=(
            "Train: path under pipelines/<inst>/ to config TOML. "
            "Empty → <model_name>/<config_file_name>."
        ),
    )
    parser.add_argument(
        "--ssi_features_toml_relative_to_institution",
        default="",
        help=(
            "Train: path under pipelines/<inst>/ to features TOML. "
            "Empty → <model_name>/<features_table_name>."
        ),
    )
    parser.add_argument(
        "--ssi_pipelines_workspace_root",
        default="",
        help="Override …/student-success-intervention/pipelines.",
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
        help="Institution folder under pipelines/ (matches SSI repo).",
    )
    parser.add_argument(
        "--model_name",
        default="",
        help=(
            "Subfolder under pipelines/<inst>/ containing preprocessing.py "
            "(e.g. john_jay_col_graduation_6years_time_cuny_transfer); same string as the "
            "registered Unity Catalog model name for legacy jobs."
        ),
    )
    parser.add_argument(
        "--DB_workspace",
        default="",
        help=(
            "Unity Catalog workspace name (e.g. dev_sst_02). When set, "
            "CATALOG placeholders in the config TOML are rewritten before "
            "school preprocessing runs."
        ),
    )
    args = parser.parse_args()

    inst = (args.databricks_institution_name or "").strip()
    if not inst:
        raise SystemExit("--databricks_institution_name is required when preprocessing runs.")
    model_name = (args.model_name or "").strip()
    if not model_name:
        raise SystemExit(
            "--model_name is required when preprocessing runs "
            "(folder under pipelines/<inst>/ with preprocessing.py)."
        )

    ws = (args.ssi_pipelines_workspace_root or "").strip() or None
    root = ws or DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT

    if args.run_type == "train":
        cfg_rel = (args.ssi_config_toml_relative_to_institution or "").strip() or None
        feat_rel = (args.ssi_features_toml_relative_to_institution or "").strip() or None
        effective_config, effective_features = resolve_ssi_training_workspace_toml_paths(
            inst,
            model_name,
            args.config_file_name,
            args.features_table_name,
            workspace_root=ws,
            ssi_config_toml_relative_to_institution=cfg_rel,
            ssi_features_toml_relative_to_institution=feat_rel,
        )
        LOGGER.info(
            "Training: SSI config %s, features %s",
            effective_config,
            effective_features,
        )
    else:
        p = (args.config_file_path or "").strip()
        if not p:
            raise SystemExit("--config_file_path is required when run_type=predict.")
        effective_config = p

    effective_config = materialize_legacy_config_with_uc_catalog(
        effective_config, args.DB_workspace
    )

    LOGGER.info(
        "Loading preprocessing from pipelines/%s/%s/preprocessing.py (root=%s)",
        inst,
        model_name,
        (args.ssi_pipelines_workspace_root or "").strip() or SSI_PIPELINES_WORKSPACE_ROOT,
    )
    py_file, inst_dir = resolve_ssi_preprocessing_py(
        inst,
        model_name,
        workspace_root=root,
    )
    LOGGER.info("Workspace preprocessing file: %s", py_file)
    mod = load_module_from_file(py_file, inst_dir)
    label = str(py_file)

    run = getattr(mod, "run", None)
    if run is None:
        raise RuntimeError(f"{label!r} must define a run() function.")
    LOGGER.info(
        "Running %s.run(config_file_path=%r, run_type=%r)",
        label,
        effective_config,
        args.run_type,
    )
    run(config_file_path=effective_config, run_type=args.run_type)


if __name__ == "__main__":
    main()
