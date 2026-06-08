"""
Run school-specific legacy preprocessing before ``training_h2o`` or ``inference_h2o``.

**SSI workspace layout (fixed)**

Workspace root defaults to ``…/student-success-intervention/pipelines`` (override with
``--ssi_pipelines_workspace_root``).

- **Preprocessing** is always at
  ``<workspace_root>/<databricks_institution_name>/<model_name>/preprocessing.py``.

- **Training** — config and features TOMLs are read from the institution bronze UC volume:

  ``/Volumes/<DB_workspace>/<databricks_institution_name>_bronze/bronze_volume/training_inputs/``

  - Config: ``<config_file_name>`` (default ``config.toml``).
  - Features: ``<features_table_name>`` (default ``features_table.toml``).

  Preprocessing code still loads from the SSI workspace mirror at
  ``<workspace_root>/<databricks_institution_name>/<model_name>/preprocessing.py``.

``--config_file_path`` is required for ``run_type=predict`` (from ``legacy_inference_inputs``);
``run_type=train`` ignores it and uses the workspace paths above.

For ``run_type=predict``, optional ``--term_filter`` (JSON list of strings) overrides
``[inference].term`` in the config when non-empty; omit or null to use the config.
School ``run()`` may accept ``term_filter`` and apply cohort/term filtering before writing
``model_features`` (so ``inference_h2o`` loads an already-filtered table).

``databricks_institution_name`` must still match the first folder under ``pipelines/`` for
Unity Catalog naming; it is the same segment used in the SSI repo layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import os
import shutil
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
from edvise.utils.databricks import normalize_legacy_uc_model_short_name

DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT = (
    "/Workspace/Users/6c8d8d76-1399-4065-aeb5-9474d32773cf/"
    "student-success-intervention/pipelines"
)
SSI_PIPELINES_WORKSPACE_ROOT = DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT

# Fixed basenames under ``…/bronze_volume/training_inputs/`` on UC.
DEFAULT_LEGACY_CONFIG_BASENAME = "config.toml"
DEFAULT_FEATURES_TABLE_NAME = "features_table.toml"

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def normalize_fs_path(raw: str) -> Path:
    p = raw.strip()
    if p.startswith("dbfs:/Workspace"):
        p = "/Workspace" + p[len("dbfs:/Workspace") :]
    elif p.startswith("dbfs:/"):
        p = "/dbfs/" + p[6:].lstrip("/")
    return Path(p)


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


def legacy_training_inputs_uc_dir(db_workspace: str, institution_id: str) -> Path:
    """``/Volumes/<catalog>/<inst>_bronze/bronze_volume/training_inputs``."""
    ws = (db_workspace or "").strip()
    inst = (institution_id or "").strip()
    if not ws:
        raise ValueError("DB_workspace must be non-empty.")
    if not inst:
        raise ValueError("databricks_institution_name must be non-empty.")
    return normalize_fs_path(
        f"/Volumes/{ws}/{inst}_bronze/bronze_volume/training_inputs"
    )


def resolve_legacy_training_toml_paths(
    db_workspace: str,
    institution_id: str,
    *,
    config_file_name: str = DEFAULT_LEGACY_CONFIG_BASENAME,
    features_table_name: str = DEFAULT_FEATURES_TABLE_NAME,
) -> tuple[str, str]:
    """
    Resolve training config and features TOML paths on the institution bronze UC volume.

    Layout: ``{training_inputs}/{config_file_name}`` and ``{training_inputs}/{features_table_name}``.
    """
    base = legacy_training_inputs_uc_dir(db_workspace, institution_id)
    cfg_name = (config_file_name or "").strip() or DEFAULT_LEGACY_CONFIG_BASENAME
    feat_name = (features_table_name or "").strip() or DEFAULT_FEATURES_TABLE_NAME
    if "/" in cfg_name or "\\" in cfg_name:
        raise ValueError(f"config_file_name must be a single filename: {cfg_name!r}")
    if "/" in feat_name or "\\" in feat_name:
        raise ValueError(f"features_table_name must be a single filename: {feat_name!r}")

    cfg_p = (base / cfg_name).resolve()
    feat_p = (base / feat_name).resolve()
    try:
        cfg_p.relative_to(base.resolve())
        feat_p.relative_to(base.resolve())
    except ValueError as e:
        raise ValueError(
            f"Resolved TOML path escapes training_inputs directory {base}"
        ) from e
    if not cfg_p.is_file():
        raise FileNotFoundError(f"Training config TOML not found: {cfg_p}")
    if not feat_p.is_file():
        raise FileNotFoundError(f"Features table TOML not found: {feat_p}")
    return str(cfg_p), str(feat_p)


def copy_legacy_uc_config_for_training(config_uc_path: str) -> str:
    """
    Copy a UC (or any read-only) config TOML to a writable temp file.

    ``training_h2o`` updates run metadata and pipeline_version in the config; using a copy
    avoids mutating the canonical UC object.
    """
    src = normalize_fs_path(config_uc_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Config TOML not found: {src}")
    fd, tmp_path = tempfile.mkstemp(prefix="edvise_legacy_train_cfg_", suffix=".toml")
    try:
        os.close(fd)
        shutil.copy2(str(src), tmp_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    LOGGER.info("Copied UC legacy config %s -> %s (writable training copy)", src, tmp_path)
    return tmp_path


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
        default=DEFAULT_LEGACY_CONFIG_BASENAME,
        help=(
            "Train only: config TOML basename under "
            "/Volumes/<DB_workspace>/<inst>_bronze/bronze_volume/training_inputs/."
        ),
    )
    parser.add_argument(
        "--features_table_name",
        default=DEFAULT_FEATURES_TABLE_NAME,
        help=(
            "Train only: features TOML basename under "
            "/Volumes/<DB_workspace>/<inst>_bronze/bronze_volume/training_inputs/."
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
            "(e.g. john_jay_col_graduation_6years_time_cuny_transfer), or the full UC name "
            "catalog.inst_gold.<that_folder>; the short segment is used for SSI paths."
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
    parser.add_argument(
        "--term_filter",
        type=str,
        default=None,
        help=(
            "Predict only: JSON list of term labels (e.g. [\"fall 2025\"]). "
            "Omit or null to use [inference].term from config. Forwarded if "
            "``preprocessing.run`` accepts a ``term_filter`` parameter."
        ),
    )
    args = parser.parse_args()

    inst = (args.databricks_institution_name or "").strip()
    if not inst:
        raise SystemExit("--databricks_institution_name is required when preprocessing runs.")
    model_name = normalize_legacy_uc_model_short_name(
        args.model_name or "",
        workspace=(args.DB_workspace or ""),
        institution=inst,
    )
    if not model_name:
        raise SystemExit(
            "--model_name is required when preprocessing runs "
            "(folder under pipelines/<inst>/ with preprocessing.py)."
        )

    ws = (args.ssi_pipelines_workspace_root or "").strip() or None
    root = ws or DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT

    if args.run_type == "train":
        db_ws = (args.DB_workspace or "").strip()
        if not db_ws:
            raise SystemExit("--DB_workspace is required when run_type=train.")
        effective_config, effective_features = resolve_legacy_training_toml_paths(
            db_ws,
            inst,
            config_file_name=args.config_file_name,
            features_table_name=args.features_table_name,
        )
        LOGGER.info(
            "Training: UC config %s, features %s",
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
    run_kwargs: dict = {
        "config_file_path": effective_config,
        "run_type": args.run_type,
    }
    if "term_filter" in inspect.signature(run).parameters:
        run_kwargs["term_filter"] = getattr(args, "term_filter", None)
        LOGGER.info(
            "Running %s.run(config_file_path=%r, run_type=%r, term_filter=%r)",
            label,
            effective_config,
            args.run_type,
            run_kwargs.get("term_filter"),
        )
    else:
        LOGGER.info(
            "Running %s.run(config_file_path=%r, run_type=%r)",
            label,
            effective_config,
            args.run_type,
        )
    run(**run_kwargs)


if __name__ == "__main__":
    main()
