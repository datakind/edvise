"""
Run school-specific legacy postprocessing after ``inference_h2o`` or ``training_h2o``.

**SSI workspace layout (fixed)**

Workspace root defaults to ``…/student-success-intervention/pipelines`` (override with
``--ssi_pipelines_workspace_root``).

- **Postprocessing** (optional) is at
  ``<workspace_root>/<databricks_institution_name>/<model_name>/postprocessing.py``.

When ``[postprocessing].enabled`` is false or absent in the project config TOML, or the
workspace file is missing, this script exits successfully without running school code.

``run_type=predict``: ``--config_file_path`` from ``legacy_inference_inputs``;
``--job_root_dir`` must match ``inference_h2o`` (CSV under ``ext/inference_output``).

``run_type=train``: ``--config_file_path`` from ``training_h2o`` task values (updated
config under silver volume) or UC ``training_inputs/{config_file_name}``; ``--gold_table_path``
points at the catalog/schema written by ``training_h2o`` (``{gold_table_path}.advisor_output``).
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path

from edvise import configs, dataio
from edvise.scripts.legacy_preprocessing import (
    DEFAULT_LEGACY_CONFIG_BASENAME,
    DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT,
    SSI_PIPELINES_WORKSPACE_ROOT,
    load_module_from_file,
    materialize_legacy_config_with_uc_catalog,
    normalize_fs_path,
    resolve_legacy_training_toml_paths,
)
from edvise.utils.databricks import normalize_legacy_uc_model_short_name

logging.basicConfig(level=logging.INFO, force=True)
LOGGER = logging.getLogger(__name__)


def legacy_postprocessing_enabled(cfg_path: str, *, DB_workspace: str = "") -> bool:
    """Return whether ``[postprocessing].enabled`` is set in the project config TOML."""
    effective = materialize_legacy_config_with_uc_catalog(cfg_path, DB_workspace)
    cfg = dataio.read.read_config(
        effective,
        schema=configs.legacy.LegacyProjectConfig,
    )
    return bool(cfg.postprocessing is not None and cfg.postprocessing.enabled)


def resolve_postprocess_config_path(args: argparse.Namespace) -> str:
    """
    Resolve project config TOML for postprocessing.

    Predict jobs pass ``--config_file_path`` from ``legacy_inference_inputs``.
    Train jobs prefer ``--config_file_path`` from ``training_h2o`` task values, then
    ``{silver_volume_path}/{model_run_id}/training/{config_file_name}``, then UC training_inputs.
    """
    cfg_path = (args.config_file_path or "").strip()
    if cfg_path:
        return cfg_path

    if args.run_type != "train":
        raise SystemExit("--config_file_path is required.")

    silver = (getattr(args, "silver_volume_path", None) or "").strip()
    model_run_id = (getattr(args, "model_run_id", None) or "").strip()
    cfg_name = (getattr(args, "config_file_name", None) or "").strip()
    if not cfg_name:
        cfg_name = DEFAULT_LEGACY_CONFIG_BASENAME
    if silver and model_run_id:
        candidate = os.path.join(silver, model_run_id, "training", cfg_name)
        if os.path.isfile(candidate):
            LOGGER.info("Resolved training config from silver volume: %s", candidate)
            return candidate

    inst = (args.databricks_institution_name or "").strip()
    db_ws = (args.DB_workspace or "").strip()
    if inst and db_ws:
        try:
            resolved, _ = resolve_legacy_training_toml_paths(
                db_ws, inst, config_file_name=cfg_name
            )
            LOGGER.info("Resolved training config from UC training_inputs: %s", resolved)
            return resolved
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(
                "Could not resolve training config for postprocessing. "
                f"Pass --config_file_path or ensure UC training_inputs/{cfg_name} exists."
            ) from exc

    raise SystemExit("--config_file_path is required for train postprocessing.")


def resolve_ssi_postprocessing_py(
    institution_id: str,
    model_name: str,
    *,
    workspace_root: str | None = None,
) -> tuple[Path | None, Path]:
    """
    ``postprocessing.py`` at ``pipelines/<inst>/<model_name>/postprocessing.py``.

    Returns ``(postprocessing_py_or_none, institution_base)``. Missing file → ``(None, …)``.
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
            "that may contain postprocessing.py)."
        )
    institution_base = (root_r / inst).resolve()
    try:
        institution_base.relative_to(root_r)
    except ValueError as e:
        raise ValueError(
            f"Institution path {institution_base} is not under pipelines root {root_r}"
        ) from e

    py = (institution_base / mn / "postprocessing.py").resolve()
    try:
        py.relative_to(institution_base)
    except ValueError as e:
        raise ValueError(
            f"postprocessing.py path {py} escapes institution directory {institution_base}"
        ) from e
    if not py.is_file():
        return None, institution_base
    return py, institution_base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional legacy postprocessing via SSI workspace postprocessing.py."
    )
    parser.add_argument(
        "--config_file_path",
        default="",
        help="Project config TOML (from inference_setup); controls [postprocessing].enabled.",
    )
    parser.add_argument(
        "--ssi_pipelines_workspace_root",
        default="",
        help="Override …/student-success-intervention/pipelines.",
    )
    parser.add_argument(
        "--run_type",
        default="predict",
        choices=("train", "predict"),
        help="Forwarded to the school postprocessing module.",
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
            "Subfolder under pipelines/<inst>/ containing postprocessing.py "
            "(e.g. john_jay_col_primary_0_ckpt_90_credits_earned)."
        ),
    )
    parser.add_argument(
        "--DB_workspace",
        default="",
        help=(
            "Unity Catalog workspace name (e.g. dev_sst_02). When set, "
            "CATALOG placeholders in the config TOML are rewritten before "
            "school postprocessing runs."
        ),
    )
    parser.add_argument(
        "--db_run_id",
        default="",
        help="Inference job run id (for logging).",
    )
    parser.add_argument(
        "--job_root_dir",
        default="",
        help=(
            "Gold volume inference job folder (same as inference_h2o). "
            "School postprocessing reads ``{job_root_dir}/ext/inference_output``."
        ),
    )
    parser.add_argument(
        "--gold_table_path",
        default="",
        help=(
            "Train only: UC gold schema prefix (e.g. dev_sst_02.inst_gold). "
            "Reads ``{gold_table_path}.advisor_output`` written by training_h2o."
        ),
    )
    parser.add_argument(
        "--silver_volume_path",
        default="",
        help="Train only: fallback path to resolve updated config under model run folder.",
    )
    parser.add_argument(
        "--config_file_name",
        default="",
        help="Train only: config TOML basename for silver-volume and UC fallback resolution.",
    )
    parser.add_argument(
        "--model_run_id",
        default="",
        help=(
            "Train only: selected MLflow model run id (folder name under silver volume "
            "after training_h2o renames the run root)."
        ),
    )
    args = parser.parse_args()

    cfg_path = resolve_postprocess_config_path(args)

    if not legacy_postprocessing_enabled(cfg_path, DB_workspace=args.DB_workspace):
        LOGGER.info(
            "[postprocessing].enabled is false or unset in %s; skipping.",
            cfg_path,
        )
        return

    inst = (args.databricks_institution_name or "").strip()
    if not inst:
        raise SystemExit(
            "--databricks_institution_name is required when postprocessing is enabled."
        )
    model_name = normalize_legacy_uc_model_short_name(
        args.model_name or "",
        workspace=(args.DB_workspace or ""),
        institution=inst,
    )
    if not model_name:
        raise SystemExit(
            "--model_name is required when postprocessing is enabled."
        )

    ws = (args.ssi_pipelines_workspace_root or "").strip() or None
    root = ws or DEFAULT_SSI_PIPELINES_WORKSPACE_ROOT

    py_file, inst_dir = resolve_ssi_postprocessing_py(
        inst,
        model_name,
        workspace_root=root,
    )
    if py_file is None:
        LOGGER.info(
            "No postprocessing.py at pipelines/%s/%s/; skipping (optional).",
            inst,
            model_name,
        )
        return

    job_root = (args.job_root_dir or "").strip()
    if args.run_type == "predict" and not job_root:
        raise SystemExit(
            "--job_root_dir is required when run_type=predict and postprocessing runs."
        )
    gold_table = (args.gold_table_path or "").strip()
    if args.run_type == "train" and not gold_table:
        raise SystemExit(
            "--gold_table_path is required when run_type=train and postprocessing runs."
        )

    effective_config = materialize_legacy_config_with_uc_catalog(
        cfg_path, args.DB_workspace
    )

    LOGGER.info(
        "Loading postprocessing from pipelines/%s/%s/postprocessing.py (root=%s)",
        inst,
        model_name,
        (args.ssi_pipelines_workspace_root or "").strip() or SSI_PIPELINES_WORKSPACE_ROOT,
    )
    LOGGER.info("Workspace postprocessing file: %s", py_file)
    mod = load_module_from_file(py_file, inst_dir)
    label = str(py_file)

    run = getattr(mod, "run", None)
    if run is None:
        raise RuntimeError(f"{label!r} must define a run() function.")

    run_kwargs: dict = {
        "config_file_path": effective_config,
        "run_type": args.run_type,
        "db_run_id": args.db_run_id,
        "job_root_dir": job_root,
        "gold_table_path": gold_table,
        "DB_workspace": args.DB_workspace,
    }
    sig = inspect.signature(run)
    run_kwargs = {k: v for k, v in run_kwargs.items() if k in sig.parameters}

    LOGGER.info(
        "Running %s.run(%s)",
        label,
        ", ".join(f"{k}={v!r}" for k, v in run_kwargs.items()),
    )
    run(**run_kwargs)


if __name__ == "__main__":
    main()
