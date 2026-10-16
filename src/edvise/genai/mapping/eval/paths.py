"""Unity Catalog paths shared by the Step 2a and Step 2b notebook evaluators."""

from __future__ import annotations

import os
from pathlib import Path

from edvise.configs import genai as genai_cfg
from edvise.genai.mapping.eval.models import MODEL_BASE_SLUG, SHOT_TAG
from edvise.genai.mapping.shared.reference_pin import resolve_sma_few_shot_pin


def folder_slug_2b(model: str) -> str:
    """Per-model artifact folder name under ``…/genai_mapping/eval/`` for Step 2b."""
    base = MODEL_BASE_SLUG.get(model, model.replace("-", "_"))
    return f"{base}_2b_{SHOT_TAG}"


def _eval_uc_catalog() -> str | None:
    """Workspace catalog for ``/Volumes/<catalog>/…`` (matches job ``DB_workspace``)."""
    for key in ("EDVISE_EVAL_UC_CATALOG", "GENAI_HITL_CATALOG", "DB_workspace"):
        value = os.environ.get(key)
        if value and str(value).strip():
            return str(value).strip()
    return None


def _require_uc_catalog_for_eval() -> str:
    catalog = _eval_uc_catalog()
    if not catalog:
        raise ValueError(
            "SMA eval loads only from Unity Catalog volumes. Set one of "
            "EDVISE_EVAL_UC_CATALOG, GENAI_HITL_CATALOG, or DB_workspace."
        )
    return catalog


def _silver_genai_root_path(institution_id: str) -> Path:
    """``…/{institution_id}_silver/silver_volume/genai_mapping`` on the UC volume."""
    return Path(
        genai_cfg.silver_genai_mapping_root(
            institution_id, catalog=_require_uc_catalog_for_eval()
        )
    )


def _eval_run_root(target_id: str) -> Path:
    """Writable eval workspace: ``<silver_genai_root(target)>/eval``."""
    return _silver_genai_root_path(target_id) / "eval"


def _resolve_gold_mapping_manifest_for_eval(target_id: str) -> Path:
    """
    Ground-truth manifest for Step 2a scoring.

    * ``EDVISE_EVAL_GOLD_MANIFEST_PATH`` — absolute file on a volume (or any path).
    * Else ``<silver_genai_root(target)>/eval/gold/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_GOLD_MANIFEST_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"EDVISE_EVAL_GOLD_MANIFEST_PATH not found: {path}")
        return path.resolve()
    path = _eval_run_root(target_id) / "gold" / "manifest_map.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Gold mapping manifest not found. Place ground-truth at "
            f"{path} (same shape as pipeline ``manifest_map.json``) or set "
            "EDVISE_EVAL_GOLD_MANIFEST_PATH."
        )
    return path


def _resolve_gold_transformation_map_for_eval(target_id: str) -> Path:
    """
    Ground-truth transformation map for Step 2b scoring.

    * ``EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH``
    * Else ``<silver_genai_root(target)>/eval/gold/transformation_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH not found: {path}"
            )
        return path.resolve()
    path = _eval_run_root(target_id) / "gold" / "transformation_map.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Gold transformation map not found. Place ground-truth at "
            f"{path} or set EDVISE_EVAL_GOLD_TRANSFORMATION_MAP_PATH."
        )
    return path


def _resolve_target_mapping_manifest_for_eval(target_id: str) -> Path:
    """
    Target mapping manifest for Step 2b prompts (field plans).

    * ``EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH``
    * Else ``<silver_genai_root(target)>/active/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH not found: {path}"
            )
        return path.resolve()
    path = _silver_genai_root_path(target_id) / "active" / "manifest_map.json"
    if not path.is_file():
        raise FileNotFoundError(
            "Target mapping manifest for 2b not found at pipeline active path "
            f"{path}. Promote SMA artifacts or set "
            "EDVISE_EVAL_TARGET_MAPPING_MANIFEST_PATH."
        )
    return path


def _resolve_reference_mapping_manifest(reference_id: str) -> Path:
    """
    Few-shot reference manifest for Step 2a.

    * ``EDVISE_EVAL_REFERENCE_MANIFEST_PATH`` — explicit file.
    * Else pinned ``…/genai_mapping/references/<id>/current/manifest_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_MANIFEST_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_MANIFEST_PATH not found: {path}"
            )
        return path.resolve()
    few = resolve_sma_few_shot_pin(reference_id, catalog=_require_uc_catalog_for_eval())
    return few.manifest_map


def _resolve_reference_transformation_map_for_eval(reference_id: str) -> Path:
    """
    Few-shot reference transformation map for Step 2b.

    * ``EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH``
    * Else pinned ``…/genai_mapping/references/<id>/current/transformation_map.json``.
    """
    override = os.environ.get("EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_REFERENCE_TRANSFORMATION_MAP_PATH not found: {path}"
            )
        return path.resolve()
    few = resolve_sma_few_shot_pin(reference_id, catalog=_require_uc_catalog_for_eval())
    return few.transformation_map


def _resolve_target_schema_contract_for_eval(target_id: str) -> Path:
    """
    Target enriched schema contract for Step 2a / 2b.

    * ``EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH``
    * Else ``…/genai_mapping/active/enriched_schema_contract.json``.
    """
    override = os.environ.get("EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH")
    if override and str(override).strip():
        path = Path(str(override).strip()).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH not found: {path}"
            )
        return path.resolve()
    path = (
        _silver_genai_root_path(target_id) / "active" / "enriched_schema_contract.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Target schema contract not found at pipeline active path "
            f"{path}. Run IA / promote ``enriched_schema_contract.json`` or set "
            "EDVISE_EVAL_TARGET_SCHEMA_CONTRACT_PATH."
        )
    return path
