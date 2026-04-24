"""
On-disk SMA (2a) HITL + manifest outputs in the shapes expected by the resolver.

See :mod:`edvise.genai.mapping.schema_mapping_agent.hitl.schemas` for the
per-institution file layout (``sma_hitl.json``, ``sma_manifest_output.json``).
"""

from __future__ import annotations

from pathlib import Path

from edvise.genai.mapping.schema_mapping_agent.hitl.schemas import (
    InstitutionSMAHITLItems,
    SMAHITLItem,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine import (
    apply_refinement_review_status_safety_net,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    FieldMappingManifest,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.validation import (
    ManifestValidationError,
)
from edvise.genai.mapping.shared.hitl.json_io import (
    read_pydantic_json,
    write_pydantic_json,
)

SMA_HITL_BASENAME = "sma_hitl.json"
SMA_MANIFEST_OUTPUT_BASENAME = "sma_manifest_output.json"


def write_sma_hitl_artifact(
    output_dir: str | Path,
    envelope: InstitutionSMAHITLItems,
    *,
    basename: str = SMA_HITL_BASENAME,
) -> Path:
    """Write the HITL envelope JSON (default ``sma_hitl.json``). Creates parent dirs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / basename
    write_pydantic_json(path, envelope)
    return path


def write_sma_manifest_artifact(
    output_dir: str | Path,
    manifest: FieldMappingManifest,
    *,
    basename: str = SMA_MANIFEST_OUTPUT_BASENAME,
) -> Path:
    """Write the refined single-entity manifest (default ``sma_manifest_output.json``)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / basename
    write_pydantic_json(path, manifest)
    return path


def write_sma_hitl_and_manifest_artifacts(
    output_dir: str | Path,
    *,
    hitl: InstitutionSMAHITLItems,
    manifest: FieldMappingManifest,
    hitl_basename: str = SMA_HITL_BASENAME,
    manifest_basename: str = SMA_MANIFEST_OUTPUT_BASENAME,
    validation_errors: list[ManifestValidationError] | None = None,
) -> tuple[Path, Path]:
    """Write HITL envelope and manifest to the same directory.

    When ``validation_errors`` is provided, runs
    :func:`~edvise.genai.mapping.schema_mapping_agent.manifest.prompts.refine.apply_refinement_review_status_safety_net`
    on the manifest before writing (post-parse contract enforcement).
    """
    if validation_errors is not None:
        apply_refinement_review_status_safety_net(
            manifest, validation_errors, hitl.items
        )
    h = write_sma_hitl_artifact(output_dir, hitl, basename=hitl_basename)
    m = write_sma_manifest_artifact(output_dir, manifest, basename=manifest_basename)
    return h, m


def load_sma_hitl(hitl_path: str | Path) -> InstitutionSMAHITLItems:
    """Load ``sma_hitl.json`` (or any path) as :class:`InstitutionSMAHITLItems`."""
    path = Path(hitl_path)
    try:
        return read_pydantic_json(path, InstitutionSMAHITLItems)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"SMA HITL file not found: {path}") from e


def load_sma_manifest_output(manifest_path: str | Path) -> FieldMappingManifest:
    """Load ``sma_manifest_output.json`` as :class:`FieldMappingManifest`."""
    return read_pydantic_json(Path(manifest_path), FieldMappingManifest)


def unique_sma_hitl_items_by_item_id(items: list[SMAHITLItem]) -> list[SMAHITLItem]:
    """Return items in order, keeping the first occurrence of each ``item_id``."""
    seen: set[str] = set()
    out: list[SMAHITLItem] = []
    for it in items:
        if it.item_id in seen:
            continue
        seen.add(it.item_id)
        out.append(it)
    return out


__all__ = [
    "SMA_HITL_BASENAME",
    "SMA_MANIFEST_OUTPUT_BASENAME",
    "load_sma_hitl",
    "load_sma_manifest_output",
    "unique_sma_hitl_items_by_item_id",
    "write_sma_hitl_artifact",
    "write_sma_manifest_artifact",
    "write_sma_hitl_and_manifest_artifacts",
]
