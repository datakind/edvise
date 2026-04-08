"""
Write IdentityAgent outputs in the shapes expected by :mod:`hitl.resolver`.

Grain / term *contract* JSON uses ``datasets[table].grain_contract`` and
``datasets[table].term_config`` so ``resolve_items`` / ``apply_hook_spec`` can
navigate configs consistently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from edvise.genai.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.identity_agent.hitl.schemas import HITLItem, InstitutionHITLItems
from edvise.genai.identity_agent.term_normalization.schemas import TermContract


def build_grain_config_for_resolver(
    institution_id: str,
    contracts_by_dataset: Mapping[str, GrainContract],
) -> dict:
    """
    Resolver-shaped JSON: ``datasets[table].grain_contract`` holds the pass-1 contract.
    """
    return {
        "institution_id": institution_id,
        "datasets": {
            name: {
                "grain_contract": gc.model_dump(mode="json"),
            }
            for name, gc in contracts_by_dataset.items()
        },
    }


def build_term_config_for_resolver(
    institution_id: str,
    contracts_by_dataset: Mapping[str, TermContract],
) -> dict:
    """
    Resolver-shaped JSON: ``datasets[table]`` mirrors :class:`TermContract` fields
    (``term_config`` is the nested term order config).
    """
    out: dict = {"institution_id": institution_id, "datasets": {}}
    for name, tc in contracts_by_dataset.items():
        out["datasets"][name] = tc.model_dump(mode="json")
    return out


def write_identity_grain_artifacts(
    output_dir: str | Path,
    institution_id: str,
    contracts_by_dataset: Mapping[str, GrainContract],
    hitl_items: list[HITLItem],
) -> tuple[Path, Path]:
    """
    Write ``identity_grain_output.json`` (resolver config) and ``identity_grain_hitl.json``.

    Returns
    -------
    tuple[Path, Path]
        Paths ``(config_path, hitl_path)``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_grain_config_for_resolver(institution_id, contracts_by_dataset)
    env = InstitutionHITLItems(
        institution_id=institution_id,
        domain="grain",
        items=hitl_items,
    )
    config_path = output_dir / "identity_grain_output.json"
    hitl_path = output_dir / "identity_grain_hitl.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    hitl_path.write_text(env.model_dump_json(indent=2))
    return config_path, hitl_path


def write_identity_term_artifacts(
    output_dir: str | Path,
    institution_id: str,
    contracts_by_dataset: Mapping[str, TermContract],
    hitl_items: list[HITLItem],
) -> tuple[Path, Path]:
    """
    Write ``identity_term_output.json`` (resolver config) and ``identity_term_hitl.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_term_config_for_resolver(institution_id, contracts_by_dataset)
    env = InstitutionHITLItems(
        institution_id=institution_id,
        domain="term",
        items=hitl_items,
    )
    config_path = output_dir / "identity_term_output.json"
    hitl_path = output_dir / "identity_term_hitl.json"
    config_path.write_text(json.dumps(cfg, indent=2))
    hitl_path.write_text(env.model_dump_json(indent=2))
    return config_path, hitl_path


def dedupe_hitl_items(items: list[HITLItem]) -> list[HITLItem]:
    """Keep first occurrence per ``item_id`` (e.g. when top-level and nested lists overlap)."""
    seen: set[str] = set()
    out: list[HITLItem] = []
    for it in items:
        if it.item_id in seen:
            continue
        seen.add(it.item_id)
        out.append(it)
    return out


__all__ = [
    "build_grain_config_for_resolver",
    "build_term_config_for_resolver",
    "dedupe_hitl_items",
    "write_identity_grain_artifacts",
    "write_identity_term_artifacts",
]
