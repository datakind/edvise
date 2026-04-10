"""
On-disk Identity Agent outputs in the shapes expected by :mod:`hitl.resolver`.

Grain / term *contract* JSON uses ``datasets[table].grain_contract`` and
``datasets[table].term_config`` so ``resolve_items`` / ``apply_hook_spec`` can
navigate configs consistently.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from edvise.genai.mapping.identity_agent.grain_inference.schemas import GrainContract
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    HITLItem,
    InstitutionHITLItems,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import TermContract


def build_grain_config_for_resolver(
    institution_id: str,
    contracts_by_dataset: Mapping[str, GrainContract],
) -> dict:
    """
    Resolver-shaped JSON: ``datasets[table].grain_contract`` holds the grain contract.
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


def unique_hitl_items_by_item_id(items: list[HITLItem]) -> list[HITLItem]:
    """Return items in order, keeping the first occurrence of each ``item_id``."""
    seen: set[str] = set()
    out: list[HITLItem] = []
    for it in items:
        if it.item_id in seen:
            continue
        seen.add(it.item_id)
        out.append(it)
    return out


def load_grain_contracts_from_resolver_config(
    config_path: str | Path,
    *,
    expected_institution_id: str | None = None,
) -> dict[str, GrainContract]:
    """
    Load per-dataset :class:`GrainContract` values from resolver-shaped JSON
    (e.g. ``identity_grain_output.json`` written by :func:`write_identity_grain_artifacts`
    and updated by :func:`resolve_items`).

    Parameters
    ----------
    config_path:
        Path to JSON with ``datasets[table].grain_contract``.
    expected_institution_id:
        If set, must match the file's top-level ``institution_id``.
    """
    path = Path(config_path)
    raw = json.loads(path.read_text())
    inst = raw.get("institution_id")
    if expected_institution_id is not None and inst != expected_institution_id:
        raise ValueError(
            f"{path.name}: institution_id {inst!r} != expected {expected_institution_id!r}"
        )
    datasets = raw.get("datasets") or {}
    out: dict[str, GrainContract] = {}
    for name, payload in datasets.items():
        if not isinstance(payload, dict):
            raise TypeError(f"{path.name}: datasets[{name!r}] must be an object, got {type(payload)}")
        gc_raw = payload.get("grain_contract")
        if gc_raw is None:
            raise ValueError(f"{path.name}: missing grain_contract for dataset {name!r}")
        out[name] = GrainContract.model_validate(gc_raw)
    return out


def load_term_contracts_from_resolver_config(
    config_path: str | Path,
    *,
    expected_institution_id: str | None = None,
) -> dict[str, TermContract]:
    """
    Load per-dataset :class:`TermContract` values from resolver-shaped JSON
    (e.g. ``identity_term_output.json`` from :func:`write_identity_term_artifacts` / ``resolve_items``).

    Parameters
    ----------
    config_path:
        Path to JSON whose ``datasets[table]`` objects are full term contract dumps.
    expected_institution_id:
        If set, must match the file's top-level ``institution_id``.
    """
    path = Path(config_path)
    raw = json.loads(path.read_text())
    inst = raw.get("institution_id")
    if expected_institution_id is not None and inst != expected_institution_id:
        raise ValueError(
            f"{path.name}: institution_id {inst!r} != expected {expected_institution_id!r}"
        )
    datasets = raw.get("datasets") or {}
    out: dict[str, TermContract] = {}
    for name, payload in datasets.items():
        out[name] = TermContract.model_validate(payload)
    return out


__all__ = [
    "build_grain_config_for_resolver",
    "build_term_config_for_resolver",
    "load_grain_contracts_from_resolver_config",
    "load_term_contracts_from_resolver_config",
    "unique_hitl_items_by_item_id",
    "write_identity_grain_artifacts",
    "write_identity_term_artifacts",
]
