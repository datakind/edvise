"""Load Step 2a and Step 2b evaluation cases from the pinned reference library."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json
from edvise.genai.mapping.shared.reference_pin import resolve_sma_few_shot_pin
from edvise.genai.mapping.shared.reference_select import (
    list_active_reference_pins,
    select_reference_id,
)

LOGGER = logging.getLogger(__name__)
PROMPT_VARIANT = "batched"
STEP2B_PROMPT_VARIANT = "production"
_CONTRACT_NAME = "enriched_schema_contract.json"


@dataclass(frozen=True)
class Step2aEvalContext:
    """Gold, contract, and few-shot reference for one pinned reference."""

    catalog: str
    target_institution_id: str
    reference_institution_id: str
    target_contract: dict[str, Any]
    reference_manifest: dict[str, Any]
    gold_manifest: dict[str, Any]
    output_path: str

    def evaluation_record(self) -> dict[str, Any]:
        """One ``mlflow.genai.evaluate`` record.

        ``inputs`` are the arguments of ``predict_fn``. ``expectations`` hold
        the gold the scorers compare against.
        """
        return {
            "inputs": {
                "institution_id": self.target_institution_id,
                "reference_institution_id": self.reference_institution_id,
            },
            "expectations": {
                "gold_manifest": self.gold_manifest,
                "schema_contract": self.target_contract,
            },
        }


@dataclass(frozen=True)
class Step2bEvalContext:
    """Gold transformation map and the manifest the production prompt reads."""

    catalog: str
    target_institution_id: str
    reference_institution_id: str
    target_contract: dict[str, Any]
    mapping_manifest: dict[str, Any]
    reference_transformation_map: dict[str, Any]
    gold_transformation_map: dict[str, Any]
    output_path: str

    def evaluation_record(self) -> dict[str, Any]:
        return {
            "inputs": {
                "institution_id": self.target_institution_id,
                "reference_institution_id": self.reference_institution_id,
            },
            "expectations": {
                "gold_transformation_map": self.gold_transformation_map,
            },
        }


@dataclass(frozen=True)
class _ReferencePair:
    catalog: str
    target_institution_id: str
    reference_institution_id: str
    target_contract: dict[str, Any]
    target_root: Path
    target_manifest_path: Path
    target_transformation_map_path: Path
    few_shot_manifest_path: Path
    few_shot_transformation_map_path: Path


def load_step2a_reference_cases(
    *,
    catalog: str,
    active_pins: list[dict[str, Any]] | None = None,
) -> list[Step2aEvalContext]:
    """Build one case per active reference pin.

    Each pin's ``manifest_map.json`` is the gold. The few-shot example is the
    other pin ``select_reference_id`` would give that institution on onboard.
    Pins without an enriched schema contract are skipped.
    """
    cases: list[Step2aEvalContext] = []
    for pair in _load_reference_pairs(
        catalog=catalog, active_pins=active_pins, step_label="Step 2a"
    ):
        gold = _json_object(
            pair.target_manifest_path,
            label=f"Reference {pair.target_institution_id!r} manifest",
        )
        reference_manifest = _json_object(
            pair.few_shot_manifest_path,
            label=f"Few-shot manifest for {pair.reference_institution_id!r}",
        )
        _require_institution_id(
            reference_manifest,
            expected=pair.reference_institution_id,
            label="Reference manifest",
        )
        cases.append(
            Step2aEvalContext(
                catalog=pair.catalog,
                target_institution_id=pair.target_institution_id,
                reference_institution_id=pair.reference_institution_id,
                target_contract=pair.target_contract,
                reference_manifest=reference_manifest,
                gold_manifest=gold,
                output_path=str(
                    pair.target_root.parent
                    / "eval_scratch"
                    / f"{pair.target_institution_id}_mapping_manifest.json"
                ),
            )
        )
    return cases


def load_step2b_reference_cases(
    *,
    catalog: str,
    active_pins: list[dict[str, Any]] | None = None,
) -> list[Step2bEvalContext]:
    """Build one Step 2b case per active reference pin.

    The pin's approved ``manifest_map.json`` is the prompt input and its
    ``transformation_map.json`` is the gold. The few-shot example is the other
    pin's transformation map, chosen the same way onboard chooses a reference.
    Reference pins do not carry ``identity_term_output.json``, so the prompt
    omits term config.
    """
    cases: list[Step2bEvalContext] = []
    for pair in _load_reference_pairs(
        catalog=catalog, active_pins=active_pins, step_label="Step 2b"
    ):
        mapping_manifest = _json_object(
            pair.target_manifest_path,
            label=f"Mapping manifest for {pair.target_institution_id!r}",
        )
        gold_transformation_map = _json_object(
            pair.target_transformation_map_path,
            label=f"Gold transformation map for {pair.target_institution_id!r}",
        )
        reference_transformation_map = _json_object(
            pair.few_shot_transformation_map_path,
            label=(
                f"Reference transformation map for {pair.reference_institution_id!r}"
            ),
        )
        _require_institution_id(
            mapping_manifest,
            expected=pair.target_institution_id,
            label="Mapping manifest",
        )
        _require_institution_id(
            gold_transformation_map,
            expected=pair.target_institution_id,
            label="Gold transformation map",
        )
        _require_institution_id(
            reference_transformation_map,
            expected=pair.reference_institution_id,
            label="Reference transformation map",
        )
        cases.append(
            Step2bEvalContext(
                catalog=pair.catalog,
                target_institution_id=pair.target_institution_id,
                reference_institution_id=pair.reference_institution_id,
                target_contract=pair.target_contract,
                mapping_manifest=mapping_manifest,
                reference_transformation_map=reference_transformation_map,
                gold_transformation_map=gold_transformation_map,
                output_path=str(
                    pair.target_root.parent
                    / "eval_scratch"
                    / f"{pair.target_institution_id}_transformation_map.json"
                ),
            )
        )
    return cases


def _load_reference_pairs(
    *,
    catalog: str,
    active_pins: list[dict[str, Any]] | None,
    step_label: str,
) -> list[_ReferencePair]:
    """Pair each active pin with the other pin onboard would use as few-shot."""
    catalog_name = catalog.strip()
    if not catalog_name:
        raise ValueError("catalog must be non-empty")
    os.environ["EDVISE_EVAL_UC_CATALOG"] = catalog_name
    pins = (
        list(active_pins)
        if active_pins is not None
        else list_active_reference_pins(catalog_name)
    )
    if len(pins) < 2:
        raise ValueError(
            f"{step_label} evaluation needs at least two active reference pins so each "
            "case can use another pin as the few-shot example."
        )

    pairs: list[_ReferencePair] = []
    skipped: list[str] = []
    for row in pins:
        reference_id = str(row.get("reference_id") or "").strip()
        if not reference_id:
            continue
        pin = resolve_sma_few_shot_pin(reference_id, catalog=catalog_name)
        contract_path = pin.current_root / _CONTRACT_NAME
        if not contract_path.is_file():
            skipped.append(reference_id)
            continue
        contract = load_json(str(contract_path))
        if not isinstance(contract, dict):
            raise TypeError(
                f"Reference {reference_id!r} contract and manifest must be JSON objects."
            )
        selection = select_reference_id(
            catalog=catalog_name,
            institution_id=reference_id,
            query_contract=contract,
            active_pins=pins,
        )
        few_shot = resolve_sma_few_shot_pin(
            selection.reference_id, catalog=catalog_name
        )
        pairs.append(
            _ReferencePair(
                catalog=catalog_name,
                target_institution_id=reference_id,
                reference_institution_id=selection.reference_id,
                target_contract=contract,
                target_root=pin.current_root,
                target_manifest_path=pin.manifest_map,
                target_transformation_map_path=_artifact_path(
                    pin, "transformation_map.json"
                ),
                few_shot_manifest_path=few_shot.manifest_map,
                few_shot_transformation_map_path=_artifact_path(
                    few_shot, "transformation_map.json"
                ),
            )
        )

    if skipped:
        LOGGER.warning(
            "Skipped reference pins without %s: %s",
            _CONTRACT_NAME,
            ", ".join(skipped),
        )
    if not pairs:
        raise ValueError(
            "No reference pins had both a manifest and an enriched schema contract."
        )
    return pairs


def _artifact_path(pin: Any, name: str) -> Path:
    """Pinned artifact path, defaulting to ``current/<name>`` when unset.

    Step 2a fixtures only populate the manifest. Step 2b reads the
    transformation map from the same pin object when it is present.
    """
    attr = name.removesuffix(".json")
    path = getattr(pin, attr, None)
    if isinstance(path, Path):
        return path
    return Path(str(pin.current_root)) / name


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = load_json(str(path))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object.")
    return payload


def _require_institution_id(
    payload: dict[str, Any], *, expected: str, label: str
) -> None:
    found = payload.get("institution_id")
    if found != expected:
        raise ValueError(
            f"{label} institution_id mismatch: file has {found!r}, expected {expected!r}."
        )
