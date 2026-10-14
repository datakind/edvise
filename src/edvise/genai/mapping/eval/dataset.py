"""Load Step 2a evaluation cases from the pinned reference library."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json
from edvise.genai.mapping.shared.reference_pin import resolve_sma_few_shot_pin
from edvise.genai.mapping.shared.reference_select import (
    list_active_reference_pins,
    select_reference_id,
)

LOGGER = logging.getLogger(__name__)
PROMPT_VARIANT = "batched"
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
            "Step 2a evaluation needs at least two active reference pins so each "
            "case can use another pin as the few-shot example."
        )

    cases: list[Step2aEvalContext] = []
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
        gold = load_json(str(pin.manifest_map))
        if not isinstance(contract, dict) or not isinstance(gold, dict):
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
        reference_manifest = load_json(str(few_shot.manifest_map))
        if not isinstance(reference_manifest, dict):
            raise TypeError(
                f"Few-shot manifest for {selection.reference_id!r} must be a JSON object."
            )
        found_id = reference_manifest.get("institution_id")
        if found_id != selection.reference_id:
            raise ValueError(
                "Reference manifest institution_id mismatch: "
                f"file has {found_id!r}, expected {selection.reference_id!r}."
            )
        cases.append(
            Step2aEvalContext(
                catalog=catalog_name,
                target_institution_id=reference_id,
                reference_institution_id=selection.reference_id,
                target_contract=contract,
                reference_manifest=reference_manifest,
                gold_manifest=gold,
                output_path=str(
                    pin.current_root.parent
                    / "eval_scratch"
                    / f"{reference_id}_mapping_manifest.json"
                ),
            )
        )

    if skipped:
        LOGGER.warning(
            "Skipped reference pins without %s: %s",
            _CONTRACT_NAME,
            ", ".join(skipped),
        )
    if not cases:
        raise ValueError(
            "No reference pins had both a manifest and an enriched schema contract."
        )
    return cases
