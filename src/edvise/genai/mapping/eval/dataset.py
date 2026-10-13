"""Build one Step 2a evaluation record from the existing Unity Catalog gold files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from edvise.genai.mapping.schema_mapping_agent.manifest.eval import (
    _eval_run_root,
    _resolve_gold_mapping_manifest_for_eval,
    _resolve_reference_mapping_manifest,
    _resolve_target_schema_contract_for_eval,
)
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import load_json

DEFAULT_TARGET_INSTITUTION_ID = "synthetic_coastal_cc"
DEFAULT_REFERENCE_INSTITUTION_ID = "synthetic_metro_research_uni"
PROMPT_VARIANT = "batched"


@dataclass(frozen=True)
class Step2aEvalContext:
    """Closed-over prompt inputs plus the record passed to ``mlflow.genai.evaluate``."""

    catalog: str
    target_institution_id: str
    reference_institution_id: str
    target_contract: dict[str, Any]
    reference_manifest: dict[str, Any]
    gold_manifest: dict[str, Any]
    output_path: str

    def evaluation_record(self) -> dict[str, Any]:
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


def load_step2a_eval_context(
    *,
    catalog: str,
    target_institution_id: str = DEFAULT_TARGET_INSTITUTION_ID,
    reference_institution_id: str = DEFAULT_REFERENCE_INSTITUTION_ID,
) -> Step2aEvalContext:
    """Load gold, the target contract, and the pinned reference manifest.

    Sets ``EDVISE_EVAL_UC_CATALOG`` because the shared path resolvers read the
    catalog from the environment.
    """
    catalog_name = catalog.strip()
    if not catalog_name:
        raise ValueError("catalog is required")
    os.environ["EDVISE_EVAL_UC_CATALOG"] = catalog_name

    gold_path = _resolve_gold_mapping_manifest_for_eval(target_institution_id)
    contract_path = _resolve_target_schema_contract_for_eval(target_institution_id)
    reference_path = _resolve_reference_mapping_manifest(reference_institution_id)

    gold_manifest = load_json(str(gold_path))
    target_contract = load_json(str(contract_path))
    reference_manifest = load_json(str(reference_path))
    manifest_institution_id = reference_manifest.get("institution_id")
    if manifest_institution_id != reference_institution_id:
        raise ValueError(
            "Reference manifest institution_id mismatch. "
            f"Expected {reference_institution_id!r} but {reference_path} "
            f"contains {manifest_institution_id!r}."
        )

    output_path = str(
        _eval_run_root(target_institution_id)
        / "scratch"
        / f"{target_institution_id}_mapping_manifest.json"
    )
    return Step2aEvalContext(
        catalog=catalog_name,
        target_institution_id=target_institution_id,
        reference_institution_id=reference_institution_id,
        target_contract=target_contract,
        reference_manifest=reference_manifest,
        gold_manifest=gold_manifest,
        output_path=output_path,
    )
