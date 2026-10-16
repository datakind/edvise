"""Run the production Step 2a batched prompt for one gateway model."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
from edvise.genai.mapping.eval.dataset import Step2aEvalContext
from edvise.genai.mapping.schema_mapping_agent.manifest.eval import run_once
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    build_step2a_batched_prompt,
)
from edvise.genai.mapping.shared.databricks_ai_gateway import (
    create_openai_client_for_databricks_gateway,
)


@dataclass
class PredictionObservation:
    """Latency, response size, and predicted manifests captured while ``predict_fn`` runs."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    manifests: list[dict[str, Any]] = field(default_factory=list)


def make_step2a_predict_fn(
    *,
    cases: list[Step2aEvalContext],
    gateway_model_id: str,
    pipeline_version: str = "eval",
    client: Any | None = None,
    observation: PredictionObservation | None = None,
) -> Any:
    """Return the ``predict_fn`` ``mlflow.genai.evaluate`` calls once per record.

    The returned manifest has ``institution_id`` and ``pipeline_version`` filled
    in the same way onboard does before envelope validation. A failed call
    returns an error object so the scorers can record zeros instead of raising.
    """
    by_institution = {case.target_institution_id: case for case in cases}
    holder: dict[str, Any] = {"client": client}

    def predict_fn(
        institution_id: str, reference_institution_id: str
    ) -> dict[str, Any]:
        context = by_institution.get(institution_id)
        if context is None:
            raise ValueError(
                f"Evaluation record institution_id {institution_id!r} is not one of "
                f"the loaded reference cases: {sorted(by_institution)}."
            )
        if reference_institution_id != context.reference_institution_id:
            raise ValueError(
                "Evaluation record reference_institution_id "
                f"{reference_institution_id!r} does not match loaded context "
                f"{context.reference_institution_id!r}."
            )
        prompt = build_step2a_batched_prompt(
            institution_id=institution_id,
            output_path=context.output_path,
            institution_schema_contract=context.target_contract,
            reference_manifests=[context.reference_manifest],
            reference_institution_ids=[reference_institution_id],
            cohort_schema_class=RawEdviseStudentDataSchema,
            course_schema_class=RawEdviseCourseDataSchema,
        )
        if holder["client"] is None:
            holder["client"] = create_openai_client_for_databricks_gateway(
                disable_tracing=False
            )
        result = run_once(gateway_model_id, prompt, holder["client"])
        predicted = _manifest_from_call(
            result,
            institution_id=institution_id,
            pipeline_version=pipeline_version,
        )
        if observation is not None:
            observation.calls.append(
                {
                    "institution_id": institution_id,
                    "reference_institution_id": reference_institution_id,
                    "latency_s": result.get("latency_s"),
                    "response_chars": result.get("response_chars"),
                    "error": result.get("error"),
                    "reasoning": result.get("reasoning") or [],
                }
            )
            observation.manifests.append(
                {"institution_id": institution_id, "manifest": predicted}
            )
        return predicted

    return predict_fn


def _manifest_from_call(
    result: dict[str, Any],
    *,
    institution_id: str,
    pipeline_version: str,
) -> dict[str, Any]:
    """Turn one gateway call into the manifest the scorers receive."""
    if not result.get("success"):
        return {"error": result.get("error") or "Step 2a prediction failed"}
    raw = result.get("response")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        return {"error": str(exc), "response": raw}
    if not isinstance(parsed, dict):
        return {
            "error": "Step 2a prediction was not a JSON object",
            "response": raw,
        }
    parsed["institution_id"] = institution_id
    parsed["pipeline_version"] = pipeline_version or "eval"
    return parsed
