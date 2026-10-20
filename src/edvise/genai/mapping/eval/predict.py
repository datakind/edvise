"""Run the production Step 2a and Step 2b prompts for one gateway model."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema
from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
from edvise.genai.mapping.eval.dataset import Step2aEvalContext, Step2bEvalContext
from edvise.genai.mapping.eval.gateway import run_once
from edvise.genai.mapping.schema_mapping_agent.manifest.prompts import (
    build_step2a_batched_prompt,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
    dedupe_transformation_plans_in_wrapper,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.prompt import (
    build_step2b_prompt,
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
    return _make_predict_fn(
        cases=cases,
        gateway_model_id=gateway_model_id,
        pipeline_version=pipeline_version,
        client=client,
        observation=observation,
        label="Step 2a",
        artifact_key="manifest",
        prompt_for=_step2a_prompt,
        finalize=lambda predicted: predicted,
    )


def make_step2b_predict_fn(
    *,
    cases: list[Step2bEvalContext],
    gateway_model_id: str,
    pipeline_version: str = "eval",
    client: Any | None = None,
    observation: PredictionObservation | None = None,
) -> Any:
    """Return the ``predict_fn`` for one Step 2b model.

    The returned map has ``institution_id`` and ``pipeline_version`` filled in,
    then duplicate plans removed, the same way onboard does before validation.
    """
    return _make_predict_fn(
        cases=cases,
        gateway_model_id=gateway_model_id,
        pipeline_version=pipeline_version,
        client=client,
        observation=observation,
        label="Step 2b",
        artifact_key="transformation_map",
        prompt_for=_step2b_prompt,
        finalize=_finalize_step2b,
    )


def _make_predict_fn(
    *,
    cases: Sequence[Any],
    gateway_model_id: str,
    pipeline_version: str,
    client: Any | None,
    observation: PredictionObservation | None,
    label: str,
    artifact_key: str,
    prompt_for: Callable[[Any, str], str],
    finalize: Callable[[dict[str, Any]], dict[str, Any]],
) -> Any:
    by_institution = {case.target_institution_id: case for case in cases}
    holder: dict[str, Any] = {"client": client}

    def predict_fn(
        institution_id: str, reference_institution_id: str
    ) -> dict[str, Any]:
        context = _require_context(
            by_institution, institution_id, reference_institution_id
        )
        prompt = prompt_for(context, reference_institution_id)
        if holder["client"] is None:
            holder["client"] = create_openai_client_for_databricks_gateway(
                disable_tracing=False
            )
        result = run_once(gateway_model_id, prompt, holder["client"])
        predicted = finalize(
            _object_from_call(
                result,
                institution_id=institution_id,
                pipeline_version=pipeline_version,
                label=label,
            )
        )
        _record_prediction(
            observation,
            institution_id=institution_id,
            reference_institution_id=reference_institution_id,
            result=result,
            artifact_key=artifact_key,
            artifact=predicted,
        )
        return predicted

    return predict_fn


def _step2a_prompt(context: Step2aEvalContext, reference_institution_id: str) -> str:
    return build_step2a_batched_prompt(
        institution_id=context.target_institution_id,
        output_path=context.output_path,
        institution_schema_contract=context.target_contract,
        reference_manifests=[context.reference_manifest],
        reference_institution_ids=[reference_institution_id],
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
    )


def _step2b_prompt(context: Step2bEvalContext, reference_institution_id: str) -> str:
    return build_step2b_prompt(
        institution_id=context.target_institution_id,
        output_path=context.output_path,
        institution_mapping_manifest=context.mapping_manifest,
        institution_schema_contract=context.target_contract,
        cohort_schema_class=RawEdviseStudentDataSchema,
        course_schema_class=RawEdviseCourseDataSchema,
        reference_transformation_maps=[context.reference_transformation_map],
        reference_institution_ids=[reference_institution_id],
    )


def _finalize_step2b(predicted: dict[str, Any]) -> dict[str, Any]:
    if "transformation_maps" in predicted:
        dedupe_transformation_plans_in_wrapper(predicted)
    return predicted


def _require_context(
    by_institution: dict[str, Any],
    institution_id: str,
    reference_institution_id: str,
) -> Any:
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
    return context


def _record_prediction(
    observation: PredictionObservation | None,
    *,
    institution_id: str,
    reference_institution_id: str,
    result: dict[str, Any],
    artifact_key: str,
    artifact: dict[str, Any],
) -> None:
    if observation is None:
        return
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
        {"institution_id": institution_id, artifact_key: artifact}
    )


def _object_from_call(
    result: dict[str, Any],
    *,
    institution_id: str,
    pipeline_version: str,
    label: str,
) -> dict[str, Any]:
    if not result.get("success"):
        return {"error": result.get("error") or f"{label} prediction failed"}
    raw = result.get("response")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        return {"error": str(exc), "response": raw}
    if not isinstance(parsed, dict):
        return {"error": f"{label} prediction was not a JSON object", "response": raw}
    parsed["institution_id"] = institution_id
    parsed["pipeline_version"] = pipeline_version or "eval"
    return parsed
