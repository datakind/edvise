"""Decision metrics for raw Step 2a manifests and Step 2b transformation maps.

These functions stay independent of ``mlflow.genai`` so they can be tested on
MLflow 2.22. The runner adapts them to MLflow scorers at execution time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from edvise.genai.mapping.schema_mapping_agent.manifest.eval import (
    score_result,
    validate_envelope_dict,
)
from edvise.genai.mapping.schema_mapping_agent.transformation.eval import (
    score_t2b_result,
)


@dataclass(frozen=True)
class Step2aAssessment:
    envelope_valid: float
    source_exact: float
    execution_ready: float
    diagnostic: dict[str, Any]


def assess_step2a(outputs: Any, expectations: dict[str, Any]) -> Step2aAssessment:
    """Score one prediction against the gold manifest.

    ``envelope_valid`` checks the manifest shape. When that fails, the other
    two headline scores are zero and the validation error stays in
    ``diagnostic``. A valid manifest still reports ``source_exact`` over
    gold-mappable fields and ``execution_ready`` over every gold field.
    """
    gold_manifest = expectations["gold_manifest"]
    schema_contract = expectations.get("schema_contract")
    institution_id = _institution_id(gold_manifest)
    manifest, parse_error = _coerce_manifest(outputs)
    if manifest is None:
        return _failed_assessment(
            parse_error or "prediction was not a manifest",
            institution_id=institution_id,
        )

    is_valid, validation_error = validate_envelope_dict(manifest)
    scores = score_result(
        {"model": "step2a", "success": True, "response": json.dumps(manifest)},
        gold_manifest,
        schema_contract=schema_contract,
    )
    diagnostic = _diagnostic(
        scores,
        validation_passed=is_valid,
        validation_error=validation_error,
        institution_id=_institution_id(gold_manifest),
    )
    if not is_valid or scores is None:
        diagnostic["headline_metrics_zeroed"] = True
        return Step2aAssessment(0.0, 0.0, 0.0, diagnostic)

    overall = scores["overall"]
    source_exact, source_undefined = _rate(
        overall.get("source_exact_accuracy_gold_mappable")
    )
    execution_ready, execution_undefined = _rate(overall.get("execution_ready_rate"))
    diagnostic["source_exact_undefined"] = source_undefined
    diagnostic["execution_ready_undefined"] = execution_undefined
    diagnostic["headline_metrics_zeroed"] = False
    return Step2aAssessment(1.0, source_exact, execution_ready, diagnostic)


@dataclass(frozen=True)
class Step2bAssessment:
    transformation_valid: float
    execution_exact: float
    execution_safe: float
    diagnostic: dict[str, Any]


def assess_step2b(outputs: Any, expectations: dict[str, Any]) -> Step2bAssessment:
    """Score one transformation map against the gold map.

    ``transformation_valid`` checks the wrapper shape. When that fails, the
    headline rates are zero and the validation error stays in ``diagnostic``.
    A valid map still reports ``execution_exact`` and ``execution_safe``.
    """
    gold = expectations["gold_transformation_map"]
    institution_id = _institution_id(gold)
    parsed, parse_error = _coerce_object(
        outputs,
        artifact_key="transformation_maps",
        label="transformation map",
    )
    if parsed is None:
        return _failed_step2b(
            parse_error or "prediction was not a transformation map",
            institution_id=institution_id,
        )

    scores = score_t2b_result(
        {"model": "step2b", "success": True, "response": json.dumps(parsed)},
        gold,
    )
    if scores is None:
        return _failed_step2b(
            "scoring produced no result", institution_id=institution_id
        )
    diagnostic = _step2b_diagnostic(scores, institution_id=institution_id)
    if not scores.get("validation_passed"):
        diagnostic["headline_metrics_zeroed"] = True
        return Step2bAssessment(0.0, 0.0, 0.0, diagnostic)

    overall = scores["overall"]
    execution_exact, exact_undefined = _rate(overall.get("execution_exact_rate"))
    execution_safe, safe_undefined = _rate(overall.get("execution_safe_rate"))
    diagnostic["execution_exact_undefined"] = exact_undefined
    diagnostic["execution_safe_undefined"] = safe_undefined
    diagnostic["headline_metrics_zeroed"] = False
    return Step2bAssessment(1.0, execution_exact, execution_safe, diagnostic)


def _coerce_manifest(outputs: Any) -> tuple[dict[str, Any] | None, str | None]:
    parsed = outputs
    if isinstance(outputs, str):
        try:
            parsed = json.loads(outputs)
        except json.JSONDecodeError as exc:
            return None, str(exc)
    if not isinstance(parsed, dict):
        return (
            None,
            f"prediction was {type(parsed).__name__}, expected a manifest object",
        )
    if parsed.get("error") and "manifests" not in parsed:
        return None, str(parsed.get("error"))
    return parsed, None


def _failed_assessment(
    error: str, *, institution_id: str | None = None
) -> Step2aAssessment:
    return Step2aAssessment(
        0.0,
        0.0,
        0.0,
        {
            "validation_passed": False,
            "validation_error": error,
            "headline_metrics_zeroed": True,
            "source_exact_undefined": False,
            "execution_ready_undefined": False,
            "institution_id": institution_id,
            "overall": None,
            "cohort": None,
            "course": None,
            "field_scores": [],
        },
    )


def _diagnostic(
    scores: dict[str, Any] | None,
    *,
    validation_passed: bool,
    validation_error: str | None,
    institution_id: str | None,
) -> dict[str, Any]:
    if not scores:
        return _failed_assessment(
            validation_error or "scoring produced no result",
            institution_id=institution_id,
        ).diagnostic
    field_scores: list[dict[str, Any]] = []
    entities: dict[str, Any] = {}
    for entity in ("cohort", "course"):
        block = dict(scores[entity])
        field_scores.extend(block.pop("field_scores", []))
        entities[entity] = block
    return {
        "validation_passed": validation_passed,
        "validation_error": validation_error,
        "headline_metrics_zeroed": not validation_passed,
        "source_exact_undefined": False,
        "execution_ready_undefined": False,
        "institution_id": institution_id,
        "overall": scores["overall"],
        **entities,
        "field_scores": field_scores,
    }


def _failed_step2b(
    error: str, *, institution_id: str | None = None
) -> Step2bAssessment:
    return Step2bAssessment(
        0.0,
        0.0,
        0.0,
        {
            "validation_passed": False,
            "validation_error": error,
            "headline_metrics_zeroed": True,
            "execution_exact_undefined": False,
            "execution_safe_undefined": False,
            "institution_id": institution_id,
            "overall": None,
            "cohort": None,
            "course": None,
            "field_scores": [],
        },
    )


def _step2b_diagnostic(
    scores: dict[str, Any], *, institution_id: str | None
) -> dict[str, Any]:
    field_scores: list[dict[str, Any]] = []
    entities: dict[str, Any] = {}
    for entity in ("cohort", "course"):
        block = dict(scores[entity])
        field_scores.extend(block.pop("field_scores", []))
        entities[entity] = block
    return {
        "validation_passed": scores.get("validation_passed"),
        "validation_error": scores.get("validation_error"),
        "headline_metrics_zeroed": not scores.get("validation_passed"),
        "execution_exact_undefined": False,
        "execution_safe_undefined": False,
        "institution_id": institution_id,
        "overall": scores["overall"],
        **entities,
        "field_scores": field_scores,
    }


def _coerce_object(
    outputs: Any, *, artifact_key: str, label: str
) -> tuple[dict[str, Any] | None, str | None]:
    parsed = outputs
    if isinstance(outputs, str):
        try:
            parsed = json.loads(outputs)
        except json.JSONDecodeError as exc:
            return None, str(exc)
    if not isinstance(parsed, dict):
        return None, f"prediction was {type(parsed).__name__}, expected a {label}"
    if parsed.get("error") and artifact_key not in parsed:
        return None, str(parsed.get("error"))
    return parsed, None


def _institution_id(gold_manifest: Any) -> str | None:
    if isinstance(gold_manifest, dict):
        value = gold_manifest.get("institution_id")
        if isinstance(value, str):
            return value
    return None


def _rate(value: Any) -> tuple[float, bool]:
    if value is None:
        return 0.0, True
    return float(value), False
