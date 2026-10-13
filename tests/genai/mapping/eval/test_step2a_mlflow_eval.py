"""Step 2a MLflow evaluation framework tests. No live model or MLflow 3 calls."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from edvise.genai.mapping.eval.dataset import (
    Step2aEvalContext,
    load_step2a_eval_context,
)
from edvise.genai.mapping.eval.models import (
    GATEWAY_MODELS_ENV,
    resolve_eval_model,
)
from edvise.genai.mapping.eval.predict import (
    PredictionObservation,
    make_step2a_predict_fn,
)
from edvise.genai.mapping.eval.run import run_step2a_evaluation
from edvise.genai.mapping.eval.scorers import assess_step2a
from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
    MappingManifestEnvelope,
)


def _mapping(
    field: str,
    *,
    table: str | None,
    column: str | None,
    join: dict | None = None,
) -> dict:
    record: dict = {
        "target_field": field,
        "source_table": table,
        "source_column": column,
        "confidence": 1.0,
    }
    if table is not None and column is not None:
        record["row_selection"] = {"strategy": "any_row"}
    if join is not None:
        record["join"] = join
    return record


def _envelope(cohort: list[dict], course: list[dict]) -> dict:
    return {
        "institution_id": "synthetic_coastal_cc",
        "pipeline_version": "eval",
        "manifests": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "raw_edvise_student",
                "mappings": cohort,
                "column_aliases": [],
            },
            "course": {
                "entity_type": "course",
                "target_schema": "raw_edvise_course",
                "mappings": course,
                "column_aliases": [],
            },
        },
    }


def _gold() -> dict:
    return _envelope(
        [
            _mapping("student_id", table="students", column="id"),
            _mapping("unused_field", table=None, column=None),
        ],
        [
            _mapping(
                "course_id",
                table="courses",
                column="id",
                join={
                    "base_table": "enrollments",
                    "lookup_table": "courses",
                    "join_keys": ["course_id"],
                },
            )
        ],
    )


def _expectations(gold: dict | None = None) -> dict:
    return {"gold_manifest": gold or _gold(), "schema_contract": None}


def test_valid_matching_manifest_scores_perfect() -> None:
    gold = _gold()
    MappingManifestEnvelope.model_validate(gold)
    assessment = assess_step2a(gold, _expectations(gold))
    assert assessment.envelope_valid == 1.0
    assert assessment.source_exact == 1.0
    assert assessment.execution_ready == 1.0
    assert assessment.diagnostic["validation_passed"] is True
    assert len(assessment.diagnostic["field_scores"]) == 3


def test_wrong_source_lowers_source_exact_more_than_unmapped_fields() -> None:
    predicted = _gold()
    predicted["manifests"]["cohort"]["mappings"][0]["source_column"] = "student_number"
    assessment = assess_step2a(predicted, _expectations())
    assert assessment.envelope_valid == 1.0
    assert assessment.source_exact == 0.5
    assert assessment.execution_ready == 0.75


def test_wrong_join_keeps_source_exact_and_lowers_execution_ready() -> None:
    predicted = _gold()
    predicted["manifests"]["course"]["mappings"][0]["join"]["join_keys"] = ["term_id"]
    assessment = assess_step2a(predicted, _expectations())
    assert assessment.envelope_valid == 1.0
    assert assessment.source_exact == 1.0
    assert assessment.execution_ready == 0.5
    course_field = next(
        row
        for row in assessment.diagnostic["field_scores"]
        if row["field"] == "course_id"
    )
    assert course_field["source_exact"] == 1
    assert course_field["execution_ready"] == 0
    assert course_field["join_keys_correct"] == 0


def test_invalid_prediction_zeros_headline_scores() -> None:
    assessment = assess_step2a({"error": "gateway timeout"}, _expectations())
    assert assessment.envelope_valid == 0.0
    assert assessment.source_exact == 0.0
    assert assessment.execution_ready == 0.0
    assert assessment.diagnostic["validation_passed"] is False
    assert "gateway timeout" in assessment.diagnostic["validation_error"]
    assert assessment.diagnostic["field_scores"] == []


def test_invalid_envelope_zeros_headline_scores_and_keeps_field_detail() -> None:
    predicted = _gold()
    del predicted["institution_id"]
    assessment = assess_step2a(predicted, _expectations())
    assert assessment.envelope_valid == 0.0
    assert assessment.source_exact == 0.0
    assert assessment.execution_ready == 0.0
    assert assessment.diagnostic["headline_metrics_zeroed"] is True
    assert assessment.diagnostic["field_scores"]


@pytest.fixture
def restore_model_env():
    keys = [
        GATEWAY_MODELS_ENV,
        "EDVISE_EVAL_MODEL_SONNET_4_5",
        "EDVISE_EVAL_MODEL_HAIKU_5",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_resolve_model_requires_a_gateway_id(restore_model_env: None) -> None:
    os.environ.pop(GATEWAY_MODELS_ENV, None)
    os.environ.pop("EDVISE_EVAL_MODEL_SONNET_4_5", None)
    with pytest.raises(ValueError, match="No gateway model id"):
        resolve_eval_model("sonnet:4.5")


def test_resolve_model_uses_explicit_id_over_environment(
    restore_model_env: None,
) -> None:
    os.environ[GATEWAY_MODELS_ENV] = json.dumps({"sonnet:4.5": "from-env-map"})
    resolved = resolve_eval_model("sonnet:4.5", gateway_model_id="from-cli")
    assert resolved.family == "sonnet"
    assert resolved.version == "4.5"
    assert resolved.gateway_model_id == "from-cli"


def test_resolve_model_reads_json_map_then_slot_env(restore_model_env: None) -> None:
    os.environ[GATEWAY_MODELS_ENV] = json.dumps({"haiku:5": "haiku-from-map"})
    mapped = resolve_eval_model("haiku:5")
    assert mapped.gateway_model_id == "haiku-from-map"

    os.environ[GATEWAY_MODELS_ENV] = "{}"
    os.environ["EDVISE_EVAL_MODEL_SONNET_4_5"] = "sonnet-from-slot"
    slotted = resolve_eval_model("Sonnet:4.5")
    assert slotted.selection == "sonnet:4.5"
    assert slotted.gateway_model_id == "sonnet-from-slot"


def test_resolve_model_rejects_unknown_selection() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_eval_model("opus:4.5")


def test_load_context_builds_one_evaluation_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gold = _gold()
    reference = {"institution_id": "synthetic_metro_research_uni", "manifests": {}}
    contract: dict = {"datasets": {}}
    files = {
        "gold": tmp_path / "manifest_map.json",
        "contract": tmp_path / "enriched_schema_contract.json",
        "reference": tmp_path / "reference_manifest_map.json",
    }
    files["gold"].write_text(json.dumps(gold))
    files["contract"].write_text(json.dumps(contract))
    files["reference"].write_text(json.dumps(reference))

    import edvise.genai.mapping.eval.dataset as dataset_mod

    monkeypatch.setattr(
        dataset_mod,
        "_resolve_gold_mapping_manifest_for_eval",
        lambda target_id: files["gold"],
    )
    monkeypatch.setattr(
        dataset_mod,
        "_resolve_target_schema_contract_for_eval",
        lambda target_id: files["contract"],
    )
    monkeypatch.setattr(
        dataset_mod,
        "_resolve_reference_mapping_manifest",
        lambda reference_id: files["reference"],
    )
    monkeypatch.setattr(
        dataset_mod,
        "_eval_run_root",
        lambda target_id: tmp_path / "eval",
    )

    context = load_step2a_eval_context(catalog="dev_sst_02")
    record = context.evaluation_record()
    assert record["inputs"] == {
        "institution_id": "synthetic_coastal_cc",
        "reference_institution_id": "synthetic_metro_research_uni",
    }
    assert record["expectations"]["gold_manifest"]["institution_id"] == (
        "synthetic_coastal_cc"
    )
    assert record["expectations"]["schema_contract"] == contract
    assert context.output_path.endswith(
        "scratch/synthetic_coastal_cc_mapping_manifest.json"
    )
    assert os.environ["EDVISE_EVAL_UC_CATALOG"] == "dev_sst_02"


def test_load_context_rejects_reference_institution_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import edvise.genai.mapping.eval.dataset as dataset_mod

    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"institution_id": "other_school"}))
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(_gold()))
    contract = tmp_path / "contract.json"
    contract.write_text("{}")
    monkeypatch.setattr(
        dataset_mod, "_resolve_gold_mapping_manifest_for_eval", lambda _id: gold
    )
    monkeypatch.setattr(
        dataset_mod, "_resolve_target_schema_contract_for_eval", lambda _id: contract
    )
    monkeypatch.setattr(
        dataset_mod, "_resolve_reference_mapping_manifest", lambda _id: reference
    )
    with pytest.raises(ValueError, match="institution_id mismatch"):
        load_step2a_eval_context(catalog="dev_sst_02")


def test_predict_fn_injects_envelope_fields_and_records_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    monkeypatch.setattr(
        predict_mod, "build_step2a_batched_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(
        predict_mod,
        "run_once",
        lambda *_args, **_kwargs: {
            "success": True,
            "response": json.dumps({"manifests": {}}),
            "latency_s": 1.25,
            "response_chars": 18,
            "error": None,
        },
    )
    context = _context()
    observation = PredictionObservation()
    predict_fn = make_step2a_predict_fn(
        context=context,
        gateway_model_id="gateway-sonnet",
        pipeline_version="v-test",
        client=object(),
        observation=observation,
    )
    predicted = predict_fn(
        institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
    )
    assert predicted["institution_id"] == "synthetic_coastal_cc"
    assert predicted["pipeline_version"] == "v-test"
    assert observation.calls[0]["latency_s"] == 1.25
    assert observation.calls[0]["response_chars"] == 18


def test_predict_fn_returns_error_payload_when_gateway_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    monkeypatch.setattr(
        predict_mod, "build_step2a_batched_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(
        predict_mod,
        "run_once",
        lambda *_args, **_kwargs: {
            "success": False,
            "response": None,
            "latency_s": 0.2,
            "response_chars": None,
            "error": "HTTP 503",
        },
    )
    predicted = make_step2a_predict_fn(
        context=_context(),
        gateway_model_id="gateway-haiku",
        client=object(),
    )(
        institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
    )
    assert predicted == {"error": "HTTP 503"}


def test_predict_fn_creates_client_without_disabling_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    seen: dict = {}

    def _client(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(
        predict_mod, "create_openai_client_for_databricks_gateway", _client
    )
    monkeypatch.setattr(
        predict_mod, "build_step2a_batched_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(
        predict_mod,
        "run_once",
        lambda *_args, **_kwargs: {
            "success": False,
            "response": None,
            "latency_s": 0.1,
            "response_chars": None,
            "error": "stopped",
        },
    )
    make_step2a_predict_fn(context=_context(), gateway_model_id="gateway-sonnet")(
        institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
    )
    assert seen == {"disable_tracing": False}


class _RunScope:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_run_logs_comparison_params_and_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod
    import edvise.genai.mapping.eval.run as run_mod

    monkeypatch.setenv(GATEWAY_MODELS_ENV, json.dumps({"sonnet:5": "gateway-sonnet-5"}))
    monkeypatch.setattr(
        run_mod, "load_step2a_eval_context", lambda **_kwargs: _context()
    )
    monkeypatch.setattr(
        predict_mod, "build_step2a_batched_prompt", lambda **_kwargs: "prompt"
    )
    monkeypatch.setattr(
        predict_mod,
        "create_openai_client_for_databricks_gateway",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        predict_mod,
        "run_once",
        lambda *_args, **_kwargs: {
            "success": True,
            "response": json.dumps(_gold()),
            "latency_s": 2.5,
            "response_chars": 40,
            "error": None,
        },
    )

    logged: dict = {"params": {}, "metrics": {}, "artifacts": {}, "experiment": None}

    def evaluate(*, data, predict_fn, scorers):
        row = data[0]
        output = predict_fn(**row["inputs"])
        logged["scores"] = [
            scorer(outputs=output, expectations=row["expectations"])
            for scorer in scorers
        ]
        return SimpleNamespace(run_id="run-123")

    mlflow = SimpleNamespace(
        genai=SimpleNamespace(
            evaluate=evaluate,
            scorers=SimpleNamespace(scorer=lambda fn: fn),
        ),
        set_experiment=lambda name: logged.__setitem__("experiment", name),
        start_run=lambda run_id: _RunScope(),
        log_params=lambda params: logged["params"].update(params),
        log_metric=lambda key, value: logged["metrics"].__setitem__(key, value),
        log_dict=lambda payload, path: logged["artifacts"].__setitem__(path, payload),
    )

    run_id = run_step2a_evaluation(
        catalog="dev_sst_02",
        model="sonnet:5",
        experiment_name="/Shared/edvise/genai_mapping/step2a_model_comparison",
        mlflow_module=mlflow,
    )
    assert run_id == "run-123"
    assert (
        logged["experiment"] == "/Shared/edvise/genai_mapping/step2a_model_comparison"
    )
    assert logged["scores"] == [1.0, 1.0, 1.0]
    assert logged["params"] == {
        "model_family": "sonnet",
        "model_version": "5",
        "gateway_model_id": "gateway-sonnet-5",
        "prompt_variant": "batched",
        "institution_id": "synthetic_coastal_cc",
        "reference_institution_id": "synthetic_metro_research_uni",
        "catalog": "dev_sst_02",
    }
    assert logged["metrics"] == {"latency_s": 2.5, "response_chars": 40.0}
    artifact = logged["artifacts"]["diagnostics/step2a_field_detail.json"]
    assert artifact["cases"][0]["validation_passed"] is True
    assert len(artifact["cases"][0]["field_scores"]) == 3
    assert artifact["predictions"][0]["latency_s"] == 2.5


def _context() -> Step2aEvalContext:
    return Step2aEvalContext(
        catalog="dev_sst_02",
        target_institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
        target_contract={"datasets": {}},
        reference_manifest={"institution_id": "synthetic_metro_research_uni"},
        gold_manifest=_gold(),
        output_path="/Volumes/dev_sst_02/synthetic_coastal_cc_silver/scratch/manifest.json",
    )
