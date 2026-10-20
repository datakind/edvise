"""Step 2b MLflow evaluation tests. No live model or MLflow 3 calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from edvise.genai.mapping.eval.dataset import (
    Step2bEvalContext,
    load_step2b_reference_cases,
)
from edvise.genai.mapping.eval.predict import (
    PredictionObservation,
    make_step2b_predict_fn,
)
from edvise.genai.mapping.eval.run import (
    EvalStep,
    run_evaluation,
    run_step2b_evaluation,
    supported_eval_steps,
)
from edvise.genai.mapping.eval.scorers import assess_step2b


def _step(function_name: str, column: str) -> dict:
    return {"function_name": function_name, "column": column}


def _plan(field: str, *steps: dict) -> dict:
    return {"target_field": field, "steps": list(steps)}


def _transformation_map(
    *,
    institution_id: str = "synthetic_coastal_cc",
    cohort: list[dict] | None = None,
    course: list[dict] | None = None,
) -> dict:
    return {
        "institution_id": institution_id,
        "pipeline_version": "eval",
        "transformation_maps": {
            "cohort": {
                "entity_type": "cohort",
                "target_schema": "raw_edvise_student",
                "plans": cohort or [_plan("student_id", _step("cast_string", "id"))],
            },
            "course": {
                "entity_type": "course",
                "target_schema": "raw_edvise_course",
                "plans": course
                or [_plan("course_id", _step("cast_string", "course_code"))],
            },
        },
    }


def _expectations(gold: dict | None = None) -> dict:
    return {"gold_transformation_map": gold or _transformation_map()}


def test_matching_transformation_map_scores_perfect() -> None:
    gold = _transformation_map()
    assessment = assess_step2b(gold, _expectations(gold))
    assert assessment.transformation_valid == 1.0
    assert assessment.execution_exact == 1.0
    assert assessment.execution_safe == 1.0
    assert assessment.diagnostic["validation_passed"] is True
    assert len(assessment.diagnostic["field_scores"]) == 2


def test_extra_step_stays_execution_safe_and_lowers_exact() -> None:
    predicted = _transformation_map(
        cohort=[
            _plan(
                "student_id",
                _step("cast_string", "id"),
                _step("lowercase", "id"),
            )
        ]
    )
    assessment = assess_step2b(predicted, _expectations())
    assert assessment.transformation_valid == 1.0
    assert assessment.execution_exact == 0.5
    assert assessment.execution_safe == 1.0


def test_wrong_column_lowers_execution_safe() -> None:
    predicted = _transformation_map(
        cohort=[_plan("student_id", _step("cast_string", "student_number"))]
    )
    assessment = assess_step2b(predicted, _expectations())
    assert assessment.transformation_valid == 1.0
    assert assessment.execution_exact == 0.5
    assert assessment.execution_safe == 0.5
    cohort_field = next(
        row
        for row in assessment.diagnostic["field_scores"]
        if row["field"] == "student_id"
    )
    assert cohort_field["execution_exact"] == 0
    assert cohort_field["execution_safe"] == 0


def test_invalid_prediction_zeros_headline_scores() -> None:
    assessment = assess_step2b({"error": "gateway timeout"}, _expectations())
    assert assessment.transformation_valid == 0.0
    assert assessment.execution_exact == 0.0
    assert assessment.execution_safe == 0.0
    assert assessment.diagnostic["validation_passed"] is False
    assert "gateway timeout" in assessment.diagnostic["validation_error"]
    assert assessment.diagnostic["institution_id"] == "synthetic_coastal_cc"
    assert assessment.diagnostic["field_scores"] == []


def test_invalid_wrapper_zeros_headline_scores_and_keeps_field_detail() -> None:
    predicted = _transformation_map()
    del predicted["institution_id"]
    assessment = assess_step2b(predicted, _expectations())
    assert assessment.transformation_valid == 0.0
    assert assessment.execution_exact == 0.0
    assert assessment.execution_safe == 0.0
    assert assessment.diagnostic["headline_metrics_zeroed"] is True
    assert assessment.diagnostic["field_scores"]


def test_load_reference_cases_uses_each_pin_transformation_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import edvise.genai.mapping.eval.dataset as dataset_mod

    coastal = _write_pin(
        tmp_path / "synthetic_coastal_cc" / "current",
        institution_id="synthetic_coastal_cc",
    )
    metro = _write_pin(
        tmp_path / "synthetic_metro_research_uni" / "current",
        institution_id="synthetic_metro_research_uni",
    )
    pins = {
        "synthetic_coastal_cc": coastal,
        "synthetic_metro_research_uni": metro,
    }
    monkeypatch.setattr(
        dataset_mod,
        "resolve_sma_few_shot_pin",
        lambda reference_id, *, catalog: pins[reference_id],
    )
    monkeypatch.setattr(
        dataset_mod,
        "select_reference_id",
        lambda *, institution_id, **_kwargs: SimpleNamespace(
            reference_id=(
                "synthetic_metro_research_uni"
                if institution_id == "synthetic_coastal_cc"
                else "synthetic_coastal_cc"
            )
        ),
    )

    cases = load_step2b_reference_cases(
        catalog="dev_sst_02",
        active_pins=[
            {"reference_id": "synthetic_coastal_cc"},
            {"reference_id": "synthetic_metro_research_uni"},
        ],
    )
    coastal_case = next(
        case for case in cases if case.target_institution_id == "synthetic_coastal_cc"
    )
    assert coastal_case.reference_institution_id == "synthetic_metro_research_uni"
    assert (
        coastal_case.gold_transformation_map["institution_id"] == "synthetic_coastal_cc"
    )
    assert (
        coastal_case.reference_transformation_map["institution_id"]
        == "synthetic_metro_research_uni"
    )
    assert coastal_case.mapping_manifest["institution_id"] == "synthetic_coastal_cc"
    record = coastal_case.evaluation_record()
    assert (
        record["inputs"]["reference_institution_id"] == "synthetic_metro_research_uni"
    )
    assert record["expectations"]["gold_transformation_map"]["institution_id"] == (
        "synthetic_coastal_cc"
    )
    assert coastal_case.output_path.endswith(
        "synthetic_coastal_cc/eval_scratch/synthetic_coastal_cc_transformation_map.json"
    )


def test_load_reference_cases_rejects_transformation_map_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import edvise.genai.mapping.eval.dataset as dataset_mod

    coastal = _write_pin(
        tmp_path / "synthetic_coastal_cc" / "current",
        institution_id="synthetic_coastal_cc",
    )
    metro = _write_pin(
        tmp_path / "synthetic_metro_research_uni" / "current",
        institution_id="synthetic_metro_research_uni",
        transformation_institution_id="other_school",
    )
    pins = {
        "synthetic_coastal_cc": coastal,
        "synthetic_metro_research_uni": metro,
    }
    monkeypatch.setattr(
        dataset_mod,
        "resolve_sma_few_shot_pin",
        lambda reference_id, *, catalog: pins[reference_id],
    )
    monkeypatch.setattr(
        dataset_mod,
        "select_reference_id",
        lambda **_kwargs: SimpleNamespace(reference_id="synthetic_metro_research_uni"),
    )
    with pytest.raises(ValueError, match="institution_id mismatch"):
        load_step2b_reference_cases(
            catalog="dev_sst_02",
            active_pins=[
                {"reference_id": "synthetic_coastal_cc"},
                {"reference_id": "synthetic_metro_research_uni"},
            ],
        )


def test_predict_fn_injects_wrapper_fields_and_records_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    monkeypatch.setattr(predict_mod, "build_step2b_prompt", lambda **_kwargs: "prompt")
    monkeypatch.setattr(
        predict_mod,
        "run_once",
        lambda *_args, **_kwargs: {
            "success": True,
            "response": json.dumps({"transformation_maps": {}}),
            "latency_s": 1.5,
            "response_chars": 24,
            "reasoning": [{"type": "reasoning", "summary": []}],
            "error": None,
        },
    )
    observation = PredictionObservation()
    predicted = make_step2b_predict_fn(
        cases=[_context()],
        gateway_model_id="gateway-sonnet",
        pipeline_version="v-test",
        client=object(),
        observation=observation,
    )(
        institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
    )
    assert predicted["institution_id"] == "synthetic_coastal_cc"
    assert predicted["pipeline_version"] == "v-test"
    assert observation.calls[0]["latency_s"] == 1.5
    assert observation.manifests == [
        {
            "institution_id": "synthetic_coastal_cc",
            "transformation_map": predicted,
        }
    ]


def test_predict_fn_returns_error_payload_when_gateway_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    monkeypatch.setattr(predict_mod, "build_step2b_prompt", lambda **_kwargs: "prompt")
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
    predicted = make_step2b_predict_fn(
        cases=[_context()],
        gateway_model_id="gateway-haiku",
        client=object(),
    )(
        institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
    )
    assert predicted == {"error": "HTTP 503"}


class _RunScope:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_run_logs_one_mlflow_run_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import edvise.genai.mapping.eval.predict as predict_mod

    monkeypatch.setattr(predict_mod, "build_step2b_prompt", lambda **_kwargs: "prompt")
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
            "response": json.dumps(_transformation_map()),
            "latency_s": 3.0,
            "response_chars": 80,
            "error": None,
        },
    )

    logged: dict = {"params": [], "metrics": {}, "artifacts": {}, "experiment": None}
    run_counter = {"n": 0}

    def evaluate(*, data, predict_fn, scorers):
        row = data[0]
        output = predict_fn(**row["inputs"])
        logged["scores"] = [
            scorer(outputs=output, expectations=row["expectations"])
            for scorer in scorers
        ]
        run_counter["n"] += 1
        return SimpleNamespace(run_id=f"run-{run_counter['n']}")

    mlflow = SimpleNamespace(
        genai=SimpleNamespace(
            evaluate=evaluate,
            scorers=SimpleNamespace(scorer=lambda fn: fn),
        ),
        set_experiment=lambda name: logged.__setitem__("experiment", name),
        start_run=lambda run_id: _RunScope(),
        log_params=lambda params: logged["params"].append(dict(params)),
        log_metric=lambda key, value: logged["metrics"].__setitem__(key, value),
        log_dict=lambda payload, path: logged["artifacts"].__setitem__(path, payload),
        log_table=lambda data, artifact_file: logged["artifacts"].__setitem__(
            artifact_file, data
        ),
    )
    run_ids = run_step2b_evaluation(
        catalog="dev_sst_02",
        models="sonnet:4.5,haiku:4.5",
        ds_run_as="00000000-0000-0000-0000-000000000001",
        cases=[_context()],
        mlflow_module=mlflow,
    )
    assert run_ids == ["run-1", "run-2"]
    assert logged["experiment"].endswith("edvise/genai_mapping/step2b_model_comparison")
    assert logged["scores"] == [1.0, 1.0, 1.0]
    assert logged["params"][0]["prompt_variant"] == "production"
    assert logged["params"][0]["gateway_model_id"] == "system.ai.claude-sonnet-4-5"
    assert logged["metrics"] == {"latency_s": 3.0, "response_chars": 80.0}
    table = logged["artifacts"]["diagnostics/step2b_overall_by_school.json"]
    assert table["institution_id"] == ["synthetic_coastal_cc"]
    assert table["execution_exact_rate"] == [1.0]
    assert table["execution_safe_rate"] == [1.0]
    saved = logged["artifacts"]["transformation_maps/synthetic_coastal_cc.json"]
    assert saved["institution_id"] == "synthetic_coastal_cc"
    detail = logged["artifacts"]["diagnostics/step2b_field_detail.json"]
    assert detail["cases"][0]["validation_passed"] is True
    assert len(detail["cases"][0]["field_scores"]) == 2


@pytest.mark.parametrize("step", ["2a", "2b"])
def test_cli_step_selects_the_matching_runner(
    monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    import edvise.genai.mapping.eval.run as run_mod
    import edvise.genai.mapping.shared.utilities as utilities

    called: list[str] = []

    def run_evaluation(**kwargs: object) -> list[str]:
        called.append(str(kwargs["step"]))
        return []

    monkeypatch.setattr(utilities, "disable_openai_autolog_for_gateway", lambda: None)
    monkeypatch.setattr(run_mod, "run_evaluation", run_evaluation)
    script = _load_eval_script()
    assert (
        script.main(
            [
                "--catalog",
                "dev_sst_02",
                "--ds_run_as",
                "00000000-0000-0000-0000-000000000001",
                "--models",
                "sonnet:4.5,haiku:4.5",
                "--step",
                step,
            ]
        )
        == 0
    )
    assert called == [step]


def test_registered_steps_drive_the_job_parameter() -> None:
    assert supported_eval_steps() == ("2a", "2b")


def test_run_evaluation_dispatches_the_registered_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edvise.genai.mapping.eval.run as run_mod

    seen: dict[str, object] = {}

    def runner(**kwargs: object) -> list[str]:
        seen.update(kwargs)
        return ["run-ia"]

    monkeypatch.setitem(
        run_mod._EVAL_STEPS,
        "ia",
        EvalStep("ia", "Identity Agent", runner),
    )
    run_ids = run_evaluation(
        step="IA",
        catalog="dev_sst_02",
        models="sonnet:4.5,haiku:4.5",
        ds_run_as="00000000-0000-0000-0000-000000000001",
    )
    assert run_ids == ["run-ia"]
    assert seen["catalog"] == "dev_sst_02"
    assert seen["models"] == "sonnet:4.5,haiku:4.5"


def test_run_evaluation_rejects_an_unknown_step() -> None:
    with pytest.raises(ValueError, match="2a, 2b"):
        run_evaluation(
            step="2c",
            catalog="dev_sst_02",
            models="sonnet:4.5,haiku:4.5",
            ds_run_as="00000000-0000-0000-0000-000000000001",
        )


def test_cli_rejects_an_unknown_step() -> None:
    script = _load_eval_script()
    with pytest.raises(SystemExit):
        script.main(
            [
                "--catalog",
                "dev_sst_02",
                "--ds_run_as",
                "00000000-0000-0000-0000-000000000001",
                "--models",
                "sonnet:4.5,haiku:4.5",
                "--step",
                "2c",
            ]
        )


def _load_eval_script():
    import edvise

    path = (
        Path(edvise.__file__).resolve().parents[2]
        / "src/edvise/genai/mapping/scripts/edvise_genai_eval.py"
    )
    spec = importlib.util.spec_from_file_location("edvise_genai_eval", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pin(
    current: Path,
    *,
    institution_id: str,
    transformation_institution_id: str | None = None,
) -> SimpleNamespace:
    current.mkdir(parents=True)
    manifest = {"institution_id": institution_id, "manifests": {}}
    transformation_map = _transformation_map(
        institution_id=transformation_institution_id or institution_id
    )
    (current / "manifest_map.json").write_text(json.dumps(manifest))
    (current / "transformation_map.json").write_text(json.dumps(transformation_map))
    (current / "enriched_schema_contract.json").write_text(json.dumps({"datasets": {}}))
    return SimpleNamespace(
        current_root=current,
        manifest_map=current / "manifest_map.json",
        transformation_map=current / "transformation_map.json",
    )


def _context() -> Step2bEvalContext:
    gold = _transformation_map()
    return Step2bEvalContext(
        catalog="dev_sst_02",
        target_institution_id="synthetic_coastal_cc",
        reference_institution_id="synthetic_metro_research_uni",
        target_contract={"datasets": {}},
        mapping_manifest={"institution_id": "synthetic_coastal_cc"},
        reference_transformation_map=_transformation_map(
            institution_id="synthetic_metro_research_uni"
        ),
        gold_transformation_map=gold,
        output_path=(
            "/Volumes/dev_sst_02/synthetic_coastal_cc_silver/scratch/"
            "transformation_map.json"
        ),
    )
