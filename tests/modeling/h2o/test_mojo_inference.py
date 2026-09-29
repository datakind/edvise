import json
import os
import zipfile
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from edvise.modeling.h2o_ml import inference, utils


_MODEL_INI = """
[info]
algo = gbm
response_column = y
weights_column = sample_weight

[columns]
x1
x2
y
sample_weight

[domains]
"""


def test_algo_from_model_id_reads_automl_prefixes():
    assert utils.algo_from_model_id("StackedEnsemble_AllModels_1_AutoML_1") == (
        "stackedensemble"
    )
    assert utils.algo_from_model_id("XGBoost_3_AutoML_1") == "xgboost"
    assert utils.algo_from_model_id("GBM_grid_1_AutoML_1") == "gbm"
    assert utils.algo_from_model_id("DRF_1_AutoML_1") == "drf"
    assert utils.algo_from_model_id("GLM_1_AutoML_1") == "glm"
    assert utils.algo_from_model_id("DeepLearning_1") is None
    assert utils.algo_from_model_id(None) is None


def test_uses_mojo_backend_only_for_complete_tree_artifacts():
    assert utils.uses_mojo_backend("gbm", has_mojo=True, has_genmodel_jar=True)
    assert utils.uses_mojo_backend("XGBoost", has_mojo=True, has_genmodel_jar=True)
    assert not utils.uses_mojo_backend("glm", has_mojo=True, has_genmodel_jar=True)
    assert not utils.uses_mojo_backend(
        "stackedensemble", has_mojo=True, has_genmodel_jar=True
    )
    assert not utils.uses_mojo_backend("gbm", has_mojo=True, has_genmodel_jar=False)
    assert not utils.uses_mojo_backend("gbm", has_mojo=False, has_genmodel_jar=True)
    assert not utils.uses_mojo_backend(None, has_mojo=True, has_genmodel_jar=True)


def test_parse_mojo_model_ini_drops_non_predictors():
    algo, features = utils.parse_mojo_model_ini(_MODEL_INI)
    assert algo == "gbm"
    assert features == ["x1", "x2"]


def test_try_export_mojo_normalizes_filenames(tmp_path):
    class _Model:
        algo = "gbm"

        def download_mojo(self, path=".", get_genmodel_jar=False, genmodel_name=""):
            assert get_genmodel_jar is True
            with open(os.path.join(path, "h2o-genmodel.jar"), "w") as jar_file:
                jar_file.write("jar")
            mojo_path = os.path.join(path, "GBM_1.zip")
            with open(mojo_path, "w") as mojo_file:
                mojo_file.write("zip")
            return mojo_path

    mojo_path, jar_path = utils._try_export_mojo(_Model(), str(tmp_path))
    assert os.path.basename(mojo_path) == "model.zip"
    assert os.path.basename(jar_path) == "h2o-genmodel.jar"
    assert not (tmp_path / "GBM_1.zip").exists()


def _save_model(model, path, force=True):
    dest = os.path.join(path, "saved_model")
    with open(dest, "w") as handle:
        handle.write("model")
    return dest


def _gbm_model():
    model = mock.Mock()
    model.algo = "gbm"
    model._model_json = {
        "output": {
            "names": ["x1", "y"],
            "response_column": {"name": "y"},
        }
    }
    model.actual_params = {}
    return model


def test_logger_keeps_binary_model_when_mojo_export_fails(monkeypatch):
    model = _gbm_model()
    model.download_mojo.side_effect = RuntimeError("unsupported")
    logged: list[str] = []
    monkeypatch.setattr(utils.h2o, "save_model", _save_model)
    monkeypatch.setattr(
        utils.mlflow,
        "log_text",
        lambda text, artifact_file: logged.append(artifact_file),
    )
    monkeypatch.setattr(
        utils.mlflow, "log_artifact", lambda *args, **kwargs: logged.append("artifact")
    )

    utils.log_h2o_model_metadata_for_uc(model, "model", signature=None)

    assert "artifact" in logged
    assert "model/MLmodel" in logged


def test_logger_allows_glm_without_mojo_and_records_features(monkeypatch):
    model = _gbm_model()
    model.algo = "glm"
    model.download_mojo.side_effect = RuntimeError("unsupported")
    logged: list[str] = []
    monkeypatch.setattr(utils.h2o, "save_model", _save_model)
    monkeypatch.setattr(
        utils.mlflow,
        "log_text",
        lambda text, artifact_file: logged.append(artifact_file),
    )
    monkeypatch.setattr(utils.mlflow, "log_artifact", lambda *args, **kwargs: None)

    utils.log_h2o_model_metadata_for_uc(model, "model", signature=None)

    assert "model/used_features.json" in logged
    assert "model/MLmodel" in logged


def test_log_model_metadata_records_algo(monkeypatch):
    params: list[tuple[str, str]] = []
    monkeypatch.setattr(
        utils.mlflow, "log_param", lambda key, value: params.append((key, value))
    )
    monkeypatch.setattr(utils.mlflow, "log_metrics", lambda metrics: None)
    model = mock.Mock()
    model.algo = "drf"
    model._parms = {}

    utils.log_model_metadata_to_mlflow("DRF_1", model, {"auc": 0.8})

    assert ("model_id", "DRF_1") in params
    assert ("algo", "drf") in params


def test_load_inference_model_uses_mojo_feature_artifact(monkeypatch):
    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "xgboost")
    monkeypatch.setattr(
        utils,
        "_artifact_names",
        lambda run_id, parent: {
            "model.zip",
            "h2o-genmodel.jar",
            "used_features.json",
        },
    )

    def fake_download(run_id, artifact_path, dst_dir=None):
        dest = os.path.join(dst_dir, os.path.basename(artifact_path))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if artifact_path.endswith("used_features.json"):
            with open(dest, "w") as handle:
                json.dump({"features": ["x1", "x2"]}, handle)
        else:
            with open(dest, "w") as handle:
                handle.write("bytes")
        return dest

    monkeypatch.setattr(utils, "download_artifact_file", fake_download)
    loaded = utils.load_inference_model("run-1")

    assert loaded.uses_mojo
    assert loaded.algo == "xgboost"
    assert loaded.feature_names == ["x1", "x2"]
    assert os.path.isfile(loaded.mojo_zip_path)
    assert os.path.isfile(loaded.genmodel_jar_path)


def test_load_inference_model_reads_model_ini_when_feature_artifact_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "gbm")
    monkeypatch.setattr(
        utils,
        "_artifact_names",
        lambda run_id, parent: {"model.zip", "h2o-genmodel.jar"},
    )

    def fake_download(run_id, artifact_path, dst_dir=None):
        dest = os.path.join(dst_dir, os.path.basename(artifact_path))
        if artifact_path.endswith("model.zip"):
            with zipfile.ZipFile(dest, "w") as zf:
                zf.writestr("model.ini", _MODEL_INI)
        else:
            with open(dest, "w") as handle:
                handle.write("jar")
        return dest

    monkeypatch.setattr(utils, "download_artifact_file", fake_download)
    loaded = utils.load_inference_model("run-1")
    assert loaded.feature_names == ["x1", "x2"]


def test_mojo_import_failure_falls_back_to_the_cluster(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "gbm")
    monkeypatch.setattr(
        utils,
        "_artifact_names",
        lambda run_id, parent: {"model.zip", "h2o-genmodel.jar"},
    )

    def fail_download(*args, **kwargs):
        raise OSError("download failed")

    monkeypatch.setattr(utils, "download_artifact_file", fail_download)
    monkeypatch.setattr(utils, "load_h2o_model", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(utils, "predictor_names_from_h2o_model", lambda model: ["x"])

    loaded = utils.load_inference_model("gbm-run")

    assert loaded.backend == "h2o"
    assert loaded.h2o_model is sentinel


def test_glm_and_incomplete_mojo_start_the_cluster(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(utils, "load_h2o_model", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(utils, "predictor_names_from_h2o_model", lambda model: ["x"])

    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "glm")
    monkeypatch.setattr(
        utils,
        "_artifact_names",
        lambda run_id, parent: {"model.zip", "h2o-genmodel.jar"},
    )
    glm_loaded = utils.load_inference_model("glm-run")
    assert glm_loaded.backend == "h2o"
    assert glm_loaded.h2o_model is sentinel

    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "gbm")
    monkeypatch.setattr(utils, "_artifact_names", lambda run_id, parent: {"model.zip"})
    gbm_loaded = utils.load_inference_model("gbm-run")
    assert gbm_loaded.backend == "h2o"

    monkeypatch.setattr(utils, "resolve_logged_algo", lambda run_id: "stackedensemble")
    monkeypatch.setattr(
        utils,
        "_artifact_names",
        lambda run_id, parent: {"model.zip", "h2o-genmodel.jar"},
    )
    ensemble_loaded = utils.load_inference_model("ensemble-run")
    assert ensemble_loaded.backend == "h2o"


def test_predict_mojo_calibrates_before_threshold(monkeypatch):
    def fake_score(features, **kwargs):
        assert kwargs["predict_contributions"] is False
        return pd.DataFrame({"predict": [1, 0], "p0": [0.25, 0.8], "p1": [0.75, 0.2]})

    monkeypatch.setattr(inference, "score_mojo_frame", fake_score)

    class _Calibrator:
        def transform(self, probs):
            return np.asarray(probs) * 0.5

    labels, probs = inference.predict_mojo(
        pd.DataFrame({"x": [1.0, 2.0]}),
        mojo_zip_path="model.zip",
        genmodel_jar_path="h2o-genmodel.jar",
        pos_label=True,
        calibrator=_Calibrator(),
        classification_threshold=0.5,
    )
    np.testing.assert_allclose(probs, [0.375, 0.1])
    assert labels.tolist() == [0, 0]


def test_mojo_contributions_scale_to_probability_without_exploding(monkeypatch):
    def fake_score(features, **kwargs):
        assert kwargs["predict_contributions"] is True
        # Second row's logit is 0.002. Dividing the probability by that logit
        # would turn the feature contribution of 120 into about 30,000.
        return pd.DataFrame({"x": [2.0, 120.0], "BiasTerm": [0.0, -119.998]})

    monkeypatch.setattr(inference, "score_mojo_frame", fake_score)
    contribs = inference.compute_mojo_contributions(
        pd.DataFrame({"x": [1.0, 2.0]}),
        mojo_zip_path="model.zip",
        genmodel_jar_path="h2o-genmodel.jar",
        batch_rows=10,
        output_space=True,
        drop_bias=False,
    )
    bias = contribs["BiasTerm"].to_numpy()
    scaled = contribs["x"].to_numpy()
    np.testing.assert_allclose(bias[0], 0.5)
    expected_prob = inference._sigmoid(np.array([2.0, 0.002]))
    np.testing.assert_allclose(bias + scaled, expected_prob)
    assert abs(scaled[1]) < 1.0

    link_space = inference.compute_mojo_contributions(
        pd.DataFrame({"x": [1.0, 2.0]}),
        mojo_zip_path="model.zip",
        genmodel_jar_path="h2o-genmodel.jar",
        output_space=False,
        drop_bias=True,
    )
    np.testing.assert_allclose(link_space["x"].to_numpy(), [2.0, 120.0])


def test_loaded_model_dispatch(monkeypatch):
    monkeypatch.setattr(
        inference,
        "predict_mojo",
        lambda features, **kwargs: (np.array([1]), np.array([0.7])),
    )
    mojo_model = utils.LoadedInferenceModel(
        feature_names=["x"],
        algo="gbm",
        backend="mojo",
        mojo_zip_path="model.zip",
        genmodel_jar_path="h2o-genmodel.jar",
    )
    labels, probs = inference.predict_loaded_model(
        pd.DataFrame({"x": [1.0]}), mojo_model
    )
    assert labels.tolist() == [1]
    np.testing.assert_allclose(probs, [0.7])

    sentinel = object()
    seen: dict[str, object] = {}

    def fake_h2o_contribs(model, df, background_data=None, **kwargs):
        seen["model"] = model
        seen["background"] = background_data
        return pd.DataFrame({"x": [0.2]})

    monkeypatch.setattr(inference, "compute_h2o_shap_contributions", fake_h2o_contribs)
    glm_model = utils.LoadedInferenceModel(
        feature_names=["x"],
        algo="glm",
        backend="h2o",
        h2o_model=sentinel,
    )
    background = pd.DataFrame({"x": [0.0]})
    contribs = inference.contributions_for_loaded_model(
        glm_model, pd.DataFrame({"x": [1.0]}), background
    )
    assert seen["model"] is sentinel
    assert seen["background"] is background
    assert contribs["x"].tolist() == [0.2]


def test_mojo_contributions_reject_row_filters():
    with pytest.raises(NotImplementedError):
        inference.compute_mojo_contributions(
            pd.DataFrame({"x": [1.0]}),
            mojo_zip_path="model.zip",
            genmodel_jar_path="h2o-genmodel.jar",
            top_n=3,
        )


def _java_can_score_mojo() -> bool:
    try:
        from h2o.backend.server import H2OLocalServer

        java = H2OLocalServer._find_java()
        H2OLocalServer._check_java(java=java, verbose=False)
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _java_can_score_mojo(),
    reason="H2O and MOJO scoring require a Java runtime",
)
def test_gbm_mojo_probabilities_and_treeshap_match_native_h2o():
    import h2o
    from h2o.estimators import H2OGradientBoostingEstimator

    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=60),
            "x2": rng.normal(size=60),
            "cat": np.where(np.arange(60) % 2 == 0, "a", "b"),
            "y": (rng.random(60) > 0.45).astype(int),
        }
    )
    h2o.init(
        ip="127.0.0.1",
        port=15432,
        nthreads=1,
        max_mem_size="1G",
        strict_version_check=False,
    )
    try:
        hf = h2o.H2OFrame(frame)
        hf["cat"] = hf["cat"].asfactor()
        hf["y"] = hf["y"].asfactor()
        model = H2OGradientBoostingEstimator(ntrees=8, max_depth=2, seed=7)
        model.train(x=["x1", "x2", "cat"], y="y", training_frame=hf)
        native_probs = model.predict(hf).as_data_frame()["p1"].to_numpy(dtype=float)
        native_link = model.predict_contributions(
            hf, output_space=False
        ).as_data_frame()

        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            mojo_path = model.download_mojo(path=tmp_dir, get_genmodel_jar=True)
            jar_path = os.path.join(tmp_dir, "h2o-genmodel.jar")
            mojo_algo, mojo_features = utils.features_from_mojo_zip(mojo_path)
            assert mojo_algo == "gbm"
            assert mojo_features == ["x1", "x2", "cat"]
            features = frame[["x1", "x2", "cat"]]
            _labels, probs = inference.predict_mojo(
                features,
                mojo_zip_path=mojo_path,
                genmodel_jar_path=jar_path,
                pos_label=True,
            )
            link_contribs = inference.compute_mojo_contributions(
                features,
                mojo_zip_path=mojo_path,
                genmodel_jar_path=jar_path,
                output_space=False,
                drop_bias=False,
            )
            output_contribs = inference.compute_mojo_contributions(
                features,
                mojo_zip_path=mojo_path,
                genmodel_jar_path=jar_path,
                output_space=True,
                drop_bias=False,
            )
        np.testing.assert_allclose(probs, native_probs, atol=1e-6)
        _assert_same_contributions(link_contribs, native_link)
        link_sum = link_contribs.sum(axis=1).to_numpy(dtype=float)
        np.testing.assert_allclose(
            output_contribs.sum(axis=1).to_numpy(dtype=float),
            inference._sigmoid(link_sum),
            atol=1e-5,
        )
        assert output_contribs.drop(columns="BiasTerm").abs().to_numpy().max() < 1.0
    finally:
        h2o.cluster().shutdown(prompt=False)


def _assert_same_contributions(actual: pd.DataFrame, expected: pd.DataFrame) -> None:
    columns = [col for col in expected.columns if col in actual.columns]
    assert columns
    np.testing.assert_allclose(
        actual[columns].to_numpy(dtype=float),
        expected[columns].to_numpy(dtype=float),
        atol=1e-5,
    )
