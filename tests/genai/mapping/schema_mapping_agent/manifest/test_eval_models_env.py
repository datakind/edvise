import os

import pytest

from edvise.genai.mapping.eval import models as eval_models


@pytest.fixture
def restore_models():
    orig = list(eval_models.MODELS)
    prev = os.environ.get("EDVISE_EVAL_MODELS")
    try:
        yield
    finally:
        eval_models.MODELS[:] = orig
        if prev is None:
            os.environ.pop("EDVISE_EVAL_MODELS", None)
        else:
            os.environ["EDVISE_EVAL_MODELS"] = prev


def test_apply_eval_models_from_env_sonnet_slug(restore_models):
    os.environ["EDVISE_EVAL_MODELS"] = "sonnet"
    eval_models.apply_eval_models_from_env()
    assert eval_models.MODELS == ["claude-sonnet-test-genai-ai-data-cleaning"]


def test_apply_eval_models_from_env_order_and_dedupe(restore_models):
    os.environ["EDVISE_EVAL_MODELS"] = "haiku, sonnet, haiku"
    eval_models.apply_eval_models_from_env()
    assert eval_models.MODELS == [
        "claude-haiku-test-genai-data-cleaning",
        "claude-sonnet-test-genai-ai-data-cleaning",
    ]


def test_apply_eval_models_from_env_unknown_raises(restore_models):
    os.environ["EDVISE_EVAL_MODELS"] = "gpt-4"
    with pytest.raises(ValueError, match="unknown entry"):
        eval_models.apply_eval_models_from_env()


def test_apply_eval_models_from_env_noop_when_unset(restore_models):
    os.environ.pop("EDVISE_EVAL_MODELS", None)
    before = list(eval_models.MODELS)
    eval_models.apply_eval_models_from_env()
    assert eval_models.MODELS == before
