"""Cross-agent helpers: LLM output cleanup and MLflow/OpenAI gateway bootstrap."""

from __future__ import annotations

import json
from typing import Any, cast


def load_json(path: str) -> dict[str, Any]:
    """Read a JSON object from ``path``."""
    with open(path) as f:
        return cast(dict[str, Any], json.load(f))


def strip_json_fences(text: str) -> str:
    """Strip markdown code fences from JSON text."""
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1 :]
    if text.endswith("```"):
        text = text[: text.rindex("```")].rstrip()
    return text


def disable_openai_autolog_for_gateway() -> None:
    """Disable OpenAI autolog while leaving MLflow tracing available.

    Gateway streaming chunks should not become their own traces. Evaluation
    still needs tracing so ``mlflow.genai.evaluate`` can record the prediction.
    """
    try:
        import mlflow.openai as mlflow_openai
    except ImportError:
        return
    try:
        mlflow_openai.autolog(disable=True)
    except Exception:
        pass


def disable_mlflow_side_effects_for_openai_gateway() -> None:
    """
    Disable MLflow GenAI tracing / OpenAI autolog before ``openai`` is imported.

    Databricks runtimes may enable ``mlflow.openai.autolog`` early; if ``openai`` is imported
    after autolog is on, MLflow wraps :meth:`openai.resources.chat.completions.Completions.create`.
    Call this from job entrypoints **before** any ``edvise.genai`` import that pulls in ``openai``.
    """
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.tracing.disable()
    except Exception:
        pass
    disable_openai_autolog_for_gateway()


__all__ = [
    "disable_mlflow_side_effects_for_openai_gateway",
    "disable_openai_autolog_for_gateway",
    "strip_json_fences",
]
