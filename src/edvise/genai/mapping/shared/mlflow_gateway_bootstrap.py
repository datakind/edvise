"""
Disable MLflow GenAI tracing / OpenAI autolog before ``openai`` is imported.

Databricks runtimes may enable ``mlflow.openai.autolog`` early; if ``openai`` is imported
after autolog is on, MLflow wraps :meth:`openai.resources.chat.completions.Completions.create`.
Call
:func:`disable_mlflow_side_effects_for_openai_gateway` from job entrypoints **before** any
``edvise.genai`` import that pulls in ``openai``.
"""

from __future__ import annotations


def disable_mlflow_side_effects_for_openai_gateway() -> None:
    try:
        import mlflow
    except ImportError:
        return
    try:
        mlflow.tracing.disable()
    except Exception:
        pass
    try:
        import mlflow.openai as mlflow_openai

        mlflow_openai.autolog(disable=True)
    except Exception:
        pass
