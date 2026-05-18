"""
Compare ``HookFunctionSpec.draft`` (AST) to the runtime function after import.

Used by :func:`~edvise.genai.mapping.identity_agent.hitl.resolver.validate_hook` as the
structural check for grain/transform when literal smoke tests do not apply.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable


def _param_names_from_functiondef(node: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for a in node.args.posonlyargs:
        names.append(a.arg)
    for a in node.args.args:
        names.append(a.arg)
    if node.args.vararg is not None:
        names.append(node.args.vararg.arg)
    for a in node.args.kwonlyargs:
        names.append(a.arg)
    if node.args.kwarg is not None:
        names.append(node.args.kwarg.arg)
    return names


def _draft_return_semantic_str(func_def: ast.FunctionDef) -> tuple[str | None, bool]:
    """
    Return text comparable to :func:`inspect.signature` return_annotation.

    String forward refs (``-> "pd.DataFrame"``) are returned as ``pd.DataFrame`` — no outer
    quote characters — so they match runtime ``str`` annotations and compare fairly to
    resolved types (see :func:`_return_ann_match`).
    """
    if func_def.returns is None:
        return None, False
    node = func_def.returns
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value, True
    return ast.unparse(node).strip(), True


def _runtime_ann_short(ann: object) -> str:
    if ann is inspect.Signature.empty:
        return ""
    if isinstance(ann, type):
        return ann.__qualname__
    if isinstance(ann, str):
        return ann
    return str(ann)


def _return_ann_match(draft_ret: str, ann: object) -> bool:
    """
    True if draft return annotation is consistent with runtime annotation.

    Handles quoted forward refs (``"pd.DataFrame"`` in source) vs runtime ``str`` or
    resolved class objects (e.g. ``pandas.DataFrame``).
    """
    d = draft_ret.replace(" ", "")
    if ann is inspect.Signature.empty:
        return False
    if isinstance(ann, str):
        return d == ann.replace(" ", "")
    if isinstance(ann, type):
        q = ann.__qualname__.replace(" ", "")
        n = ann.__name__.replace(" ", "")
        if d in (q, n):
            return True
        # e.g. draft "pd.DataFrame" vs runtime pandas.DataFrame (__qualname__ "DataFrame")
        if d.endswith("." + q) or d.endswith("." + n):
            return True
        return False
    return d == str(ann).replace(" ", "")


def signature_mismatches(
    fn: Callable[..., object],
    *,
    expected_name: str,
    draft: str | None,
) -> list[str]:
    """
    Return human-readable mismatch messages, or an empty list if draft matches runtime.

    Compares parameter names (order) and return annotation when the draft's ``def`` includes
    a ``->`` return annotation.
    """
    errs: list[str] = []
    if not draft or not str(draft).strip():
        return [f"[{expected_name}] hook_spec draft is empty — cannot verify signature"]

    try:
        tree = ast.parse(draft)
    except SyntaxError as e:
        return [f"[{expected_name}] draft is not valid Python: {e}"]

    func_def: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == expected_name:
            func_def = node
            break

    if func_def is None:
        return [
            f"[{expected_name}] draft must contain a top-level `def {expected_name}(...):` "
            f"matching the configured name"
        ]

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as e:
        return [f"[{expected_name}] could not introspect runtime function: {e}"]

    draft_params = _param_names_from_functiondef(func_def)
    runtime_params = list(sig.parameters.keys())

    if draft_params != runtime_params:
        errs.append(
            f"[{expected_name}] parameter names/order mismatch: draft {draft_params!r} "
            f"vs module {runtime_params!r}"
        )

    draft_ret_str, has_draft_return = _draft_return_semantic_str(func_def)
    if has_draft_return and draft_ret_str is not None:
        ann = sig.return_annotation
        if ann is inspect.Signature.empty:
            errs.append(
                f"[{expected_name}] draft declares return type {draft_ret_str!r} but the "
                f"loaded function has no return annotation"
            )
        elif not _return_ann_match(draft_ret_str, ann):
            errs.append(
                f"[{expected_name}] return annotation mismatch: draft `{draft_ret_str}` "
                f"vs runtime `{_runtime_ann_short(ann)}`"
            )

    return errs


__all__ = ["signature_mismatches"]
