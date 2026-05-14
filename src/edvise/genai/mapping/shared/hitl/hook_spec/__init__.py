"""HookSpec models (:mod:`.schemas`), JSON parsing (:mod:`.parse`), and canonical paths (:mod:`.paths`).

``parse`` / ``paths`` are not imported at package load time; use submodule imports
(``from …hook_spec.parse import parse_hook_spec``) or lazy package attributes
(``from …hook_spec import parse_hook_spec``) so importing only :mod:`.schemas` stays lightweight.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HITLDomain",
    "HookFunctionSpec",
    "HookSpec",
    "RawHookSpecInput",
    "default_hook_module_relpath",
    "ensure_hook_spec_file",
    "hook_modules_root_from_bronze_volume",
    "parse_hook_spec",
    "resolve_hook_module_path",
]


def __getattr__(name: str) -> Any:
    if name in ("RawHookSpecInput", "parse_hook_spec"):
        from .parse import RawHookSpecInput, parse_hook_spec

        return {"RawHookSpecInput": RawHookSpecInput, "parse_hook_spec": parse_hook_spec}[name]
    if name in (
        "default_hook_module_relpath",
        "ensure_hook_spec_file",
        "hook_modules_root_from_bronze_volume",
        "resolve_hook_module_path",
    ):
        from . import paths as _paths

        return getattr(_paths, name)
    if name in ("HITLDomain", "HookFunctionSpec", "HookSpec"):
        from . import schemas as _schemas

        return getattr(_schemas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return list(__all__)
