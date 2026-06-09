"""
Version-aware inference job parameter resolution for ``runs/submit``.

The archived inference YAML at ``pipeline_version`` is the contract source of truth.
Launcher / webapp inputs use current (or stable) names; optional ``parameter_aliases.json``
in the release bundle maps archived names to launcher or stable-trigger paths.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PARAMETER_ALIASES_FILENAME = "parameter_aliases.json"

_BUNDLE_VAR = re.compile(r"\$\{var\.[^}]+\}")
_BUNDLE_VAR_NAME = re.compile(r"^\$\{var\.([^}]+)\}$")
_JOB_PARAM_REF = re.compile(r"\{\{\s*job\.parameters\.([A-Za-z0-9_]+)\s*\}\}")
_UNRESOLVED_RUN_ID = "{{job.run_id}}"
_BUNDLE_SNAPSHOT_YML = "databricks_bundle_snapshot/databricks.yml"

_SENSITIVE_KEY_FRAGMENTS = frozenset(
    {"password", "secret", "token", "credential", "api_key", "apikey"}
)


@dataclass(frozen=True)
class ParameterSpec:
    """One job parameter from archived inference YAML."""

    name: str
    default: str
    referenced_by_tasks: list[str] = field(default_factory=list)
    default_explicit: bool = False

    @property
    def required(self) -> bool:
        """Referenced by tasks and no concrete archived default."""
        if not self.referenced_by_tasks:
            return False
        if self.name == "db_run_id":
            return False
        if self.default_explicit and not str(self.default).strip():
            return False
        return not has_concrete_default(self.default)


def has_concrete_default(value: str | None) -> bool:
    """Return whether ``default`` is usable without launcher overrides."""
    if value is None:
        return False
    s = str(value).strip()
    if not s:
        return False
    if _contains_bundle_var(s):
        return False
    if s == _UNRESOLVED_RUN_ID:
        return False
    return True


def _contains_bundle_var(value: str) -> bool:
    return _BUNDLE_VAR.search(value) is not None


def resolve_bundle_var_default(default: str, bundle_vars: dict[str, str]) -> str | None:
    """Resolve ``${var.name}`` using defaults from archived ``databricks.yml``."""
    match = _BUNDLE_VAR_NAME.match(str(default).strip())
    if not match:
        return None
    resolved = bundle_vars.get(match.group(1))
    if resolved is None:
        return None
    s = str(resolved).strip()
    return s if s else None


def load_bundle_variable_defaults(release_dir: Path) -> dict[str, str]:
    """
    Read ``variables.*.default`` from the materialized bundle ``databricks.yml``.

    Used to resolve job-parameter defaults like ``${var.schema_type}`` on ``runs/submit``.
    """
    path = release_dir / _BUNDLE_SNAPSHOT_YML
    if not path.is_file():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        LOGGER.warning("Could not read bundle variable defaults from %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        return {}
    variables = raw.get("variables")
    if not isinstance(variables, dict):
        return {}
    out: dict[str, str] = {}
    for name, spec in variables.items():
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(spec, dict):
            continue
        default = spec.get("default")
        if default is None:
            continue
        if _contains_bundle_var(str(default)):
            continue
        s = str(default).strip()
        if s:
            out[name.strip()] = s
    return out


def collect_referenced_job_parameters(tasks: list[Any]) -> dict[str, list[str]]:
    """Map each ``{{job.parameters.NAME}}`` to task keys that reference it."""
    refs: dict[str, list[str]] = {}

    def walk(obj: Any, task_key: str) -> None:
        if isinstance(obj, str):
            for match in _JOB_PARAM_REF.finditer(obj):
                name = match.group(1)
                bucket = refs.setdefault(name, [])
                if task_key not in bucket:
                    bucket.append(task_key)
        elif isinstance(obj, dict):
            for val in obj.values():
                walk(val, task_key)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, task_key)

    if not isinstance(tasks, list):
        return refs
    for task in tasks:
        if not isinstance(task, dict):
            continue
        key = task.get("task_key")
        task_key = key.strip() if isinstance(key, str) else ""
        if task_key:
            walk(task, task_key)
    return refs


def build_parameter_contract(job: dict[str, Any]) -> list[ParameterSpec]:
    """Build parameter contract from an archived inference job definition."""
    raw_params = job.get("parameters")
    param_list = raw_params if isinstance(raw_params, list) else []
    referenced = collect_referenced_job_parameters(
        job.get("tasks") if isinstance(job.get("tasks"), list) else []
    )

    specs: list[ParameterSpec] = []
    seen: set[str] = set()
    for entry in param_list:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        pname = name.strip()
        if pname in seen:
            continue
        seen.add(pname)
        default_explicit = "default" in entry
        default = entry.get("default", "") if default_explicit else ""
        default_str = str(default) if default is not None else ""
        specs.append(
            ParameterSpec(
                name=pname,
                default=default_str,
                referenced_by_tasks=list(referenced.get(pname, [])),
                default_explicit=default_explicit,
            )
        )

    for pname, task_keys in sorted(referenced.items()):
        if pname in seen:
            continue
        specs.append(
            ParameterSpec(
                name=pname,
                default="",
                referenced_by_tasks=list(task_keys),
            )
        )
    return specs


def parameter_contract_as_dicts(contract: list[ParameterSpec]) -> list[dict[str, Any]]:
    """Serialize contract for logging / release metadata."""
    return [
        {
            "parameter_name": spec.name,
            "default": spec.default,
            "required": spec.required,
            "referenced_by_tasks": list(spec.referenced_by_tasks),
        }
        for spec in contract
    ]


def load_parameter_aliases(release_dir: Path) -> dict[str, str]:
    """
    Load optional ``parameter_aliases.json`` from a release bundle directory.

    Maps **archived parameter name** → **launcher flat key** or **stable path**
    (dot notation, e.g. ``datasets.cohort``).
    """
    path = release_dir / PARAMETER_ALIASES_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        msg = f"Invalid {PARAMETER_ALIASES_FILENAME} at {path}: {exc}"
        raise ValueError(msg) from exc
    if not isinstance(data, dict):
        msg = f"{PARAMETER_ALIASES_FILENAME} root must be a JSON object"
        raise TypeError(msg)
    raw = data.get("parameter_aliases", data)
    if not isinstance(raw, dict):
        msg = f"{PARAMETER_ALIASES_FILENAME} parameter_aliases must be an object"
        raise TypeError(msg)
    out: dict[str, str] = {}
    for archived, source in raw.items():
        if not isinstance(archived, str) or not archived.strip():
            continue
        if source is None:
            continue
        source_str = str(source).strip()
        if source_str:
            out[archived.strip()] = source_str
    return out


def resolve_stable_path(payload: dict[str, Any], path: str) -> str | None:
    """Resolve ``datasets.cohort`` style paths from a nested stable-trigger dict."""
    parts = [p for p in path.split(".") if p]
    if not parts:
        return None
    cur: Any = payload
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    s = str(cur).strip()
    return s if s else None


def _resolve_alias_source(
    source: str,
    *,
    launcher_overrides: dict[str, str],
    stable_trigger: dict[str, Any] | None,
) -> str | None:
    if "." in source:
        if stable_trigger is None:
            return None
        return resolve_stable_path(stable_trigger, source)
    direct = launcher_overrides.get(source)
    if _non_empty(direct):
        return str(direct).strip()
    if stable_trigger is not None:
        top = stable_trigger.get(source)
        if top is not None and not isinstance(top, (dict, list)):
            s = str(top).strip()
            if s:
                return s
    return None


def _non_empty(value: str | None) -> bool:
    return bool(value is not None and str(value).strip())


def validate_extra_overrides(
    extra_overrides: dict[str, str],
    contract: list[ParameterSpec],
) -> None:
    """Reject unknown keys in ``inference_parameters_json`` (archived names only)."""
    known = {spec.name for spec in contract}
    unknown = sorted(set(extra_overrides) - known)
    if unknown:
        msg = (
            "inference_parameters_json contains unknown archived parameter(s) "
            f"{unknown}; expected one of {sorted(known)}"
        )
        raise ValueError(msg)


def validate_referenced_parameters(
    values: dict[str, str],
    contract: list[ParameterSpec],
) -> None:
    """Fail fast when a task-referenced parameter is missing or empty."""
    missing: list[str] = []
    for spec in contract:
        if not spec.required:
            continue
        if _non_empty(values.get(spec.name)):
            continue
        tasks = ", ".join(spec.referenced_by_tasks) or "?"
        missing.append(f"{spec.name!r} (referenced by {tasks})")
    if missing:
        msg = (
            "Archived inference job is missing required parameter value(s): "
            + "; ".join(missing)
            + ". Supply via launcher args, inference_parameters_json, "
            "parameter_aliases.json, or archived defaults."
        )
        raise ValueError(msg)


def resolve_archived_parameter_values(
    contract: list[ParameterSpec],
    *,
    launcher_overrides: dict[str, str],
    extra_overrides: dict[str, str] | None = None,
    stable_trigger: dict[str, Any] | None = None,
    parameter_aliases: dict[str, str] | None = None,
    bundle_variable_defaults: dict[str, str] | None = None,
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """
    Merge launcher inputs into archived parameter names for ``runs/submit``.

    Precedence (lowest → highest):

    1. Concrete defaults from archived inference job YAML
    2. ``${var.*}`` defaults resolved from archived ``databricks.yml`` variables
    3. Explicit empty archived defaults (e.g. ``term_filter: ""``)
    4. ``parameter_aliases.json`` (archived name ← launcher key or stable path)
    5. Direct launcher override when key matches archived name (overrides defaults)
    6. ``extra_overrides`` (``inference_parameters_json``; archived names only)
    """
    extra = dict(extra_overrides or {})
    aliases = dict(parameter_aliases or {})
    bundle_vars = dict(bundle_variable_defaults or {})
    validate_extra_overrides(extra, contract)

    values: dict[str, str] = {}
    for spec in contract:
        if has_concrete_default(spec.default):
            values[spec.name] = str(spec.default).strip()
            continue
        from_bundle = resolve_bundle_var_default(spec.default, bundle_vars)
        if _non_empty(from_bundle):
            values[spec.name] = str(from_bundle).strip()
            continue
        if spec.default_explicit and not str(spec.default).strip():
            values[spec.name] = ""

    for archived_name, source in aliases.items():
        if _non_empty(values.get(archived_name)):
            continue
        resolved = _resolve_alias_source(
            source,
            launcher_overrides=launcher_overrides,
            stable_trigger=stable_trigger,
        )
        if _non_empty(resolved):
            values[archived_name] = str(resolved).strip()

    for spec in contract:
        direct = launcher_overrides.get(spec.name)
        if _non_empty(direct):
            values[spec.name] = str(direct).strip()

    for name, val in extra.items():
        if val is not None and str(val).strip():
            values[name] = str(val).strip()

    validate_referenced_parameters(values, contract)
    _log_resolved_parameters(values, contract, logger=logger)
    return values


def _log_resolved_parameters(
    values: dict[str, str],
    contract: list[ParameterSpec],
    *,
    logger: logging.Logger,
) -> None:
    redacted: dict[str, str] = {}
    for spec in contract:
        val = values.get(spec.name, "")
        if not _non_empty(val):
            redacted[spec.name] = ""
            continue
        key_lower = spec.name.lower()
        if any(frag in key_lower for frag in _SENSITIVE_KEY_FRAGMENTS):
            redacted[spec.name] = "***"
        elif len(val) > 120:
            redacted[spec.name] = val[:117] + "..."
        else:
            redacted[spec.name] = val
    required = [spec.name for spec in contract if spec.required]
    logger.info(
        "Resolved archived inference parameters (required=%s): %s",
        required,
        redacted,
    )


def build_stable_trigger_payload(
    *,
    institution: str,
    model_name: str,
    workspace: str,
    cohort_dataset: str = "",
    course_dataset: str = "",
    output_bucket: str = "",
    notification_to: str = "",
    notification_cc: str = "",
    inference_output_run_id: str = "",
    ds_run_as: str = "",
    service_account_executer: str = "",
) -> dict[str, Any]:
    """
    Layer-1 stable trigger shape (webapp / launcher logical schema).

    Maps to archived names via ``parameter_aliases.json`` stable paths over time.
    """
    payload: dict[str, Any] = {
        "institution": institution.strip(),
        "model": model_name.strip(),
        "workspace": workspace.strip(),
        "datasets": {
            "cohort": cohort_dataset.strip(),
            "course": course_dataset.strip(),
        },
        "outputs": {
            "bucket": output_bucket.strip(),
            "run_id": inference_output_run_id.strip(),
        },
        "notifications": {
            "to": notification_to.strip(),
            "cc": notification_cc.strip(),
        },
        "run_as": {
            "ds": ds_run_as.strip(),
            "service_account_executer": service_account_executer.strip(),
        },
    }
    return payload


def resolve_versioned_job_parameters(
    job: dict[str, Any],
    release_dir: Path,
    *,
    launcher_overrides: dict[str, str],
    extra_overrides: dict[str, str] | None = None,
    stable_trigger: dict[str, Any] | None = None,
    logger: logging.Logger = LOGGER,
) -> dict[str, str]:
    """Build validated archived-name parameter map for ``build_submit_run_body``."""
    contract = build_parameter_contract(job)
    aliases = load_parameter_aliases(release_dir)
    bundle_vars = load_bundle_variable_defaults(release_dir)
    if aliases:
        logger.info(
            "Loaded %s parameter alias(es) from %s",
            len(aliases),
            release_dir / PARAMETER_ALIASES_FILENAME,
        )
    if bundle_vars:
        logger.info(
            "Loaded %s bundle variable default(s) from archived databricks.yml",
            len(bundle_vars),
        )
    return resolve_archived_parameter_values(
        contract,
        launcher_overrides=launcher_overrides,
        extra_overrides=extra_overrides,
        stable_trigger=stable_trigger,
        parameter_aliases=aliases,
        bundle_variable_defaults=bundle_vars,
        logger=logger,
    )
