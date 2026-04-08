"""
hitl_resolver.py

Public API:
    check_gate(hitl_path)
        — blocks pipeline if any HITLItems are still pending

    resolve_items(hitl_path, config_path, resolved_by)
        — applies terminal resolutions to grain/term config

    get_hook_items(hitl_path)
        — returns one representative HITLItem per hook_group_id (or per ungrouped item)
        — these require a hook generation LLM call before the pipeline can advance

    apply_hook_spec(hitl_path, config_path, item_id, hook_spec, apply_to_group, resolved_by)
        — writes a generated HookSpec to the correct config field
        — when apply_to_group=True, fans out to all items sharing the same hook_group_id

    validate_hook(hitl_path, config_path, item_id, hook_group_id)
        — unit tests a generated hook against example_input/example_output from HookFunctionSpec

Notebook usage pattern:
    # Gate check before advancing
    check_gate("institutions/jjc/identity_grain_hitl.json")

    # Apply terminal resolutions
    resolve_items(
        hitl_path="institutions/jjc/identity_grain_hitl.json",
        config_path="institutions/jjc/identity_grain_output.json",
        resolved_by="vish"
    )

    # Hook generation loop — one call per group
    hook_items = get_hook_items("institutions/jjc/identity_grain_hitl.json")
    for item in hook_items:
        generated_spec = <LLM hook generation call using item.hitl_context>
        validate_hook(
            hitl_path="institutions/jjc/identity_grain_hitl.json",
            config_path="institutions/jjc/identity_grain_output.json",
            hook_group_id=item.hook_group_id or item.item_id,
        )
        apply_hook_spec(
            hitl_path="institutions/jjc/identity_grain_hitl.json",
            config_path="institutions/jjc/identity_grain_output.json",
            item_id=item.item_id,
            hook_spec=generated_spec,
            apply_to_group=True,
        )

    # Confirm gate clear before advancing to term normalization
    check_gate("institutions/jjc/identity_grain_hitl.json")
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from edvise.genai.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    HITLResolution,
    HITLStatus,
    InstitutionHITLItems,
    ReentryDepth,
    TermResolution,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class HITLBlockingError(Exception):
    """Raised when unresolved HITL items are blocking pipeline progression."""
    pass


class HITLValidationError(Exception):
    """Raised when a HITL file is malformed or reviewer left multiple options."""
    pass


class HookValidationError(Exception):
    """Raised when a generated hook fails unit testing against example inputs."""
    pass


# ---------------------------------------------------------------------------
# Load / save helpers
# ---------------------------------------------------------------------------

def _load_hitl(hitl_path: Path) -> InstitutionHITLItems:
    if not hitl_path.exists():
        raise FileNotFoundError(f"HITL file not found: {hitl_path}")
    return InstitutionHITLItems.model_validate_json(hitl_path.read_text())


def _save_hitl(envelope: InstitutionHITLItems, hitl_path: Path) -> None:
    hitl_path.write_text(envelope.model_dump_json(indent=2))


def _load_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text())


def _save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# 1. check_gate
# ---------------------------------------------------------------------------

def check_gate(hitl_path: str | Path) -> None:
    """
    Raises HITLBlockingError if any items in the HITL file are still pending.
    Prints a clear summary and returns cleanly if all items are resolved or skipped.
    Safe to call repeatedly — never mutates.
    """
    hitl_path = Path(hitl_path)
    envelope  = _load_hitl(hitl_path)

    if not envelope.items:
        print("✓ No HITL items — pipeline gate clear.")
        return

    if envelope.is_clear:
        resolved = sum(1 for i in envelope.items if i.status == HITLStatus.RESOLVED)
        skipped  = sum(1 for i in envelope.items if i.status == HITLStatus.SKIPPED)
        print(f"✓ HITL gate clear — {resolved} resolved, {skipped} skipped.")
        return

    summary = "\n".join(
        f"  [{i.item_id}] {i.table} — {i.hitl_question[:80]}..."
        for i in envelope.pending
    )
    raise HITLBlockingError(
        f"\n{len(envelope.pending)} unresolved HITL item(s) blocking pipeline:\n{summary}\n\n"
        f"To resolve, edit {hitl_path.name}:\n"
        f"  • Delete 2 of the 3 options, leaving only your chosen option.\n"
        f"  • Set status to 'resolved'.\n"
        f"  • Re-run this cell."
    )


# ---------------------------------------------------------------------------
# 2. resolve_items
# ---------------------------------------------------------------------------

def resolve_items(
    hitl_path:   str | Path,
    config_path: str | Path,
    resolved_by: str | None = None,
) -> None:
    """
    For each pending HITLItem where reviewer left exactly one option:
      - TERMINAL reentry  → applies resolution to config, marks item resolved
      - GENERATE_HOOK     → skips config mutation, surfaces via get_hook_items()

    Writes updated config and HITL envelope back to disk.
    """
    hitl_path   = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config   = _load_config(config_path)

    for item in envelope.items:
        if item.status != HITLStatus.PENDING:
            continue

        selected = _validate_selection(item)

        if selected.reentry == ReentryDepth.GENERATE_HOOK:
            print(
                f"⚠  [{item.item_id}] Requires hook generation — "
                f"call get_hook_items() and apply_hook_spec() after hook gen."
            )
            continue

        if isinstance(selected.resolution, GrainResolution):
            _apply_grain_resolution(config, item, selected.resolution)
        elif isinstance(selected.resolution, TermResolution):
            _apply_term_resolution(config, item, selected.resolution)

        item.status     = HITLStatus.RESOLVED
        item.resolution = _make_resolution(selected.option_id, resolved_by)
        print(f"✓ [{item.item_id}] Resolved via '{selected.option_id}'.")

    _save_config(config, config_path)
    _save_hitl(envelope, hitl_path)
    print(f"\nUpdated config written to {config_path.name}.")


# ---------------------------------------------------------------------------
# 3. get_hook_items
# ---------------------------------------------------------------------------

def get_hook_items(hitl_path: str | Path) -> list[HITLItem]:
    """
    Returns one representative HITLItem per hook group (or per ungrouped item)
    where the selected option has reentry=GENERATE_HOOK.

    Items sharing a hook_group_id need only one hook generation call.
    Pass the representative item's hitl_context to the hook generation prompt.
    Pass the generated HookSpec to apply_hook_spec(apply_to_group=True) to fan
    the result out to all members of the group.

    Notebook usage:
        hook_items = get_hook_items("institutions/jjc/identity_grain_hitl.json")
        hook_items = get_hook_items("institutions/jjc/identity_term_hitl.json")
    """
    hitl_path = Path(hitl_path)
    envelope  = _load_hitl(hitl_path)

    seen_groups: set[str] = set()
    result: list[HITLItem] = []

    for item in envelope.items:
        if item.status != HITLStatus.PENDING:
            continue
        selected = item.selected_option()
        if not selected or selected.reentry != ReentryDepth.GENERATE_HOOK:
            continue

        # Deduplicate by hook_group_id — first encountered is representative
        group_key = item.hook_group_id or item.item_id
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        result.append(item)

    return result


# ---------------------------------------------------------------------------
# 4. apply_hook_spec
# ---------------------------------------------------------------------------

def apply_hook_spec(
    hitl_path:      str | Path,
    config_path:    str | Path,
    item_id:        str,
    hook_spec:      HookSpec,
    apply_to_group: bool = False,
    resolved_by:    str | None = None,
) -> None:
    """
    Writes a generated HookSpec to the correct config field and marks item(s) resolved.

    When apply_to_group=True, writes the same HookSpec to all items sharing
    the same hook_group_id as the named item_id. Use this when multiple tables
    share the same term encoding or dedup pattern.

    For IDENTITY_GRAIN items: writes hook_spec to dedup_policy.hook_spec
    For IDENTITY_TERM items:  writes hook_spec to term_config.hook_spec

    Notebook usage:
        apply_hook_spec(
            hitl_path="institutions/jjc/identity_grain_hitl.json",
            config_path="institutions/jjc/identity_grain_output.json",
            item_id="jjc_demo_dedup",
            hook_spec=generated_spec,
            apply_to_group=True,
            resolved_by="vish"
        )
        apply_hook_spec(
            hitl_path="institutions/jjc/identity_term_hitl.json",
            config_path="institutions/jjc/identity_term_output.json",
            item_id="jjc_student_term",
            hook_spec=generated_spec,
            apply_to_group=True,
            resolved_by="vish"
        )
    """
    hitl_path   = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config   = _load_config(config_path)

    # Resolve target items
    anchor     = _find_item(envelope, item_id)
    group_id   = anchor.hook_group_id
    target_items = (
        _group_members(envelope, group_id)
        if apply_to_group and group_id
        else [anchor]
    )

    for item in target_items:
        _write_hook_spec_to_config(config, item, hook_spec)
        item.status     = HITLStatus.RESOLVED
        item.resolution = _make_resolution(
            item.selected_option().option_id if item.selected_option() else "hook_applied",
            resolved_by,
        )
        print(f"✓ [{item.item_id}] hook_spec written to '{item.domain.value}' config for table '{item.table}'.")

    _save_config(config, config_path)
    _save_hitl(envelope, hitl_path)
    print(f"\nUpdated config written to {config_path.name}.")


# ---------------------------------------------------------------------------
# 5. validate_hook
# ---------------------------------------------------------------------------

def validate_hook(
    config_path:    str | Path,
    hitl_path:      str | Path,
    *,
    item_id:        str | None = None,
    hook_group_id:  str | None = None,
) -> None:
    """
    Unit tests a generated hook against example_input / example_output from
    each HookFunctionSpec. Dynamically imports the hook file and calls each
    function. Raises HookValidationError on any failure.

    Pass either item_id or hook_group_id — hook_group_id uses the first group
    member as the representative (all share the same hook file and functions).

    Notebook usage:
        validate_hook(
            config_path="institutions/jjc/identity_grain_output.json",
            hitl_path="institutions/jjc/identity_grain_hitl.json",
            hook_group_id="jjc_dedup_format_a",
        )
        validate_hook(
            config_path="institutions/jjc/identity_term_output.json",
            hitl_path="institutions/jjc/identity_term_hitl.json",
            hook_group_id="jjc_term_format_a",
        )
    """
    if item_id is None and hook_group_id is None:
        raise ValueError("Provide either item_id or hook_group_id.")

    hitl_path   = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config   = _load_config(config_path)

    # Resolve representative item
    if hook_group_id:
        members = _group_members(envelope, hook_group_id)
        if not members:
            raise HITLValidationError(f"No items found for hook_group_id='{hook_group_id}'.")
        item = members[0]
    else:
        item = _find_item(envelope, item_id)

    # Load hook_spec from config
    hook_spec_dict = _read_hook_spec_from_config(config, item)
    if hook_spec_dict is None:
        raise HITLValidationError(
            f"[{item.item_id}] hook_spec is null in config — "
            f"run apply_hook_spec() before validate_hook()."
        )

    hook_file = Path(hook_spec_dict["file"])
    if not hook_file.exists():
        raise HookValidationError(f"Hook file not found: {hook_file}")

    # Dynamically import hook module
    spec   = importlib.util.spec_from_file_location("_hook_module", hook_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Test each function
    failures: list[str] = []
    for fn_spec in hook_spec_dict["functions"]:
        name           = fn_spec["name"]
        example_input  = fn_spec.get("example_input")
        example_output = fn_spec.get("example_output")
        expected_type  = fn_spec.get("expected_type")

        if example_input is None:
            print(f"  ⚠  [{name}] No example_input — skipping.")
            continue

        fn = getattr(module, name, None)
        if fn is None:
            failures.append(f"[{name}] Function not found in {hook_file}.")
            continue

        try:
            result = fn(example_input)
        except Exception as e:
            failures.append(f"[{name}] Raised exception on input {example_input!r}: {e}")
            continue

        if expected_type:
            actual_type = type(result).__name__
            if actual_type != expected_type:
                failures.append(
                    f"[{name}] Expected return type '{expected_type}', got '{actual_type}'."
                )

        if example_output is not None:
            # Coerce example_output to result type for comparison
            try:
                coerced = type(result)(example_output)
            except (ValueError, TypeError):
                coerced = example_output
            if result != coerced:
                failures.append(
                    f"[{name}] Expected output {coerced!r}, got {result!r} "
                    f"for input {example_input!r}."
                )
            else:
                print(f"  ✓ [{name}] {example_input!r} → {result!r}")

    if failures:
        raise HookValidationError(
            f"Hook validation failed for '{hook_file}':\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    print(f"✓ All hook functions validated for {hook_file}.")


# ---------------------------------------------------------------------------
# Resolution handlers
# ---------------------------------------------------------------------------

def _apply_grain_resolution(
    config:     dict,
    item:       HITLItem,
    resolution: GrainResolution,
) -> None:
    table     = item.target.table
    grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)

    if resolution.candidate_key_override:
        grain_cfg["post_clean_primary_key"] = resolution.candidate_key_override
        grain_cfg["join_keys_for_2a"]       = resolution.candidate_key_override
        print(f"  → post_clean_primary_key overridden: {resolution.candidate_key_override}")

    if resolution.dedup_strategy:
        grain_cfg["dedup_policy"]["strategy"] = resolution.dedup_strategy
        grain_cfg["dedup_policy"]["sort_by"]  = resolution.dedup_sort_by
        grain_cfg["dedup_policy"]["keep"]     = resolution.dedup_keep
        print(f"  → dedup_policy updated: strategy={resolution.dedup_strategy}")


def _apply_term_resolution(
    config:     dict,
    item:       HITLItem,
    resolution: TermResolution,
) -> None:
    table    = item.target.table
    term_cfg = _get_nested(config, table, "term_config", item.item_id)

    if resolution.exclude_tokens:
        existing = term_cfg.setdefault("exclude_tokens", [])
        for token in resolution.exclude_tokens:
            if token not in existing:
                existing.append(token)
        print(f"  → exclude_tokens appended: {resolution.exclude_tokens}")

    if resolution.season_map_append:
        existing = term_cfg.setdefault("season_map", [])
        existing.extend(resolution.season_map_append)
        print(f"  → season_map extended: {resolution.season_map_append}")

    if resolution.term_col_override:
        term_cfg["term_col"] = resolution.term_col_override
        print(f"  → term_col overridden: {resolution.term_col_override}")


def _write_hook_spec_to_config(
    config:    dict,
    item:      HITLItem,
    hook_spec: HookSpec,
) -> None:
    table = item.target.table
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)
        grain_cfg["dedup_policy"]["hook_spec"] = hook_spec.model_dump()
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        term_cfg["hook_spec"] = hook_spec.model_dump()
    else:
        raise HITLValidationError(
            f"[{item.item_id}] apply_hook_spec only handles IDENTITY_GRAIN and "
            f"IDENTITY_TERM domains, got '{item.domain.value}'."
        )


def _read_hook_spec_from_config(config: dict, item: HITLItem) -> dict | None:
    table = item.target.table
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)
        return grain_cfg.get("dedup_policy", {}).get("hook_spec")
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        return term_cfg.get("hook_spec") if term_cfg else None
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_selection(item: HITLItem) -> HITLOption:
    selected = item.selected_option()
    if selected is None:
        raise HITLValidationError(
            f"[{item.item_id}] Expected exactly 1 option remaining after review, "
            f"got {len(item.options)}. Delete all but your chosen option."
        )
    return selected


def _find_item(envelope: InstitutionHITLItems, item_id: str) -> HITLItem:
    for item in envelope.items:
        if item.item_id == item_id:
            return item
    raise HITLValidationError(
        f"item_id='{item_id}' not found. "
        f"Available: {[i.item_id for i in envelope.items]}"
    )


def _group_members(envelope: InstitutionHITLItems, hook_group_id: str) -> list[HITLItem]:
    """Returns all items sharing the given hook_group_id."""
    return [i for i in envelope.items if i.hook_group_id == hook_group_id]


def _get_nested(config: dict, table: str, config_key: str, item_id: str) -> dict:
    """
    Navigate config["datasets"][table][config_key] with a clear error if missing.
    config_key is one of: "grain_contract", "term_config"
    """
    cfg = config.get("datasets", {}).get(table, {}).get(config_key)
    if cfg is None:
        raise HITLValidationError(
            f"[{item_id}] No '{config_key}' found for table '{table}' in config. "
            f"Available tables: {list(config.get('datasets', {}).keys())}"
        )
    return cfg


def _make_resolution(option_id: str, resolved_by: str | None) -> HITLResolution:
    return HITLResolution(
        selected_option_id=option_id,
        resolved_by=resolved_by,
        resolved_at=datetime.now(timezone.utc).isoformat(),
    )