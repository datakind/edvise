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

    apply_hook_spec(hitl_path, config_path, item_id, hook_spec, apply_to_group, resolved_by, …)
        — writes a generated HookSpec to the correct config field
        — when apply_to_group=True (term): fans out to HITL items sharing hook_group_id **and**
          every dataset listed in HITLItem.hook_group_tables for that group
        — optional materialize=True + repo_root= writes hook_spec.file as a Python module
        — optional merge_materialize_with=[...] merges other HookSpecs into one file before write
          (same path — use when multiple encodings share identity_hooks/.../term_hooks.py)

    validate_hook(hitl_path, config_path, item_id, hook_group_id)
        — compares each function's draft (AST) to the imported module's runtime signature

Notebook usage pattern:
    # Gate check before advancing
    check_gate("institutions/<institution_id>/identity_grain_hitl.json")

    # Apply terminal resolutions
    resolve_items(
        hitl_path="institutions/<institution_id>/identity_grain_hitl.json",
        config_path="institutions/<institution_id>/identity_grain_output.json",
        resolved_by="vish"
    )

    # Hook generation loop — one call per group
    hook_items = get_hook_items("institutions/<institution_id>/identity_grain_hitl.json")
    for item in hook_items:
        generated_spec = <LLM hook generation call using item.hitl_context>
        validate_hook(
            hitl_path="institutions/<institution_id>/identity_grain_hitl.json",
            config_path="institutions/<institution_id>/identity_grain_output.json",
            hook_group_id=item.hook_group_id or item.item_id,
        )
        apply_hook_spec(
            hitl_path="institutions/<institution_id>/identity_grain_hitl.json",
            config_path="institutions/<institution_id>/identity_grain_output.json",
            item_id=item.item_id,
            hook_spec=generated_spec,
            apply_to_group=True,
        )

    # Confirm gate clear before advancing to term normalization
    check_gate("institutions/<institution_id>/identity_grain_hitl.json")
"""

from __future__ import annotations

import importlib.util
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.signature_check import (
    signature_mismatches,
)
from edvise.genai.mapping.identity_agent.hitl.hook_generation.paths import (
    ensure_hook_spec_file,
    resolve_hook_module_path,
)
from edvise.genai.mapping.identity_agent.hitl.schemas import (
    GrainResolution,
    HITLDomain,
    HITLItem,
    HITLOption,
    InstitutionHITLItems,
    ReentryDepth,
    TermResolution,
)
from edvise.genai.mapping.shared.hitl import HITLBlockingError, raise_if_hitl_pending
from edvise.genai.mapping.shared.hitl.json_io import (
    read_pydantic_json,
    write_pydantic_json,
)
from edvise.genai.mapping.shared.hitl.run_log import RunEvent, append_run_log_event
from edvise.genai.mapping.shared.hitl.time import utc_now_iso
from edvise.genai.mapping.identity_agent.term_normalization.schemas import (
    SeasonMapEntry,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


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
    return read_pydantic_json(hitl_path, InstitutionHITLItems)


def _save_hitl(envelope: InstitutionHITLItems, hitl_path: Path) -> None:
    write_pydantic_json(hitl_path, envelope)


def _load_config(config_path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(config_path.read_text()))


def _save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def _append_run_log(
    run_log_path: Path,
    institution_id: str,
    item: HITLItem,
    selected: HITLOption,
    envelope: InstitutionHITLItems,
    resolved_by: str | None,
) -> None:
    """
    Append one RunEvent to run_log.json for this institution.
    Creates the file if it does not exist. Never overwrites existing events.
    """
    assert item.choice is not None
    event = RunEvent(
        timestamp=utc_now_iso(),
        resolved_by=resolved_by,
        agent="identity_agent",
        domain=envelope.domain,  # "grain" or "term"
        item_id=item.item_id,
        choice=item.choice,
        option_id=selected.option_id,
        reentry=selected.reentry.value,
    )
    append_run_log_event(run_log_path, institution_id, event)


# ---------------------------------------------------------------------------
# 1. check_gate
# ---------------------------------------------------------------------------


def check_gate(hitl_path: str | Path) -> None:
    """
    Raises HITLBlockingError if any items in the HITL file are still pending.
    Prints a clear summary and returns cleanly if all items have been reviewed.
    Safe to call repeatedly — never mutates.
    """
    hitl_path = Path(hitl_path)
    envelope = _load_hitl(hitl_path)

    if not envelope.items:
        print("✓ No HITL items — pipeline gate clear.")
        return

    if envelope.is_clear:
        print(f"✓ HITL gate clear — {len(envelope.items)} item(s) reviewed.")
        return

    raise_if_hitl_pending(
        envelope.pending,
        hitl_path=hitl_path,
        format_item=lambda i: (f"[{i.item_id}] {i.table} — {i.hitl_question[:80]}..."),
        instructions=(
            "  • Set 'choice' to the 1-based index of your selected option "
            "(1 … number of options).\n"
            "  • Re-run this cell."
        ),
    )


# ---------------------------------------------------------------------------
# 2. resolve_items
# ---------------------------------------------------------------------------


def resolve_items(
    hitl_path: str | Path,
    config_path: str | Path,
    resolved_by: str | None = None,
    run_log_path: str | Path | None = None,
) -> None:
    """
    For each HITLItem with ``choice`` set:
      - TERMINAL reentry  → applies full resolution to config
      - GENERATE_HOOK     → applies everything **except** ``hook_spec`` (e.g. ``season_map_replace``,
        ``exclude_tokens``, grain PK/dedup); ``hook_spec`` is written later via
        :func:`apply_hook_spec`. Surfaces the item via :func:`get_hook_items`.

    Writes updated config and HITL envelope back to disk.
    """
    hitl_path = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config = _load_config(config_path)

    for item in envelope.items:
        selected = _validate_selection(item)
        if selected is None:
            continue

        if selected.reentry == ReentryDepth.GENERATE_HOOK:
            # Still apply non-hook mutations (e.g. season_map_replace, exclude_tokens,
            # term_col_override, grain PK / dedup). hook_spec is deferred to apply_hook_spec.
            if selected.resolution is not None:
                coerced_gh = _coerce_resolution(item, selected.resolution)
                if isinstance(coerced_gh, TermResolution):
                    deferred_term = coerced_gh.model_copy(update={"hook_spec": None})
                    _apply_term_resolution_for_hook_group(
                        config, envelope, item, deferred_term
                    )
                elif isinstance(coerced_gh, GrainResolution):
                    deferred_grain = coerced_gh.model_copy(update={"hook_spec": None})
                    _apply_grain_resolution(config, item, deferred_grain)
            print(
                f"⚠  [{item.item_id}] Requires hook generation — "
                f"call get_hook_items() and apply_hook_spec() after hook gen."
            )
            continue

        coerced = _coerce_resolution(item, selected.resolution)
        if isinstance(coerced, GrainResolution):
            _apply_grain_resolution(config, item, coerced)
        elif isinstance(coerced, TermResolution):
            _apply_term_resolution_for_hook_group(config, envelope, item, coerced)
        else:
            raise HITLValidationError(
                f"[{item.item_id}] Internal error: coerced resolution has unexpected type "
                f"{type(coerced)}"
            )

        print(f"✓ [{item.item_id}] Applied option '{selected.option_id}'.")

        if run_log_path is not None:
            _append_run_log(
                Path(run_log_path),
                envelope.institution_id,
                item,
                selected,
                envelope,
                resolved_by,
            )

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
        hook_items = get_hook_items("institutions/<institution_id>/identity_grain_hitl.json")
        hook_items = get_hook_items("institutions/<institution_id>/identity_term_hitl.json")
    """
    hitl_path = Path(hitl_path)
    envelope = _load_hitl(hitl_path)

    seen_groups: set[str] = set()
    result: list[HITLItem] = []

    for item in envelope.items:
        if item.choice is None:
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


def _term_tables_for_hook_group(
    envelope: InstitutionHITLItems,
    group_id: str,
) -> list[str]:
    """
    Tables that should receive the same term hook_spec for ``group_id``.

    Union of (1) ``table`` for each IDENTITY_TERM HITL item with ``hook_group_id == group_id``
    and (2) every name in ``hook_group_tables`` on those items (explicit dataset list for the group).
    Ordering is stable (sorted table names).
    """
    tables: set[str] = set()
    for item in envelope.items:
        if item.domain == HITLDomain.IDENTITY_TERM and item.hook_group_id == group_id:
            tables.add(item.table)
            if item.hook_group_tables:
                tables.update(item.hook_group_tables)
    return sorted(tables)


def _term_target_tables_for_item(
    envelope: InstitutionHITLItems,
    item: HITLItem,
) -> list[str]:
    """
    Tables that should receive the same ``term_config`` mutations as :func:`apply_hook_spec`
    fan-out — union of ``hook_group_tables`` and each item's ``table`` for the group.
    """
    if item.domain != HITLDomain.IDENTITY_TERM:
        return [item.target.table]
    if item.hook_group_id:
        return _term_tables_for_hook_group(envelope, item.hook_group_id)
    return [item.target.table]


def _apply_term_resolution_for_hook_group(
    config: dict,
    envelope: InstitutionHITLItems,
    item: HITLItem,
    resolution: TermResolution,
) -> None:
    """Apply :func:`_apply_term_resolution` to every table in the item's term hook group."""
    for tbl in _term_target_tables_for_item(envelope, item):
        _apply_term_resolution(config, item, resolution, table=tbl)


def apply_hook_spec(
    hitl_path: str | Path,
    config_path: str | Path,
    item_id: str,
    hook_spec: HookSpec,
    apply_to_group: bool = False,
    resolved_by: str | None = None,
    run_log_path: str | Path | None = None,
    *,
    materialize: bool = False,
    repo_root: str | Path | None = None,
    merge_materialize_with: Sequence[HookSpec] | None = None,
) -> None:
    """
    Writes a generated HookSpec to the correct config field.

    When apply_to_group=True and the anchor item has hook_group_id:

    - **IDENTITY_TERM:** writes the same HookSpec to every table in the union of (a) HITL items
      with that ``hook_group_id`` and (b) dataset names listed in ``hook_group_tables`` on those
      items (Pass 2 should set ``hook_group_tables`` on the representative item when several
      logical datasets share one hook, using the same names as keys under ``datasets`` in
      ``identity_term_output.json``).
    - **IDENTITY_GRAIN:** writes to all HITL items sharing the same hook_group_id (unchanged).

    For IDENTITY_GRAIN items: writes hook_spec to dedup_policy.hook_spec
    For IDENTITY_TERM items: writes hook_spec to term_config.hook_spec and sets
    ``term_config.term_extraction`` to ``hook_required``. That value is the **runtime mode**
    for hook-driven extraction (year/season from the materialized module), not a “todo” flag —
    it stays ``hook_required`` after hooks are generated and materialized. Only
    ``standard`` is used for split ``year_col``/``season_col`` extraction without ``hook_spec``.

    When ``materialize`` is True, also writes a ``.py`` file at ``hook_spec.file``
    (relative to ``repo_root``) so :func:`validate_hook` can import it. Pass
    ``repo_root`` = the school's ``bronze_volumes_path`` (Unity Catalog volume root: same base as
    ``cleaned/`` and ``enriched_schema_contracts/``), unless you intentionally use a git checkout
    as the root for relative paths.

    When several HITL resolutions contribute different functions to the **same** module path
    (e.g. opaque term extractors + date-string extractors both under ``term_hooks.py``), pass
    ``merge_materialize_with=[HookSpec(...), ...]`` so the written file contains every function.
    Config fields still receive only ``hook_spec`` (per item); merge applies **only** to the
    on-disk module. Order: primary ``hook_spec`` first, then ``merge_materialize_with`` in order.

    Notebook usage:
        apply_hook_spec(
            hitl_path="institutions/<institution_id>/identity_grain_hitl.json",
            config_path="institutions/<institution_id>/identity_grain_output.json",
            item_id="<institution_id>_demo_dedup",
            hook_spec=generated_spec,
            apply_to_group=True,
            resolved_by="dk",
            materialize=True,
            repo_root="/path/to/repo",
        )
        apply_hook_spec(
            hitl_path="institutions/<institution_id>/identity_term_hitl.json",
            config_path="institutions/<institution_id>/identity_term_output.json",
            item_id="<institution_id>_student_term",
            hook_spec=generated_spec,
            apply_to_group=True,
            resolved_by="dk",
            materialize=True,
            repo_root="/path/to/repo",
        )
    """
    hitl_path = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config = _load_config(config_path)

    anchor = _find_item(envelope, item_id)
    hook_spec = ensure_hook_spec_file(
        hook_spec,
        institution_id=envelope.institution_id,
        domain=anchor.domain,
    )

    group_id = anchor.hook_group_id
    if anchor.domain == HITLDomain.IDENTITY_TERM and apply_to_group and group_id:
        tables = _term_tables_for_hook_group(envelope, group_id)
        if not tables:
            tables = [anchor.table]
        for table in tables:
            term_cfg = _get_nested(config, table, "term_config", anchor.item_id)
            _apply_term_hook_spec_dict(
                term_cfg,
                hook_spec,
                item_id=f"{anchor.item_id}:{table}",
                institution_id=envelope.institution_id,
            )
            print(
                f"✓ [{anchor.item_id}] hook_spec written to '{anchor.domain.value}' config "
                f"for table '{table}'."
            )
        for item in _group_members(envelope, group_id):
            if run_log_path is not None:
                selected = item.selected_option()
                if selected:
                    _append_run_log(
                        Path(run_log_path),
                        envelope.institution_id,
                        item,
                        selected,
                        envelope,
                        resolved_by,
                    )
    else:
        target_items = (
            _group_members(envelope, group_id)
            if apply_to_group and group_id
            else [anchor]
        )

        for item in target_items:
            _write_hook_spec_to_config(config, item, hook_spec)
            print(
                f"✓ [{item.item_id}] hook_spec written to '{item.domain.value}' config for table '{item.table}'."
            )

            if run_log_path is not None:
                selected = item.selected_option()
                if selected:
                    _append_run_log(
                        Path(run_log_path),
                        envelope.institution_id,
                        item,
                        selected,
                        envelope,
                        resolved_by,
                    )

    _save_config(config, config_path)
    _save_hitl(envelope, hitl_path)
    print(f"\nUpdated config written to {config_path.name}.")

    if materialize:
        if repo_root is None:
            raise ValueError(
                "apply_hook_spec(..., materialize=True) requires repo_root"
            )
        from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
            materialize_hook_spec_to_file,
            merge_hook_specs,
        )

        spec_to_write = hook_spec
        if merge_materialize_with:
            spec_to_write = merge_hook_specs(
                hook_spec, *merge_materialize_with, repo_root=repo_root
            )

        materialize_hook_spec_to_file(
            spec_to_write,
            repo_root=repo_root,
            domain=anchor.domain,
        )


# ---------------------------------------------------------------------------
# 5. validate_hook
# ---------------------------------------------------------------------------


def validate_hook(
    config_path: str | Path,
    hitl_path: str | Path,
    *,
    item_id: str | None = None,
    hook_group_id: str | None = None,
    hook_file_root: str | Path | None = None,
) -> None:
    """
    For every domain: compares each function's ``draft`` (AST) to the imported function's runtime
    signature — parameter names/order and return annotation when the draft includes ``->``.
    Validation stack aligns with materialization: :func:`ast.parse` and optional pyflakes on the
    written module, then this signature check after import.

    Raises HookValidationError on failure.

    Pass either item_id or hook_group_id — hook_group_id uses the first group
    member as the representative (all share the same hook file and functions).

    ``hook_file_root``: when set (typically ``bronze_volumes_path``), ``hook_spec.file`` is
    resolved under that directory — same as :func:`materialize_hook_spec_to_file` ``repo_root``.
    When omitted, ``hook_spec.file`` is resolved relative to the process current working directory.

    Notebook usage:
        validate_hook(
            config_path="institutions/<institution_id>/identity_grain_output.json",
            hitl_path="institutions/<institution_id>/identity_grain_hitl.json",
            hook_group_id="shared_dedup_format_a",
        )
        validate_hook(
            config_path="institutions/<institution_id>/identity_term_output.json",
            hitl_path="institutions/<institution_id>/identity_term_hitl.json",
            hook_group_id="shared_term_format_a",
        )
    """
    if item_id is None and hook_group_id is None:
        raise ValueError("Provide either item_id or hook_group_id.")

    hitl_path = Path(hitl_path)
    config_path = Path(config_path)

    envelope = _load_hitl(hitl_path)
    config = _load_config(config_path)

    # Resolve representative item
    if hook_group_id:
        members = _group_members(envelope, hook_group_id)
        if not members:
            raise HITLValidationError(
                f"No items found for hook_group_id='{hook_group_id}'."
            )
        item = members[0]
    else:
        if item_id is None:
            raise ValueError("Provide either item_id or hook_group_id.")
        item = _find_item(envelope, item_id)

    # Load hook_spec from config
    hook_spec_dict = _read_hook_spec_from_config(config, item)
    if hook_spec_dict is None:
        raise HITLValidationError(
            f"[{item.item_id}] hook_spec is null in config — "
            f"run apply_hook_spec() before validate_hook()."
        )

    if hook_file_root is not None:
        try:
            hook_file = resolve_hook_module_path(
                hook_spec_dict["file"], root=hook_file_root
            )
        except ValueError as e:
            raise HookValidationError(str(e)) from e
    else:
        hook_file = Path(hook_spec_dict["file"])
    if not hook_file.exists():
        raise HookValidationError(f"Hook file not found: {hook_file}")

    # Dynamically import hook module
    spec = importlib.util.spec_from_file_location("_hook_module", hook_file)
    if spec is None or spec.loader is None:
        raise HookValidationError(f"Could not load module spec for {hook_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    failures: list[str] = []
    for fn_spec in hook_spec_dict["functions"]:
        name = fn_spec["name"]
        draft = fn_spec.get("draft")

        fn = getattr(module, name, None)
        if fn is None:
            failures.append(f"[{name}] Function not found in {hook_file}.")
            continue

        failures.extend(signature_mismatches(fn, expected_name=name, draft=draft))

    if failures:
        raise HookValidationError(
            f"Hook validation failed for '{hook_file}':\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    print(f"✓ Hook signatures verified for {hook_file} ({item.domain.value}).")


# ---------------------------------------------------------------------------
# Resolution handlers — config shape must match GrainContract / TermContract validators
# ---------------------------------------------------------------------------


def _apply_grain_hook_spec_dict(
    grain_cfg: dict, hook_spec: HookSpec, *, institution_id: str
) -> None:
    """
    Write ``dedup_policy.hook_spec`` and set ``strategy='policy_required'``.

    Clears sort/keep fields that only apply to temporal_collapse so
    :class:`~edvise.genai.mapping.identity_agent.grain_inference.schemas.DedupPolicy` reloads cleanly.
    """
    hook_spec = ensure_hook_spec_file(
        hook_spec,
        institution_id=institution_id,
        domain=HITLDomain.IDENTITY_GRAIN,
    )
    dp = grain_cfg.setdefault("dedup_policy", {})
    dp["strategy"] = "policy_required"
    dp["hook_spec"] = hook_spec.model_dump(mode="json")
    dp["sort_by"] = None
    dp["sort_ascending"] = None
    dp["keep"] = None


def _apply_term_hook_spec_dict(
    term_cfg: dict, hook_spec: HookSpec, *, item_id: str, institution_id: str
) -> None:
    """
    Write ``hook_spec`` and set ``term_extraction='hook_required'``.

    ``hook_required`` means “term extraction uses these hook functions” for the combined
    ``term_col`` path — it remains correct after materialization; it is not upgraded to
    ``standard`` (``standard`` is for split year/season columns without ``hook_spec``).

    TermOrderConfig forbids ``hook_spec`` alongside ``year_col``/``season_col``; when both split
    columns and ``term_col`` are present, split columns are cleared (combined-column hook path).
    """
    hook_spec = ensure_hook_spec_file(
        hook_spec,
        institution_id=institution_id,
        domain=HITLDomain.IDENTITY_TERM,
    )
    yc = term_cfg.get("year_col")
    sc = term_cfg.get("season_col")
    tc = term_cfg.get("term_col")
    has_split = yc is not None and sc is not None
    has_partial_split = (yc is None) != (sc is None)
    if has_partial_split:
        raise HITLValidationError(
            f"[{item_id}] term_config has only one of year_col/season_col — "
            "fix the config before writing hook_spec."
        )
    if has_split:
        if tc is None:
            raise HITLValidationError(
                f"[{item_id}] Cannot write term hook_spec while only year_col/season_col are set. "
                "Set term_col (e.g. term_col_override in the same resolution) so the hook targets "
                "a combined term column, or drop split columns manually."
            )
        term_cfg["year_col"] = None
        term_cfg["season_col"] = None
    elif tc is None:
        raise HITLValidationError(
            f"[{item_id}] term_config must set term_col before hook_spec "
            "(split year/season without term_col is unsupported for hook extraction)."
        )

    term_cfg["term_extraction"] = "hook_required"
    term_cfg["hook_spec"] = hook_spec.model_dump(mode="json")


def _apply_grain_resolution(
    config: dict,
    item: HITLItem,
    resolution: GrainResolution,
) -> None:
    table = item.target.table
    grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)

    if resolution.candidate_key_override:
        grain_cfg["post_clean_primary_key"] = resolution.candidate_key_override
        grain_cfg["join_keys_for_2a"] = resolution.candidate_key_override
        print(
            f"  → post_clean_primary_key overridden: {resolution.candidate_key_override}"
        )

    if resolution.dedup_strategy:
        grain_cfg["dedup_policy"]["strategy"] = resolution.dedup_strategy
        grain_cfg["dedup_policy"]["sort_by"] = resolution.dedup_sort_by
        grain_cfg["dedup_policy"]["sort_ascending"] = resolution.dedup_sort_ascending
        grain_cfg["dedup_policy"]["keep"] = resolution.dedup_keep
        print(
            "  → dedup_policy updated: "
            f"strategy={resolution.dedup_strategy}, "
            f"sort_by={resolution.dedup_sort_by}, "
            f"ascending={resolution.dedup_sort_ascending}"
        )

    if resolution.hook_spec is not None:
        _apply_grain_hook_spec_dict(
            grain_cfg, resolution.hook_spec, institution_id=item.institution_id
        )
        print("  → dedup_policy: hook_spec set, strategy=policy_required")


def _apply_term_resolution(
    config: dict,
    item: HITLItem,
    resolution: TermResolution,
    *,
    table: str | None = None,
) -> None:
    tbl = table if table is not None else item.target.table
    term_cfg = _get_nested(config, tbl, "term_config", item.item_id)

    if resolution.exclude_tokens:
        existing = term_cfg.setdefault("exclude_tokens", [])
        for token in resolution.exclude_tokens:
            if token not in existing:
                existing.append(token)
        print(f"  → [{tbl}] exclude_tokens appended: {resolution.exclude_tokens}")

    if resolution.season_map_replace is not None:
        new_map: list[dict] = []
        for raw_entry in resolution.season_map_replace:
            entry = SeasonMapEntry.model_validate(raw_entry)
            new_map.append(entry.model_dump(mode="json"))
        term_cfg["season_map"] = new_map
        print(f"  → [{tbl}] season_map replaced ({len(new_map)} entries)")

    if resolution.season_map_append:
        existing = term_cfg.setdefault("season_map", [])
        for raw_entry in resolution.season_map_append:
            entry = SeasonMapEntry.model_validate(raw_entry)
            existing.append(entry.model_dump(mode="json"))
        print(f"  → [{tbl}] season_map extended: {resolution.season_map_append}")

    if resolution.term_col_override:
        term_cfg["term_col"] = resolution.term_col_override
        term_cfg["year_col"] = None
        term_cfg["season_col"] = None
        print(f"  → [{tbl}] term_col overridden: {resolution.term_col_override}")

    if resolution.hook_spec is not None:
        _apply_term_hook_spec_dict(
            term_cfg,
            resolution.hook_spec,
            item_id=item.item_id,
            institution_id=item.institution_id,
        )
        print(
            f"  → [{tbl}] term_config: hook_spec set; term_extraction=hook_required "
            f"(runtime mode for hook extractors — unchanged after materialize) [{item.item_id}]"
        )


def _write_hook_spec_to_config(
    config: dict,
    item: HITLItem,
    hook_spec: HookSpec,
) -> None:
    table = item.target.table
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)
        _apply_grain_hook_spec_dict(
            grain_cfg, hook_spec, institution_id=item.institution_id
        )
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        _apply_term_hook_spec_dict(
            term_cfg,
            hook_spec,
            item_id=item.item_id,
            institution_id=item.institution_id,
        )
    else:
        raise HITLValidationError(
            f"[{item.item_id}] apply_hook_spec only handles IDENTITY_GRAIN and "
            f"IDENTITY_TERM domains, got '{item.domain.value}'."
        )


def _read_hook_spec_from_config(config: dict, item: HITLItem) -> dict | None:
    table = item.target.table
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)
        raw = grain_cfg.get("dedup_policy", {}).get("hook_spec")
        return cast(dict[str, Any] | None, raw if isinstance(raw, dict) else None)
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        return term_cfg.get("hook_spec") if term_cfg else None
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_resolution(
    item: HITLItem,
    resolution: object,
) -> GrainResolution | TermResolution:
    """
    JSON-loaded HITL stores ``option.resolution`` as a plain ``dict``.
    :func:`resolve_items` must validate into :class:`GrainResolution` or
    :class:`TermResolution` before applying mutations.
    """
    if resolution is None:
        raise HITLValidationError(
            f"[{item.item_id}] resolution is null — cannot apply (use reentry=generate_hook "
            f"for hook generation, or pick a non-custom option)."
        )
    if isinstance(resolution, GrainResolution):
        return resolution
    if isinstance(resolution, TermResolution):
        return resolution
    if isinstance(resolution, dict):
        if item.domain == HITLDomain.IDENTITY_GRAIN:
            return GrainResolution.model_validate(resolution)
        if item.domain == HITLDomain.IDENTITY_TERM:
            return TermResolution.model_validate(resolution)
        raise HITLValidationError(
            f"[{item.item_id}] Unsupported HITL domain for resolution dict: {item.domain!r}"
        )
    raise HITLValidationError(
        f"[{item.item_id}] resolution must be dict, GrainResolution, or TermResolution; "
        f"got {type(resolution).__name__}"
    )


def _validate_selection(item: HITLItem) -> HITLOption | None:
    """Returns None when no choice set — caller skips unreviewed items."""
    return item.selected_option()


def _find_item(envelope: InstitutionHITLItems, item_id: str) -> HITLItem:
    for item in envelope.items:
        if item.item_id == item_id:
            return item
    raise HITLValidationError(
        f"item_id='{item_id}' not found. "
        f"Available: {[i.item_id for i in envelope.items]}"
    )


def _group_members(
    envelope: InstitutionHITLItems, hook_group_id: str
) -> list[HITLItem]:
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
    return cast(dict[str, Any], cfg)
