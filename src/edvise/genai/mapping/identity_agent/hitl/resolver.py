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
        — when apply_to_group=True, fans out to all items sharing the same hook_group_id
        — optional materialize=True + repo_root= writes hook_spec.file as a Python module

    validate_hook(hitl_path, config_path, item_id, hook_group_id)
        — unit tests a generated hook against example_input/example_output from HookFunctionSpec

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
from datetime import datetime, timezone
from pathlib import Path

from edvise.genai.mapping.identity_agent.grain_inference.schemas import HookSpec
from edvise.genai.mapping.identity_agent.hitl.hook_generation.literals import (
    coerce_hook_example_value,
)
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
    RunEvent,
    RunLog,
    TermResolution,
)
from edvise.genai.mapping.identity_agent.term_normalization.schemas import SeasonMapEntry

logger = logging.getLogger(__name__)


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
    if run_log_path.exists():
        run_log = RunLog.model_validate_json(run_log_path.read_text())
    else:
        run_log = RunLog(institution_id=institution_id)

    event = RunEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        resolved_by=resolved_by,
        agent="identity_agent",
        domain=envelope.domain,  # "grain" or "term"
        item_id=item.item_id,
        choice=item.choice,
        option_id=selected.option_id,
        reentry=selected.reentry.value,
    )
    run_log.events.append(event)
    run_log_path.write_text(run_log.model_dump_json(indent=2))


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

    summary = "\n".join(
        f"  [{i.item_id}] {i.table} — {i.hitl_question[:80]}..."
        for i in envelope.pending
    )
    raise HITLBlockingError(
        f"\n{len(envelope.pending)} unreviewed HITL item(s) blocking pipeline:\n{summary}\n\n"
        f"To resolve, edit {hitl_path.name}:\n"
        f"  • Set 'choice' to the 1-based index of your selected option (1 … number of options).\n"
        f"  • Re-run this cell."
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
      - TERMINAL reentry  → applies resolution to config
      - GENERATE_HOOK     → skips config mutation, surfaces via get_hook_items()

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
            print(
                f"⚠  [{item.item_id}] Requires hook generation — "
                f"call get_hook_items() and apply_hook_spec() after hook gen."
            )
            continue

        if isinstance(selected.resolution, GrainResolution):
            _apply_grain_resolution(config, item, selected.resolution)
        elif isinstance(selected.resolution, TermResolution):
            _apply_term_resolution(config, item, selected.resolution)

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
) -> None:
    """
    Writes a generated HookSpec to the correct config field.

    When apply_to_group=True, writes the same HookSpec to all items sharing
    the same hook_group_id as the named item_id. Use this when multiple tables
    share the same term encoding or dedup pattern.

    For IDENTITY_GRAIN items: writes hook_spec to dedup_policy.hook_spec
    For IDENTITY_TERM items:  writes hook_spec to term_config.hook_spec

    When ``materialize`` is True, also writes a ``.py`` file at ``hook_spec.file``
    (relative to ``repo_root``) so :func:`validate_hook` can import it. Pass
    ``repo_root`` = the school's ``bronze_volumes_path`` (Unity Catalog volume root: same base as
    ``cleaned/`` and ``enriched_schema_contracts/``), unless you intentionally use a git checkout
    as the root for relative paths.

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

    # Resolve target items
    group_id = anchor.hook_group_id
    target_items = (
        _group_members(envelope, group_id) if apply_to_group and group_id else [anchor]
    )

    for item in target_items:
        _write_hook_spec_to_config(config, item, hook_spec)
        # choice already set by reviewer — no status to update
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
            raise ValueError("apply_hook_spec(..., materialize=True) requires repo_root")
        from edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize import (
            materialize_hook_spec_to_file,
        )

        materialize_hook_spec_to_file(
            hook_spec,
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

    For **identity_term** items only: also runs each function with ``example_input`` /
    ``example_output`` coerced via :func:`ast.literal_eval`. If the return value does not match
    ``example_output``, a **warning** is logged (reviewer may fix examples); **execution errors**
    still fail validation. Same spirit as post-materialize smoke tests in
    :func:`~edvise.genai.mapping.identity_agent.hitl.hook_generation.materialize.materialize_hook_spec_to_file`.

    For **identity_grain** (and future non-term domains such as transform hooks):
    ``example_input`` / ``example_output`` are documentation only — no literal_eval or execution.

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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def _call_with_literal_input(fn, parsed_in):
        if isinstance(parsed_in, tuple):
            return fn(*parsed_in)
        return fn(parsed_in)

    run_literal_tests = item.domain == HITLDomain.IDENTITY_TERM
    term_example_mismatch_logged = False

    failures: list[str] = []
    for fn_spec in hook_spec_dict["functions"]:
        name = fn_spec["name"]
        example_input = fn_spec.get("example_input")
        example_output = fn_spec.get("example_output")
        draft = fn_spec.get("draft")

        fn = getattr(module, name, None)
        if fn is None:
            failures.append(f"[{name}] Function not found in {hook_file}.")
            continue

        failures.extend(signature_mismatches(fn, expected_name=name, draft=draft))

        if not run_literal_tests:
            print(
                f"  ℹ  [{name}] {item.domain.value} — examples are documentation only; "
                f"skipping literal smoke test."
            )
            continue

        if example_input is None:
            print(f"  ⚠  [{name}] No example_input — skipping literal test.")
            continue

        try:
            parsed_in = coerce_hook_example_value(example_input)
        except (ValueError, SyntaxError) as e:
            failures.append(
                f"[{name}] Could not literal_eval example_input {example_input!r}: {e}"
            )
            continue

        try:
            result = _call_with_literal_input(fn, parsed_in)
        except Exception as e:
            failures.append(
                f"[{name}] Raised exception on input {parsed_in!r}: {e}"
            )
            continue

        if example_output is not None:
            try:
                expected = coerce_hook_example_value(example_output)
            except (ValueError, SyntaxError) as e:
                failures.append(
                    f"[{name}] Could not literal_eval example_output {example_output!r}: {e}"
                )
                continue
            if result != expected:
                term_example_mismatch_logged = True
                logger.warning(
                    "Hook validate_hook example mismatch for %r: got %r, expected raw "
                    "example_output %r. Reviewer may fix examples in config — verify manually.",
                    name,
                    result,
                    example_output,
                )
            else:
                print(f"  ✓ [{name}] {parsed_in!r} → {result!r}")

    if failures:
        raise HookValidationError(
            f"Hook validation failed for '{hook_file}':\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    if run_literal_tests:
        if term_example_mismatch_logged:
            print(
                f"✓ Hook signatures verified for {hook_file}; "
                f"one or more example_input/example_output pairs mismatched execution (see warnings)."
            )
        else:
            print(
                f"✓ Hook signatures and literal examples validated for {hook_file}."
            )
    else:
        print(
            f"✓ Hook signatures verified for {hook_file} — "
            f"{item.domain.value} examples are not executed (documentation only)."
        )


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
    Write ``hook_spec``, set ``term_extraction='hook_required'``.

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
) -> None:
    table = item.target.table
    term_cfg = _get_nested(config, table, "term_config", item.item_id)

    if resolution.exclude_tokens:
        existing = term_cfg.setdefault("exclude_tokens", [])
        for token in resolution.exclude_tokens:
            if token not in existing:
                existing.append(token)
        print(f"  → exclude_tokens appended: {resolution.exclude_tokens}")

    if resolution.season_map_append:
        existing = term_cfg.setdefault("season_map", [])
        for raw_entry in resolution.season_map_append:
            entry = SeasonMapEntry.model_validate(raw_entry)
            existing.append(entry.model_dump(mode="json"))
        print(f"  → season_map extended: {resolution.season_map_append}")

    if resolution.term_col_override:
        term_cfg["term_col"] = resolution.term_col_override
        term_cfg["year_col"] = None
        term_cfg["season_col"] = None
        print(f"  → term_col overridden: {resolution.term_col_override}")

    if resolution.hook_spec is not None:
        _apply_term_hook_spec_dict(
            term_cfg,
            resolution.hook_spec,
            item_id=item.item_id,
            institution_id=item.institution_id,
        )
        print(f"  → term_config: hook_spec set, term_extraction=hook_required [{item.item_id}]")


def _write_hook_spec_to_config(
    config: dict,
    item: HITLItem,
    hook_spec: HookSpec,
) -> None:
    table = item.target.table
    if item.domain == HITLDomain.IDENTITY_GRAIN:
        grain_cfg = _get_nested(config, table, "grain_contract", item.item_id)
        _apply_grain_hook_spec_dict(grain_cfg, hook_spec, institution_id=item.institution_id)
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        _apply_term_hook_spec_dict(
            term_cfg, hook_spec, item_id=item.item_id, institution_id=item.institution_id
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
        return grain_cfg.get("dedup_policy", {}).get("hook_spec")
    elif item.domain == HITLDomain.IDENTITY_TERM:
        term_cfg = _get_nested(config, table, "term_config", item.item_id)
        return term_cfg.get("hook_spec") if term_cfg else None
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    return cfg
