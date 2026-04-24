"""
Batch-write HITL ``choice`` fields to silver JSON from Streamlit session state, plus optional UC approve.

Used by IA grain, IA term, and SMA manifest editors in the HITL workbench (``pages/1_Hitl_Review.py``).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import streamlit as st

from hitl_reviewer.silver_hitl_paths import set_item_choice, set_item_reviewer_note
from hitl_reviewer.unity_volume_files import read_unity_file_text, write_unity_file_text

_WRITE_BLOCKED_MSG = (
    "This UC gate is not **pending** (already approved or rejected); silver JSON writes are disabled."
)


def _is_grain_domain_item(item: dict[str, Any]) -> bool:
    d = str(item.get("domain") or "").lower().strip()
    return d in ("grain", "identity_grain")


def _grain_item_file_indices(items: list[Any]) -> list[int]:
    out: list[int] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if not _is_grain_domain_item(it):
            continue
        opts = it.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        out.append(i)
    return out


def _is_term_domain_item(item: dict[str, Any]) -> bool:
    d = str(item.get("domain") or "").lower().strip()
    return d in ("term", "identity_term")


def _term_item_file_indices(items: list[Any]) -> list[int]:
    out: list[int] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if not _is_term_domain_item(it):
            continue
        opts = it.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        out.append(i)
    return out


def persist_ia_grain_hitl_from_session(
    *, silver_path: str, sk: str, allow_silver_write: bool = True
) -> tuple[bool, str]:
    """
    Merge session ``ia-grain-sel-{sk}-{file_index}`` (and custom notes) for every grain item
    into one JSON write. Fails if any grain row still has no ``choice`` and no session selection.
    """
    if not allow_silver_write:
        return False, _WRITE_BLOCKED_MSG
    try:
        fresh = json.loads(read_unity_file_text(silver_path))
    except Exception as e:  # noqa: BLE001
        return False, f"Re-read failed: {e}"
    items = fresh.get("items")
    if not isinstance(items, list):
        return False, "Invalid HITL JSON: missing ``items`` array."
    fis = _grain_item_file_indices(items)
    if not fis:
        return False, "No grain items with ``options`` in this file."
    for fi in fis:
        row = items[fi]
        if not isinstance(row, dict):
            continue
        opts = row.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        n_opt = len(opts)
        sel_key = f"ia-grain-sel-{sk}-{fi}"
        disk_c = row.get("choice")
        if sel_key in st.session_state:
            try:
                sel_j = max(0, min(int(st.session_state[sel_key]), n_opt - 1))
            except (TypeError, ValueError):
                sel_j = 0
            choice_1 = sel_j + 1
            sel_opt = opts[sel_j] if isinstance(opts[sel_j], dict) else {}
            reentry = str(sel_opt.get("reentry") or "").lower()
            note: str | None = None
            if reentry == "generate_hook":
                ck = f"ia-grain-custom-{sk}-{fi}"
                note = (st.session_state.get(ck) or "").strip()
                if not note:
                    tbl = row.get("table") or fi
                    return (
                        False,
                        f"Table ``{tbl}``: custom / hook option requires a non-empty description "
                        "before saving.",
                    )
            try:
                set_item_choice(fresh, fi, choice_1)
                set_item_reviewer_note(fresh, fi, note)
            except (KeyError, TypeError) as e:
                return False, str(e)
        elif disk_c is not None:
            continue
        else:
            tbl = row.get("table") or fi
            return (
                False,
                f"No ``choice`` for table ``{tbl}`` — use Prev/Next to open each item and pick an "
                "option (or set ``choice`` manually in JSON).",
            )
    try:
        out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
        write_unity_file_text(silver_path, out, overwrite=True)
    except Exception as e:  # noqa: BLE001
        return False, f"Write failed: {e}"
    return True, ""


def persist_ia_term_hitl_from_session(
    *, silver_path: str, sk: str, allow_silver_write: bool = True
) -> tuple[bool, str]:
    """
    Merge session ``ia-term-sel-{sk}-{file_index}`` (and custom notes) for every term item
    into one JSON write. Fails if any term row still has no ``choice`` and no session selection.
    """
    if not allow_silver_write:
        return False, _WRITE_BLOCKED_MSG
    try:
        fresh = json.loads(read_unity_file_text(silver_path))
    except Exception as e:  # noqa: BLE001
        return False, f"Re-read failed: {e}"
    items = fresh.get("items")
    if not isinstance(items, list):
        return False, "Invalid HITL JSON: missing ``items`` array."
    fis = _term_item_file_indices(items)
    if not fis:
        return False, "No term items with ``options`` in this file."
    for fi in fis:
        row = items[fi]
        if not isinstance(row, dict):
            continue
        opts = row.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        n_opt = len(opts)
        sel_key = f"ia-term-sel-{sk}-{fi}"
        disk_c = row.get("choice")
        if sel_key in st.session_state:
            try:
                sel_j = max(0, min(int(st.session_state[sel_key]), n_opt - 1))
            except (TypeError, ValueError):
                sel_j = 0
            choice_1 = sel_j + 1
            sel_opt = opts[sel_j] if isinstance(opts[sel_j], dict) else {}
            reentry = str(sel_opt.get("reentry") or "").lower()
            note: str | None = None
            if reentry == "generate_hook":
                ck = f"ia-term-custom-{sk}-{fi}"
                note = (st.session_state.get(ck) or "").strip()
                if not note:
                    tbl = row.get("table") or fi
                    return (
                        False,
                        f"Table ``{tbl}``: custom / hook option requires a non-empty description "
                        "before saving.",
                    )
            try:
                set_item_choice(fresh, fi, choice_1)
                set_item_reviewer_note(fresh, fi, note)
            except (KeyError, TypeError) as e:
                return False, str(e)
        elif disk_c is not None:
            continue
        else:
            tbl = row.get("table") or fi
            return (
                False,
                f"No ``choice`` for table ``{tbl}`` — use Prev/Next to open each item and pick an "
                "option (or set ``choice`` manually in JSON).",
            )
    try:
        out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
        write_unity_file_text(silver_path, out, overwrite=True)
    except Exception as e:  # noqa: BLE001
        return False, f"Write failed: {e}"
    return True, ""


def persist_hitl_choice_radios_from_session(
    *,
    silver_path: str,
    sk: str,
    option_item_indices: list[int],
    default_choice_index: Callable[[dict[str, Any], int], int],
    allow_silver_write: bool = True,
) -> tuple[bool, str]:
    """
    Write ``choice`` for every HITL row that has ``options``, using session keys
    ``sv{sk}item{i}{item_id}`` (SMA manifest editor and generic multi-radio layout in the HITL workbench).
    """
    if not allow_silver_write:
        return False, _WRITE_BLOCKED_MSG
    try:
        fresh = json.loads(read_unity_file_text(silver_path))
    except Exception as e:  # noqa: BLE001
        return False, f"Re-read failed: {e}"
    items = fresh.get("items")
    if not isinstance(items, list):
        return False, "Invalid HITL JSON: missing ``items`` array."
    for i in option_item_indices:
        if not (0 <= i < len(items)):
            return False, f"Invalid item index {i}."
        item = items[i]
        if not isinstance(item, dict):
            continue
        opts = item.get("options")
        if not isinstance(opts, list) or len(opts) < 1:
            continue
        n = len(opts)
        rk = f"sv{sk}item{i}{item.get('item_id', i)}"
        ix = int(st.session_state.get(rk, default_choice_index(item, n)))
        ix = max(0, min(ix, n - 1))
        try:
            set_item_choice(fresh, i, ix + 1)
        except (KeyError, TypeError) as e:
            return False, str(e)
    try:
        out = json.dumps(fresh, indent=2, ensure_ascii=False) + "\n"
        write_unity_file_text(silver_path, out, overwrite=True)
    except Exception as e:  # noqa: BLE001
        return False, f"Write failed: {e}"
    return True, ""


def try_approve_uc_after_json_write(
    *,
    uc_group_pending: bool,
    approve_uc_if_complete: Callable[[], None] | None,
) -> tuple[bool, str | None]:
    """
    If ``uc_group_pending`` and ``approve_uc_if_complete`` is set, run the callback (UC SQL approve).

    Returns ``(True, None)`` on success or when UC approve was skipped.
    Returns ``(False, message)`` if the callback raised.
    """
    if not uc_group_pending or approve_uc_if_complete is None:
        return True, None
    try:
        approve_uc_if_complete()
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    return True, None
