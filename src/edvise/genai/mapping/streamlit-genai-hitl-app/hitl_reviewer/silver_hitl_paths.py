"""
HITL JSON on the institution **silver** volume, under ``genai_mapping/``, as produced by
the Databricks job entry points:

* ``edvise_genai_ia.py`` — :func:`resolve_run_paths` →
  ``{silver_genai_mapping_root}/runs/{onboard|execute}/{run_id}/identity_agent/`` with
  ``identity_grain_hitl.json`` and ``identity_term_hitl.json`` (see ``IAPaths``).
* ``edvise_genai_sma.py`` — :func:`resolve_run_paths` →
  ``{silver_genai_mapping_root}/runs/{onboard|execute}/{run_id}/schema_mapping_agent/`` with
  ``cohort_hitl_manifest.json`` and ``course_hitl_manifest.json`` (see ``SMAPaths``).

The directory segment ``runs/onboard/{onboard_run_id}/`` (or ``runs/execute/{execute_run_id}/``) is
part of the full UC path, so the same ``onboard_run_id`` keys ``hitl_reviews`` and appears inside
``artifact_path`` for standard onboard HITL rows.

``hitl_reviews.artifact_path`` in Unity Catalog is the full ``/Volumes/...`` path written at
onboard; the Streamlit app reads and writes that same path.
"""

from __future__ import annotations

from typing import Any


def artifact_path_contains_onboard_run_id(artifact_path: str, onboard_run_id: str) -> bool:
    """
    Heuristic: onboard HITL file paths from IA/SMA include the run folder segment, so
    the ``onboard_run_id`` string is usually a substring of ``artifact_path``.

    (Execute-time paths use ``execute_run_id`` in the same slot; those may not match the
    onboard id string.)
    """
    p, r = (artifact_path or "").strip(), (onboard_run_id or "").strip()
    return bool(p and r and r in p)


def set_item_choice(data: dict[str, Any], item_index: int, choice: int | None) -> None:
    """
    Set ``data['items'][item_index]['choice']`` to a 1-based index into ``options``,
    or ``None`` to clear the selection.
    """
    items = data.get("items")
    if not isinstance(items, list) or not (0 <= item_index < len(items)):
        raise KeyError("Invalid item index for HITL JSON")
    row = items[item_index]
    if not isinstance(row, dict):
        raise TypeError("HITL item is not an object")
    if choice is None:
        row["choice"] = None
    else:
        row["choice"] = int(choice)


def set_item_reviewer_note(data: dict[str, Any], item_index: int, note: str | None) -> None:
    """Set ``data['items'][item_index]['reviewer_note']`` (IdentityAgent HITL)."""
    items = data.get("items")
    if not isinstance(items, list) or not (0 <= item_index < len(items)):
        raise KeyError("Invalid item index for HITL JSON")
    row = items[item_index]
    if not isinstance(row, dict):
        raise TypeError("HITL item is not an object")
    if note is None or str(note).strip() == "":
        row["reviewer_note"] = None
    else:
        row["reviewer_note"] = str(note).strip()
