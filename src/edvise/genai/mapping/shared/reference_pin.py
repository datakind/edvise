"""
Pin GenAI mapping few-shot artifacts into the shared ``genai_mapping.references`` volume.

School ``active/`` (and onboard runs) remain mutable execute/ops state. Gold references are an
explicit copy into::

    /Volumes/<catalog>/genai_mapping/references/<reference_id>/current/

plus ``genai_reference_pin.json`` (local hash metadata) and a row in
``{catalog}.genai_mapping.reference_pins``.

Two job modes (selected by ``catalog``, not a CLI flag):

* ``publish`` — ``catalog=dev_sst_02``: copy from local ``active/``.
* ``pull`` — ``catalog=staging_sst_01``: copy ``references/<id>/current/`` from
  ``dev_sst_02``; does not read local ``active/`` (parity).
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from edvise.genai.mapping.shared.active_promotion import read_genai_active_registry
from edvise.genai.mapping.shared.volume_paths import (
    genai_reference_current_root,
    silver_genai_mapping_root,
)

LOGGER = logging.getLogger(__name__)

GENAI_REFERENCE_PIN_BASENAME: str = "genai_reference_pin.json"
"""Sidecar under ``references/<id>/current/`` recording pin provenance + content hash."""

_REFERENCE_PIN_SCHEMA_VERSION: int = 1

# Required for SMA few-shot; hashed whenever present under the pin directory.
REFERENCE_PIN_REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "manifest_map.json",
    "transformation_map.json",
)

# Optional but useful for selection / debugging; included in the hash when copied.
REFERENCE_PIN_OPTIONAL_ARTIFACTS: tuple[str, ...] = ("enriched_schema_contract.json",)

SourceKind = Literal["active", "onboard_run"]
PinMode = Literal["publish", "pull"]

# Canonical catalog for gold references (publish writes here; pull reads from here).
DEFAULT_CANONICAL_REFERENCE_CATALOG: str = "dev_sst_02"
# Replica catalog that pulls from the canonical catalog.
DEFAULT_REPLICA_REFERENCE_CATALOG: str = "staging_sst_01"


def resolve_pin_mode(catalog: str) -> PinMode:
    """
    Map UC catalog to pin action.

    ``dev_sst_02`` → publish; ``staging_sst_01`` → pull. Any other catalog is rejected.
    """
    c = str(catalog).strip()
    if c == DEFAULT_CANONICAL_REFERENCE_CATALOG:
        return "publish"
    if c == DEFAULT_REPLICA_REFERENCE_CATALOG:
        return "pull"
    raise ValueError(
        f"catalog must be {DEFAULT_CANONICAL_REFERENCE_CATALOG!r} (publish) or "
        f"{DEFAULT_REPLICA_REFERENCE_CATALOG!r} (pull); got {c!r}"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def compute_reference_content_hash(
    artifact_paths: dict[str, Path] | Iterable[tuple[str, Path]],
) -> str:
    """
    Stable content hash over named artifact files.

    Files are mixed in sorted basename order as ``name\\0bytes\\0``. Returns ``sha256:<hex>``.
    """
    if isinstance(artifact_paths, dict):
        items = list(artifact_paths.items())
    else:
        items = list(artifact_paths)
    if not items:
        raise ValueError("artifact_paths must be non-empty")
    h = hashlib.sha256()
    for name, path in sorted(items, key=lambda x: x[0]):
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Cannot hash missing artifact: {p}")
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return f"sha256:{h.hexdigest()}"


def read_genai_reference_pin(current_root: str | Path) -> dict[str, Any] | None:
    """Load ``genai_reference_pin.json`` from ``current_root`` if present."""
    p = Path(current_root) / GENAI_REFERENCE_PIN_BASENAME
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {p}")
    return data


def resolve_pinned_artifact_files(current_root: str | Path) -> dict[str, Path]:
    """
    Map basename -> path for artifacts present under ``current/`` that participate in the hash.

    Requires :data:`REFERENCE_PIN_REQUIRED_ARTIFACTS`. Optional artifacts are included when present.
    """
    root = Path(current_root)
    out: dict[str, Path] = {}
    for name in REFERENCE_PIN_REQUIRED_ARTIFACTS:
        p = root / name
        if not p.is_file():
            raise FileNotFoundError(
                f"Pinned reference missing required artifact {name!r} under {root}"
            )
        out[name] = p
    for name in REFERENCE_PIN_OPTIONAL_ARTIFACTS:
        p = root / name
        if p.is_file():
            out[name] = p
    return out


def verify_reference_pin_hash(current_root: str | Path) -> str:
    """
    Recompute the content hash for ``current_root`` and compare to ``genai_reference_pin.json``.

    Returns the verified ``content_hash``. Raises ``FileNotFoundError`` / ``ValueError`` on drift.
    """
    root = Path(current_root)
    pin = read_genai_reference_pin(root)
    if pin is None:
        raise FileNotFoundError(
            f"Missing {GENAI_REFERENCE_PIN_BASENAME} under {root}; reference is not pinned"
        )
    expected = str(pin.get("content_hash") or "").strip()
    if not expected:
        raise ValueError(
            f"{GENAI_REFERENCE_PIN_BASENAME} missing content_hash at {root}"
        )
    artifacts = resolve_pinned_artifact_files(root)
    # Prefer the pin's artifact list when present so optional files stay consistent with pin time.
    listed = pin.get("artifacts")
    if isinstance(listed, list) and listed:
        names = [str(x) for x in listed]
        missing = [n for n in names if n not in artifacts]
        if missing:
            raise FileNotFoundError(
                f"Pin lists artifacts missing on disk under {root}: {missing}"
            )
        # Extra optional files on disk that were not part of the pin must not affect the hash.
        artifacts = {n: artifacts[n] for n in names if n in artifacts}
        for req in REFERENCE_PIN_REQUIRED_ARTIFACTS:
            if req not in artifacts:
                raise FileNotFoundError(
                    f"Pin artifacts omit required {req!r} under {root}"
                )
    actual = compute_reference_content_hash(artifacts)
    if actual != expected:
        raise ValueError(
            f"Reference pin hash mismatch under {root}: pin has {expected!r}, "
            f"disk computes {actual!r}"
        )
    return actual


@dataclass(frozen=True)
class SmaFewShotPin:
    """Gold few-shot files SMA onboard reads from ``references/<id>/current/``."""

    reference_id: str
    current_root: Path
    manifest_map: Path
    transformation_map: Path
    content_hash: str


def resolve_sma_few_shot_pin(reference_id: str, *, catalog: str) -> SmaFewShotPin:
    """
    Resolve and verify the pinned few-shot used by SMA onboard (not school ``active/``).

    Requires ``manifest_map.json``, ``transformation_map.json``, and
    ``genai_reference_pin.json`` under ``references/<reference_id>/current/``.
    Raises ``FileNotFoundError`` / ``ValueError`` if the pin is missing or drifted.
    """
    ref = str(reference_id).strip()
    if not ref:
        raise ValueError("reference_id must be non-empty")
    current = Path(genai_reference_current_root(ref, catalog=catalog))
    if not current.is_dir():
        raise FileNotFoundError(
            f"Pinned few-shot not found at {current}. "
            f"Run edvise_genai_pin_reference for {ref!r} before SMA onboard "
            f"(catalog={catalog!r})."
        )
    content_hash = verify_reference_pin_hash(current)
    artifacts = resolve_pinned_artifact_files(current)
    return SmaFewShotPin(
        reference_id=ref,
        current_root=current,
        manifest_map=artifacts["manifest_map.json"],
        transformation_map=artifacts["transformation_map.json"],
        content_hash=content_hash,
    )


def resolve_reference_source_paths(
    *,
    catalog: str,
    source_institution_id: str,
    source: SourceKind = "active",
    onboard_run_id: str | None = None,
) -> dict[str, Path]:
    """
    Resolve source artifact paths on the institution silver volume.

    ``active`` — flat ``genai_mapping/active/``.
    ``onboard_run`` — IA + SMA run trees under ``runs/onboard/<onboard_run_id>/``.
    """
    inst = str(source_institution_id).strip()
    if not inst:
        raise ValueError("source_institution_id must be non-empty")
    genai = Path(silver_genai_mapping_root(inst, catalog=catalog))

    if source == "active":
        active = genai / "active"
        paths = {
            "manifest_map.json": active / "manifest_map.json",
            "transformation_map.json": active / "transformation_map.json",
            "enriched_schema_contract.json": active / "enriched_schema_contract.json",
        }
    elif source == "onboard_run":
        rid = str(onboard_run_id or "").strip()
        if not rid:
            raise ValueError("onboard_run_id is required when source='onboard_run'")
        ia = genai / "runs" / "onboard" / rid / "identity_agent"
        sma = genai / "runs" / "onboard" / rid / "schema_mapping_agent"
        paths = {
            "manifest_map.json": sma / "manifest_map.json",
            "transformation_map.json": sma / "transformation_map.json",
            "enriched_schema_contract.json": ia / "enriched_schema_contract.json",
        }
    else:
        raise ValueError(f"source must be 'active' or 'onboard_run', got {source!r}")

    for name in REFERENCE_PIN_REQUIRED_ARTIFACTS:
        if not paths[name].is_file():
            raise FileNotFoundError(
                f"Cannot pin missing required source artifact {name!r}: {paths[name]}"
            )
    return {k: v for k, v in paths.items() if v.is_file()}


def _history_dirname(*, pinned_at: str, content_hash: str) -> str:
    """Folder name under ``references/<id>/`` for an immutable history snapshot."""
    day = pinned_at[:10].replace("-", "") if pinned_at else "unknown"
    digest = content_hash.removeprefix("sha256:")[:12]
    return f"v{day}_{digest}"


def pin_reference_snapshot(
    *,
    catalog: str,
    reference_id: str,
    pipeline_version: str | None = None,
    write_history_copy: bool = True,
) -> dict[str, Any]:
    """
    Copy ``active/`` artifacts for ``reference_id`` into
    ``references/<reference_id>/current/`` and write pin metadata.

    ``reference_id`` is both the library slot and the institution whose silver
    ``genai_mapping/active/`` is the source. ``source_onboard_run_id`` and
    ``pipeline_version`` (when not passed) are read from ``genai_active_registry.json``.

    Returns the pin payload (also written as ``genai_reference_pin.json``). Does **not**
    write the UC ``reference_pins`` table — callers use :func:`upsert_reference_pin_row`.
    """
    ref = str(reference_id).strip()
    if not ref:
        raise ValueError("reference_id must be non-empty")

    source_paths = resolve_reference_source_paths(
        catalog=catalog,
        source_institution_id=ref,
        source="active",
    )

    resolved_run_id: str | None = None
    resolved_pipeline = str(pipeline_version or "").strip() or None
    active_root = Path(silver_genai_mapping_root(ref, catalog=catalog)) / "active"
    reg = read_genai_active_registry(active_root)
    if reg:
        rid = str(reg.get("onboard_run_id") or "").strip()
        resolved_run_id = rid or None
        if not resolved_pipeline:
            pver = str(reg.get("pipeline_version") or "").strip()
            resolved_pipeline = pver or None

    content_hash = compute_reference_content_hash(source_paths)
    pinned_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    artifact_names = sorted(source_paths.keys())

    current = Path(genai_reference_current_root(ref, catalog=catalog))
    current.mkdir(parents=True, exist_ok=True)

    for name, src in source_paths.items():
        dst = current / name
        shutil.copy2(src, dst)
        LOGGER.info("Pinned %s -> %s", src, dst)

    payload: dict[str, Any] = {
        "schema_version": _REFERENCE_PIN_SCHEMA_VERSION,
        "reference_id": ref,
        "source_institution_id": ref,
        "pinned_at": pinned_at,
        "content_hash": content_hash,
        "artifacts": artifact_names,
        "source": "active",
        "uc_catalog": str(catalog).strip(),
    }
    if resolved_run_id:
        payload["source_onboard_run_id"] = resolved_run_id
    if resolved_pipeline:
        payload["pipeline_version"] = resolved_pipeline

    pin_path = current / GENAI_REFERENCE_PIN_BASENAME
    _atomic_write_json(pin_path, payload)
    LOGGER.info("Wrote %s (content_hash=%r)", pin_path, content_hash)

    verify_reference_pin_hash(current)

    if write_history_copy:
        hist = current.parent / _history_dirname(
            pinned_at=pinned_at, content_hash=content_hash
        )
        if hist.exists():
            shutil.rmtree(hist)
        shutil.copytree(current, hist)
        LOGGER.info("Wrote history snapshot %s", hist)

    return payload


def pull_reference_snapshot(
    *,
    catalog: str,
    reference_id: str,
    source_catalog: str,
    write_history_copy: bool = True,
) -> dict[str, Any]:
    """
    Copy a pinned reference from ``source_catalog`` into ``catalog`` (replica).

    Does **not** read local ``active/`` — that is intentional so divergent school
    active state cannot overwrite the gold few-shot library.

    Requires readable
    ``/Volumes/<source_catalog>/genai_mapping/references/<reference_id>/current/``.
    Verifies the source pin hash, copies bytes, re-verifies on the destination, and
    returns the destination pin payload (with ``pulled_from_catalog`` set).
    """
    dest_cat = str(catalog).strip()
    src_cat = str(source_catalog).strip()
    ref = str(reference_id).strip()
    if not dest_cat or not src_cat or not ref:
        raise ValueError("catalog, source_catalog, and reference_id must be non-empty")
    if dest_cat == src_cat:
        raise ValueError(
            f"pull mode requires source_catalog != catalog (both are {dest_cat!r})"
        )

    source_current = Path(genai_reference_current_root(ref, catalog=src_cat))
    if not source_current.is_dir():
        raise FileNotFoundError(
            f"Cannot pull reference {ref!r}: source current/ missing at {source_current}. "
            f"Publish on {src_cat} first, and ensure this job can read that catalog's volumes."
        )

    source_hash = verify_reference_pin_hash(source_current)
    source_pin = read_genai_reference_pin(source_current)
    if source_pin is None:
        raise FileNotFoundError(
            f"Missing {GENAI_REFERENCE_PIN_BASENAME} under {source_current}"
        )

    dest_current = Path(genai_reference_current_root(ref, catalog=dest_cat))
    dest_current.mkdir(parents=True, exist_ok=True)

    # Replace destination current/ contents with source artifacts + pin sidecar.
    for existing in dest_current.iterdir():
        if existing.is_file():
            existing.unlink()
        elif existing.is_dir():
            shutil.rmtree(existing)

    for src in source_current.iterdir():
        if not src.is_file():
            continue
        shutil.copy2(src, dest_current / src.name)
        LOGGER.info("Pulled %s -> %s", src, dest_current / src.name)

    pulled_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = dict(source_pin)
    payload["uc_catalog"] = dest_cat
    payload["pulled_from_catalog"] = src_cat
    payload["pulled_at"] = pulled_at
    # Keep original content_hash / source_onboard_run_id / artifacts from canonical pin.
    if str(payload.get("content_hash") or "").strip() != source_hash:
        payload["content_hash"] = source_hash

    pin_path = dest_current / GENAI_REFERENCE_PIN_BASENAME
    _atomic_write_json(pin_path, payload)

    dest_hash = verify_reference_pin_hash(dest_current)
    if dest_hash != source_hash:
        raise ValueError(
            f"Pull hash mismatch for {ref!r}: source {source_hash!r} vs dest {dest_hash!r}"
        )

    if write_history_copy:
        hist = dest_current.parent / _history_dirname(
            pinned_at=str(payload.get("pinned_at") or pulled_at),
            content_hash=dest_hash,
        )
        if hist.exists():
            shutil.rmtree(hist)
        shutil.copytree(dest_current, hist)
        LOGGER.info("Wrote history snapshot %s", hist)

    LOGGER.info(
        "Pulled reference_id=%r %s -> %s content_hash=%r",
        ref,
        src_cat,
        dest_cat,
        dest_hash,
    )
    return payload


def upsert_reference_pin_row(
    catalog: str,
    pin: dict[str, Any],
    *,
    spark: Any | None = None,
) -> None:
    """
    Mark prior ``active`` rows for this ``reference_id`` as ``superseded`` and insert the new pin.

    Ensures state tables (including ``reference_pins``) exist via :func:`create_state_tables`.
    """
    from edvise.genai.mapping.state._sql import (
        REFERENCE_PINS,
        get_spark_session,
        lit,
        qualified_table,
    )
    from edvise.genai.mapping.state.table_setup import create_state_tables

    c = str(catalog).strip()
    ref = str(pin.get("reference_id") or "").strip()
    content_hash = str(pin.get("content_hash") or "").strip()
    if not c or not ref or not content_hash:
        raise ValueError("catalog, pin.reference_id, and pin.content_hash are required")

    spark = spark if spark is not None else get_spark_session()
    if spark is None:
        raise RuntimeError(
            "No active Spark session; cannot write reference_pins (pass spark= or run on Databricks)"
        )

    create_state_tables(c, spark=spark)
    t = qualified_table(c, REFERENCE_PINS)

    spark.sql(
        f"""
        UPDATE {t}
        SET status = 'superseded'
        WHERE reference_id = {lit(ref)}
          AND status = 'active'
        """
    )

    artifacts = pin.get("artifacts") or []
    if isinstance(artifacts, (list, tuple)):
        artifacts_json = json.dumps(list(artifacts))
    else:
        artifacts_json = json.dumps([])

    pin_path = genai_reference_current_root(ref, catalog=c)
    archetype = str(pin.get("archetype") or "").strip()
    pipeline_version = str(pin.get("pipeline_version") or "").strip()
    pinned_by = str(pin.get("pinned_by") or "").strip()
    source_onboard_run_id = str(pin.get("source_onboard_run_id") or "").strip()
    source_institution_id = str(pin.get("source_institution_id") or "").strip()
    pinned_at = str(pin.get("pinned_at") or "").strip()

    spark.sql(
        f"""
        INSERT INTO {t} (
          reference_id,
          archetype,
          pipeline_version,
          content_hash,
          pinned_at,
          pinned_by,
          source_onboard_run_id,
          source_institution_id,
          status,
          uc_catalog,
          artifacts,
          pin_path
        ) VALUES (
          {lit(ref)},
          {lit(archetype) if archetype else "NULL"},
          {lit(pipeline_version) if pipeline_version else "NULL"},
          {lit(content_hash)},
          {f"CAST({lit(pinned_at)} AS TIMESTAMP)" if pinned_at else "current_timestamp()"},
          {lit(pinned_by) if pinned_by else "NULL"},
          {lit(source_onboard_run_id) if source_onboard_run_id else "NULL"},
          {lit(source_institution_id) if source_institution_id else "NULL"},
          {lit("active")},
          {lit(c)},
          {lit(artifacts_json)},
          {lit(pin_path)}
        )
        """
    )
    LOGGER.info(
        "Upserted reference_pins row reference_id=%r content_hash=%r", ref, content_hash
    )


def active_reference_pin_row(
    catalog: str,
    reference_id: str,
    *,
    spark: Any | None = None,
) -> dict[str, Any] | None:
    """Return the ``status='active'`` registry row for ``reference_id``, or None."""
    from edvise.genai.mapping.state._sql import (
        REFERENCE_PINS,
        get_spark_session,
        lit,
        qualified_table,
    )

    c = str(catalog).strip()
    ref = str(reference_id).strip()
    if not c or not ref:
        raise ValueError("catalog and reference_id must be non-empty")

    spark = spark if spark is not None else get_spark_session()
    if spark is None:
        return None

    t = qualified_table(c, REFERENCE_PINS)
    rows = spark.sql(
        f"""
        SELECT
          reference_id,
          archetype,
          pipeline_version,
          content_hash,
          pinned_at,
          pinned_by,
          source_onboard_run_id,
          source_institution_id,
          status,
          uc_catalog,
          artifacts,
          pin_path
        FROM {t}
        WHERE reference_id = {lit(ref)}
          AND status = 'active'
        ORDER BY pinned_at DESC
        LIMIT 1
        """
    ).collect()
    if not rows:
        return None
    row = rows[0]
    if hasattr(row, "asDict"):
        return dict(row.asDict())  # type: ignore[attr-defined]
    return {
        "reference_id": row["reference_id"],
        "archetype": row["archetype"],
        "pipeline_version": row["pipeline_version"],
        "content_hash": row["content_hash"],
        "pinned_at": row["pinned_at"],
        "pinned_by": row["pinned_by"],
        "source_onboard_run_id": row["source_onboard_run_id"],
        "source_institution_id": row["source_institution_id"],
        "status": row["status"],
        "uc_catalog": row["uc_catalog"],
        "artifacts": row["artifacts"],
        "pin_path": row["pin_path"],
    }
