"""
On-disk layout for GenAI mapping pipeline outputs and UC registration.

**Path (one folder per job run; institution is implicit in the bronze volume root):** ::

    {bronze_volumes_path}/genai_pipeline/{onboard_run_id}/

**Release / git version** is **not** a path segment. It is recorded in
:func:`write_genai_pipeline_run_metadata` as ``genai_pipeline_run.json`` at the run root
(``pipeline_version``), and should be merged into SMA JSON by the pipeline when saving if you
need it on every manifest/transformation file.

**SMA ``pipeline_version``** on manifests/transformation maps is the Edvise/git release
(:class:`~edvise.genai.mapping.schema_mapping_agent.manifest.schemas.MappingManifestEnvelope`,
:class:`~edvise.genai.mapping.schema_mapping_agent.transformation.schemas.TransformationMap`).
The Step 2a/2b LLM prompts do **not** ask the model to output ``pipeline_version`` or
``institution_id``; the pipeline injects them when saving.

Typical subfolders under the run root:

- ``identity_hitl/`` — ``identity_grain_output.json``, ``identity_grain_hitl.json``, …
- ``schema_mapping/`` — ``{institution_id}_mapping_manifest.json``, …
- ``enriched_schema_contracts/`` — ``{institution_id}_schema_contract.json``
- ``genai_pipeline_run.json`` — run metadata (see :func:`write_genai_pipeline_run_metadata`)
- ``run_log.json`` (optional; :mod:`edvise.genai.mapping.shared.hitl.run_log`)

Unity Catalog: :func:`merge_genai_pipeline_artifact_rows` registers file paths (not JSON bodies).
A Streamlit app under ``genai/mapping/streamlit-genai-hitl-app/`` can browse these registry rows.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

LOGGER = logging.getLogger(__name__)

GENAI_ONBOARD_RUN_ID_ENV: Final[str] = "GENAI_ONBOARD_RUN_ID"
"""Override run id for local runs (set when not on a Databricks job cluster)."""

LEGACY_GENAI_PIPELINE_RUN_ID_ENV: Final[str] = "GENAI_PIPELINE_RUN_ID"
"""Deprecated env name; still read after :data:`GENAI_ONBOARD_RUN_ID_ENV` when unset."""

GENAI_PIPELINE_RUN_ID_ENV: Final[str] = GENAI_ONBOARD_RUN_ID_ENV
"""Alias of :data:`GENAI_ONBOARD_RUN_ID_ENV` (default ``env_manual_var`` for :func:`resolve_onboard_run_id`)."""

DATABRICKS_JOB_RUN_ID_ENV: Final[str] = "DATABRICKS_JOB_RUN_ID"
"""Job UI / parameters can inject the numeric job run id (e.g. from a job parameter)."""

GENAI_GIT_TAG_ENV: Final[str] = "GENAI_GIT_TAG"
GIT_TAG_ENV: Final[str] = "GIT_TAG"
GENAI_PIPELINE_VERSION_ENV: Final[str] = "GENAI_PIPELINE_VERSION"

# Spark conf keys that expose the current Databricks job run id (try in order).
_SPARK_CONF_JOB_RUN_ID_KEYS: Final[tuple[str, ...]] = (
    "spark.databricks.job.runId",
    "spark.databricks.jobRunId",
)

# Under ``bronze_volumes_path`` (Unity Catalog volume root for the school).
_GENAI_ROOT_DIR: Final[str] = "genai_pipeline"

GENAI_PIPELINE_RUN_METADATA_BASENAME: Final[str] = "genai_pipeline_run.json"
"""Run-level JSON: ``pipeline_version`` (git/edvise), ``onboard_run_id``, ``institution_id``."""


def new_onboard_run_id() -> str:
    """Return a new opaque run id (UTC timestamp + short random suffix)."""
    from uuid import uuid4

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid4().hex[:8]}"


def _databricks_job_run_id_from_spark() -> str | None:
    """
    Best-effort read of the current Databricks **job run id** from an active Spark session.

    Returns None if not on Databricks, no session, or conf keys are unset.
    """
    try:
        from pyspark.sql import SparkSession  # type: ignore
    except Exception:
        return None
    try:
        spark = SparkSession.getActiveSession()
        if spark is None:
            return None
        for key in _SPARK_CONF_JOB_RUN_ID_KEYS:
            try:
                raw = spark.conf.get(key, None)
            except Exception:
                raw = None
            if raw is None:
                continue
            s = str(raw).strip()
            if s:
                LOGGER.debug("onboard_run_id from Spark conf %s=%s", key, s)
                return s
    except Exception as e:
        LOGGER.debug("Could not read Databricks job run id from Spark (%s)", e)
    return None


def resolve_onboard_run_id(
    explicit: str | None = None,
    *,
    env_manual_var: str = GENAI_ONBOARD_RUN_ID_ENV,
    env_job_run_var: str = DATABRICKS_JOB_RUN_ID_ENV,
    create_if_missing: bool = False,
) -> str | None:
    """
    Resolve ``onboard_run_id`` for bronze artifact paths and ``genai_pipeline_artifacts`` UC rows.

    Precedence (first wins):

    1. **explicit** non-empty string
    2. **Databricks job run id** from active Spark session conf (``spark.databricks.job.runId``, …)
    3. **``os.environ[env_job_run_var]``** — inject in the job (e.g. job parameter bound to the run id)
    4. **``os.environ[env_manual_var]``** (default ``GENAI_ONBOARD_RUN_ID``) — local / manual override
    5. **``os.environ[GENAI_PIPELINE_RUN_ID]``** — legacy manual override when (4) is unset
    6. If ``create_if_missing`` is True: :func:`new_onboard_run_id`

    When nothing matches and ``create_if_missing`` is False, returns None (legacy unversioned layout).
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    spark_rid = _databricks_job_run_id_from_spark()
    if spark_rid:
        return spark_rid
    env_job = os.environ.get(env_job_run_var)
    if env_job and str(env_job).strip():
        return str(env_job).strip()
    env_manual = os.environ.get(env_manual_var)
    if env_manual and str(env_manual).strip():
        return str(env_manual).strip()
    legacy_manual = os.environ.get(LEGACY_GENAI_PIPELINE_RUN_ID_ENV)
    if legacy_manual and str(legacy_manual).strip():
        return str(legacy_manual).strip()
    if create_if_missing:
        rid = new_onboard_run_id()
        LOGGER.info("Assigned new GENAI onboard_run_id=%s", rid)
        return rid
    return None


def _sanitize_run_id_segment(run_id: str) -> str:
    """Make ``onboard_run_id`` safe as a single path segment (Databricks job run ids, etc.)."""
    s = str(run_id).strip()
    if not s:
        raise ValueError("onboard_run_id must be non-empty")
    return re.sub(r"[^\w.\-]+", "_", s)


def resolve_pipeline_version(explicit: str | None = None) -> str:
    """
    Resolve release / **git tag** style version for UC rows and ``genai_pipeline_run.json``.

    Precedence: **explicit** > ``GENAI_GIT_TAG`` > ``GIT_TAG`` > ``GENAI_PIPELINE_VERSION`` >
    installed ``edvise`` distribution version (e.g. ``0.2.0`` from the wheel / ``pyproject.toml``).

    In CI, set ``GENAI_GIT_TAG`` from ``git describe --tags --always`` if you need the exact tag.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    for key in (GENAI_GIT_TAG_ENV, GIT_TAG_ENV, GENAI_PIPELINE_VERSION_ENV):
        v = os.environ.get(key)
        if v and str(v).strip():
            return str(v).strip()
    try:
        return importlib.metadata.version("edvise")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


def write_genai_pipeline_run_metadata(
    run_root: str | Path,
    *,
    institution_id: str,
    onboard_run_id: str,
    pipeline_version: str | None = None,
) -> Path:
    """
    Write ``genai_pipeline_run.json`` at the run root with release metadata.

    This is the canonical place for **git / edvise ``pipeline_version``** on disk. SMA
    """
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    pv = resolve_pipeline_version(pipeline_version)
    payload = {
        "institution_id": str(institution_id).strip(),
        "onboard_run_id": str(onboard_run_id).strip(),
        "pipeline_version": pv,
    }
    path = root / GENAI_PIPELINE_RUN_METADATA_BASENAME
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return path


def parse_uc_catalog_from_volume_path(volume_path: str) -> str | None:
    """Return catalog name from a ``/Volumes/<catalog>/...`` path, or None."""
    p = volume_path.strip()
    if not p.startswith("/Volumes/"):
        return None
    parts = [x for x in p.split("/") if x]
    if len(parts) < 2:
        return None
    # parts[0] == "Volumes", parts[1] == catalog
    return parts[1]


def versioned_genai_run_root(
    bronze_volumes_path: str | Path,
    onboard_run_id: str,
) -> Path:
    """
    Root directory for all GenAI artifacts for one pipeline run.

    Layout (institution is implied by ``bronze_volumes_path``)::

        genai_pipeline/{onboard_run_id}/

    Raises
    ------
    ValueError
        If ``onboard_run_id`` is empty.
    """
    seg = _sanitize_run_id_segment(onboard_run_id)
    root = Path(str(bronze_volumes_path).rstrip("/"))
    return root / _GENAI_ROOT_DIR / seg


@dataclass(frozen=True)
class GenaiPipelineLayout:
    """Standard subpaths under :func:`versioned_genai_run_root`."""

    run_root: Path
    identity_hitl: Path
    schema_mapping: Path
    enriched_schema_contracts: Path

    @classmethod
    def from_bronze(
        cls,
        bronze_volumes_path: str | Path,
        onboard_run_id: str,
    ) -> GenaiPipelineLayout:
        rr = versioned_genai_run_root(bronze_volumes_path, onboard_run_id)
        return cls(
            run_root=rr,
            identity_hitl=rr / "identity_hitl",
            schema_mapping=rr / "schema_mapping",
            enriched_schema_contracts=rr / "enriched_schema_contracts",
        )


# basename / pattern -> artifact_kind stored in UC
_ARTIFACT_KIND_BY_BASENAME: dict[str, str] = {
    "identity_grain_output.json": "identity_grain_output",
    "identity_grain_hitl.json": "identity_grain_hitl",
    "identity_term_output.json": "identity_term_output",
    "identity_term_hitl.json": "identity_term_hitl",
    "sma_hitl.json": "sma_hitl",
    "sma_manifest_output.json": "sma_manifest_output",
    "run_log.json": "run_log",
    "genai_pipeline_run.json": "pipeline_run_metadata",
}

_SMA_HITL_RE = re.compile(r"^sma_hitl.*\.json$", re.IGNORECASE)
_SMA_MANIFEST_RE = re.compile(r"^sma_manifest.*\.json$", re.IGNORECASE)


def discover_artifact_files(
    run_root: str | Path,
    institution_id: str,
) -> dict[str, Path]:
    """
    Map ``artifact_kind`` -> absolute path for known JSON outputs under ``run_root``.

    Walks ``run_root`` recursively. Kinds are stable strings suitable for the UC registry.
    """
    root = Path(run_root)
    inst = str(institution_id).strip()
    if not root.is_dir():
        return {}

    out: dict[str, Path] = {}
    for path in root.rglob("*.json"):
        if not path.is_file():
            continue
        name = path.name
        if name in _ARTIFACT_KIND_BY_BASENAME:
            kind = _ARTIFACT_KIND_BY_BASENAME[name]
            out[kind] = path
            continue
        if name == f"{inst}_mapping_manifest.json":
            out["mapping_manifest"] = path
            continue
        if name == f"{inst}_transformation_map.json":
            out["transformation_map"] = path
            continue
        if name.endswith("_schema_contract.json") and name.startswith(f"{inst}_"):
            out["enriched_schema_contract"] = path
            continue
        if _SMA_HITL_RE.match(name):
            out.setdefault("sma_hitl", path)
            continue
        if _SMA_MANIFEST_RE.match(name):
            out.setdefault("sma_manifest_output", path)

    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_genai_pipeline_artifact_rows(
    *,
    institution_id: str,
    onboard_run_id: str,
    pipeline_version: str,
    bronze_volumes_path: str,
    artifact_paths: dict[str, Path],
    uc_catalog: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build rows for :func:`merge_genai_pipeline_artifact_rows`.

    Parameters
    ----------
    artifact_paths
        Map ``artifact_kind`` -> absolute path (see :func:`discover_artifact_files`).
    pipeline_version
        Git tag / release (e.g. ``0.2.0``); see :func:`resolve_pipeline_version`.
    uc_catalog
        Unity Catalog name for the volume (e.g. from :func:`parse_uc_catalog_from_volume_path`).
        If None, parsed from ``bronze_volumes_path`` when it is a ``/Volumes/`` path.
    """
    inst = str(institution_id).strip()
    rid = str(onboard_run_id).strip()
    pver = str(pipeline_version).strip() or resolve_pipeline_version()
    bronze = str(bronze_volumes_path).rstrip("/")
    cat = uc_catalog or parse_uc_catalog_from_volume_path(bronze) or ""
    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for kind in sorted(artifact_paths.keys()):
        p = artifact_paths[kind]
        rel = str(p)
        if rel.startswith(bronze):
            rel_path = rel[len(bronze) :].lstrip("/")
        else:
            rel_path = p.name
        try:
            checksum = _sha256_file(p)
        except OSError as e:
            LOGGER.warning("Could not hash artifact %s: %s", p, e)
            checksum = ""
        rows.append(
            {
                "institution_id": inst,
                "onboard_run_id": rid,
                "pipeline_version": pver,
                "artifact_kind": kind,
                "uc_catalog": cat,
                "bronze_volume_base": bronze,
                "relative_path": rel_path,
                "absolute_path": str(p.resolve()),
                "content_sha256": checksum,
                "registered_at": now,
            }
        )
    return rows


def _maybe_rename_pipeline_run_id_in_artifacts_table(spark: Any, table_name: str) -> None:
    """Older ``genai_pipeline_artifacts`` tables used ``pipeline_run_id``; normalize to ``onboard_run_id``."""
    try:
        spark.sql(f"ALTER TABLE {table_name} RENAME COLUMN pipeline_run_id TO onboard_run_id")
    except Exception:  # noqa: BLE001
        LOGGER.debug("Skipped artifact table pipeline_run_id rename for %s", table_name)


def merge_genai_pipeline_artifact_rows(
    spark: Any,
    *,
    table_name: str,
    rows: list[dict[str, Any]],
) -> bool:
    """
    Merge rows into a Delta table in Unity Catalog (creates table if missing).

    Parameters
    ----------
    spark
        Active Spark session (Databricks / connect).
    table_name
        Fully qualified name, e.g. ``catalog.genai.genai_pipeline_artifacts``.

    Returns
    -------
    bool
        True if write/merge succeeded, False on best-effort failure (logged).
    """
    if not rows:
        LOGGER.info("merge_genai_pipeline_artifact_rows: no rows; skipping")
        return True

    try:
        from delta.tables import DeltaTable  # type: ignore
    except Exception as e:
        LOGGER.warning("merge_genai_pipeline_artifact_rows: PySpark/Delta unavailable (%s)", e)
        return False

    try:
        df = spark.createDataFrame(rows)
        try:
            spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "true")
        except Exception:
            pass

        try:
            dt = DeltaTable.forName(spark, table_name)
            _maybe_rename_pipeline_run_id_in_artifacts_table(spark, table_name)
        except Exception:
            (
                df.write.format("delta")
                .mode("append")
                .option("mergeSchema", "true")
                .saveAsTable(table_name)
            )
            LOGGER.info(
                "Created genai pipeline artifacts table %s (%s rows)",
                table_name,
                len(rows),
            )
            return True

        (
            dt.alias("t")
            .merge(
                df.alias("s"),
                "t.institution_id = s.institution_id AND "
                "t.onboard_run_id = s.onboard_run_id AND "
                "t.artifact_kind = s.artifact_kind",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
        LOGGER.info(
            "Merged %s rows into %s",
            len(rows),
            table_name,
        )
        return True
    except Exception as e:
        LOGGER.warning("merge_genai_pipeline_artifact_rows failed: %s", e)
        return False


def register_discovered_artifacts_to_uc(
    spark: Any,
    *,
    table_name: str,
    institution_id: str,
    onboard_run_id: str,
    bronze_volumes_path: str,
    pipeline_version: str | None = None,
    run_root: str | Path | None = None,
    uc_catalog: str | None = None,
) -> bool:
    """
    Discover JSON artifacts under the versioned run root and merge registry rows into UC.

    If ``run_root`` is None, uses :func:`versioned_genai_run_root`.
    """
    pv = resolve_pipeline_version(pipeline_version)
    rr = (
        Path(run_root)
        if run_root is not None
        else versioned_genai_run_root(bronze_volumes_path, onboard_run_id)
    )
    paths = discover_artifact_files(rr, institution_id)
    rows = build_genai_pipeline_artifact_rows(
        institution_id=institution_id,
        onboard_run_id=onboard_run_id,
        pipeline_version=pv,
        bronze_volumes_path=bronze_volumes_path,
        artifact_paths=paths,
        uc_catalog=uc_catalog,
    )
    return merge_genai_pipeline_artifact_rows(spark, table_name=table_name, rows=rows)


__all__ = [
    "DATABRICKS_JOB_RUN_ID_ENV",
    "GENAI_GIT_TAG_ENV",
    "GENAI_ONBOARD_RUN_ID_ENV",
    "GENAI_PIPELINE_RUN_METADATA_BASENAME",
    "GENAI_PIPELINE_RUN_ID_ENV",
    "GENAI_PIPELINE_VERSION_ENV",
    "GIT_TAG_ENV",
    "LEGACY_GENAI_PIPELINE_RUN_ID_ENV",
    "GenaiPipelineLayout",
    "build_genai_pipeline_artifact_rows",
    "discover_artifact_files",
    "merge_genai_pipeline_artifact_rows",
    "new_onboard_run_id",
    "parse_uc_catalog_from_volume_path",
    "register_discovered_artifacts_to_uc",
    "resolve_onboard_run_id",
    "resolve_pipeline_version",
    "versioned_genai_run_root",
    "write_genai_pipeline_run_metadata",
]
