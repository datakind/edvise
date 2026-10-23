"""Register an NSC cohort/course pair as a webapp batch.

The institution slice is uploaded through the existing validate-sftp API, then
the validated GCS objects are copied to ``gcs_uploads/{batch_id}/``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlparse

from edvise.ingestion.nsc_sftp.file_selection import (
    classify_pdp_file_role,
    extract_file_stamp,
)
from edvise.ingestion.nsc_sftp.sftp import output_file_name_from_sftp
from edvise.utils.gcs import SUCCESS_FILENAME


def batch_name_for_stamp(stamp: str) -> str:
    """Batch name unique per institution. The stamp is unique per NSC delivery."""
    return f"nsc_{stamp}"


def external_bucket_name(db_workspace: str, inst_id: str) -> str:
    """Match the API's ``{env}_{institution uuid hex}`` bucket name."""
    workspace = db_workspace.lower().strip()
    if workspace.startswith("dev"):
        env = "dev"
    elif workspace.startswith("staging"):
        env = "staging"
    else:
        raise ValueError(
            f"Cannot derive a GCS bucket env from DB_workspace={db_workspace!r}."
        )
    inst = inst_id.replace("-", "").strip().lower()
    if not inst:
        raise ValueError("inst_id is required to derive the GCS bucket name.")
    return f"{env}_{inst}"


def bucket_from_signed_url(url: str) -> str:
    """Return the bucket name embedded in a GCS signed URL."""
    parsed = urlparse(url)
    host = parsed.netloc.split(":", 1)[0]
    path = parsed.path.lstrip("/")
    if host.endswith(".storage.googleapis.com"):
        return host[: -len(".storage.googleapis.com")]
    if host in {"storage.googleapis.com", "storage.cloud.google.com"} and path:
        return path.split("/", 1)[0]
    raise ValueError(f"Could not read a GCS bucket from signed URL host {host!r}.")


def split_csv_path(
    tmp_dir: str, file_fingerprint: str, pdp_id: str, source_file_name: str
) -> str:
    """Per-institution CSV path under the NSC tmp volume."""
    out_name = output_file_name_from_sftp(source_file_name)
    return os.path.join(tmp_dir, "splits", file_fingerprint, str(pdp_id), out_name)


@dataclass(frozen=True)
class RegistrationPlan:
    """What still has to happen before the bronze batch folder can be copied."""

    upload_names: tuple[str, ...]
    create_batch: bool
    batch_id: str
    error: str = ""


def _usable_file(file_row: Mapping[str, Any] | None) -> bool:
    if not file_row:
        return False
    return bool(file_row.get("valid")) and not bool(file_row.get("sst_generated"))


def plan_registration(
    overview: Mapping[str, Any],
    *,
    cohort_file_name: str,
    course_file_name: str,
    batch_name: str,
) -> RegistrationPlan:
    """Decide upload, validate, and create-batch steps from ``GET /input``."""
    files = {
        str(row.get("name")): row
        for row in overview.get("files") or []
        if row.get("name")
    }
    expected = {cohort_file_name, course_file_name}
    for name in expected:
        row = files.get(name)
        if row is not None and not _usable_file(row):
            return RegistrationPlan(
                upload_names=(),
                create_batch=False,
                batch_id="",
                error=(
                    f"{name} already exists for this institution but is not a "
                    "valid input file, so it cannot be reused or overwritten."
                ),
            )

    batches = {
        str(row.get("name")): row
        for row in overview.get("batches") or []
        if row.get("name")
    }
    existing = batches.get(batch_name)
    if existing is not None:
        names = set((existing.get("file_names_to_ids") or {}).keys())
        if names != expected:
            return RegistrationPlan(
                upload_names=(),
                create_batch=False,
                batch_id="",
                error=(
                    f"Batch {batch_name} already exists with files {sorted(names)}, "
                    f"expected {sorted(expected)}."
                ),
            )
        return RegistrationPlan(
            upload_names=(),
            create_batch=False,
            batch_id=str(existing.get("batch_id") or ""),
        )

    upload = tuple(
        name
        for name in (cohort_file_name, course_file_name)
        if not _usable_file(files.get(name))
    )
    return RegistrationPlan(upload_names=upload, create_batch=True, batch_id="")


@dataclass(frozen=True)
class InstitutionPair:
    inst_id: str
    pdp_id: str
    institution_name: str
    stamp: str
    cohort_file_name: str
    course_file_name: str
    cohort_fingerprint: str
    course_fingerprint: str
    cohort_split_path: str
    course_split_path: str


def pair_plan_rows(
    rows: Sequence[Mapping[str, Any]], *, tmp_dir: str
) -> tuple[list[InstitutionPair], list[str]]:
    """Pair cohort and course plan rows that share an institution and stamp.

    Returns complete pairs and human-readable problems for rows that cannot
    be paired. A row whose split CSV is missing is ignored here; the caller
    uses ``split_exists`` when deciding whether a manifest file can close.
    """
    grouped: dict[tuple[str, str], dict[str, Mapping[str, Any]]] = {}
    problems: list[str] = []
    for row in rows:
        source_name = str(row.get("file_name") or "")
        role = classify_pdp_file_role(source_name)
        if role is None:
            problems.append(f"Unclassified file {source_name}")
            continue
        try:
            stamp = extract_file_stamp(source_name)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        inst_id = str(row.get("inst_id") or "").strip()
        if not inst_id:
            problems.append(f"Missing inst_id for {source_name}")
            continue
        grouped.setdefault((inst_id, stamp), {})[role] = row

    pairs: list[InstitutionPair] = []
    for (inst_id, stamp), roles in sorted(grouped.items()):
        cohort = roles.get("cohort")
        course = roles.get("course")
        if cohort is None or course is None:
            have = ",".join(sorted(roles))
            problems.append(
                f"inst_id={inst_id} stamp={stamp} has only {have}; need cohort and course"
            )
            continue
        pdp_id = str(cohort.get("institution_id") or course.get("institution_id") or "")
        pairs.append(
            InstitutionPair(
                inst_id=inst_id,
                pdp_id=pdp_id,
                institution_name=str(
                    cohort.get("institution_name")
                    or course.get("institution_name")
                    or ""
                ),
                stamp=stamp,
                cohort_file_name=output_file_name_from_sftp(str(cohort["file_name"])),
                course_file_name=output_file_name_from_sftp(str(course["file_name"])),
                cohort_fingerprint=str(cohort["file_fingerprint"]),
                course_fingerprint=str(course["file_fingerprint"]),
                cohort_split_path=split_csv_path(
                    tmp_dir,
                    str(cohort["file_fingerprint"]),
                    pdp_id,
                    str(cohort["file_name"]),
                ),
                course_split_path=split_csv_path(
                    tmp_dir,
                    str(course["file_fingerprint"]),
                    pdp_id,
                    str(course["file_name"]),
                ),
            )
        )
    return pairs, problems


def fingerprints_ready_to_close(
    rows: Sequence[Mapping[str, Any]],
    materialized: set[tuple[str, str, str]],
) -> list[str]:
    """Fingerprints whose every on-disk split is part of a materialized batch.

    ``rows`` need ``file_fingerprint``, ``inst_id``, ``stamp``, and ``split_exists``.
    ``materialized`` entries are ``(inst_id, stamp, file_fingerprint)``.
    """
    required: dict[str, set[tuple[str, str]]] = {}
    for row in rows:
        if not row.get("split_exists"):
            continue
        fingerprint = str(row.get("file_fingerprint") or "")
        inst_id = str(row.get("inst_id") or "")
        stamp = str(row.get("stamp") or "")
        if not fingerprint or not inst_id or not stamp:
            continue
        required.setdefault(fingerprint, set()).add((inst_id, stamp))
    return sorted(
        fingerprint
        for fingerprint, keys in required.items()
        if keys
        and all(
            (inst_id, stamp, fingerprint) in materialized for inst_id, stamp in keys
        )
    )


def batch_dir_has_inputs(batch_dir: str, *file_names: str) -> bool:
    """True when ``_SUCCESS.json`` and each named file are in ``batch_dir``."""
    if not os.path.isfile(os.path.join(batch_dir, SUCCESS_FILENAME)):
        return False
    return all(os.path.isfile(os.path.join(batch_dir, name)) for name in file_names)


def register_pair(
    client: object,
    pair: InstitutionPair,
    *,
    db_workspace: str,
    bronze_dir: str,
) -> dict[str, str]:
    """Upload missing files, create ``nsc_{stamp}``, and copy validated objects."""
    import requests

    from edvise.utils.api_requests import (
        create_institution_batch,
        get_institution_input,
        get_upload_url,
        put_csv_bytes,
        validate_sftp_upload,
    )
    from edvise.utils.gcs import assert_safe_volume_segment

    batch_name = batch_name_for_stamp(pair.stamp)
    overview = get_institution_input(client, pair.inst_id)  # type: ignore[arg-type]
    decision = plan_registration(
        overview,
        cohort_file_name=pair.cohort_file_name,
        course_file_name=pair.course_file_name,
        batch_name=batch_name,
    )
    if decision.error:
        raise RuntimeError(decision.error)

    bucket = external_bucket_name(db_workspace, pair.inst_id)
    split_paths = {
        pair.cohort_file_name: pair.cohort_split_path,
        pair.course_file_name: pair.course_split_path,
    }
    for name in decision.upload_names:
        with open(split_paths[name], "rb") as handle:
            payload = handle.read()
        signed_url = get_upload_url(client, pair.inst_id, name)  # type: ignore[arg-type]
        signed_bucket = bucket_from_signed_url(signed_url)
        if signed_bucket != bucket:
            raise RuntimeError(
                f"Signed URL bucket {signed_bucket} does not match {bucket}."
            )
        try:
            put_csv_bytes(signed_url, payload)
        except requests.HTTPError as exc:
            response = exc.response
            already_there = (
                response is not None
                and response.status_code == 400
                and "already exists" in (response.text or "").lower()
            )
            if not already_there:
                raise
        validate_sftp_upload(client, pair.inst_id, name)  # type: ignore[arg-type]

    if decision.create_batch:
        created = create_institution_batch(
            client,  # type: ignore[arg-type]
            pair.inst_id,
            batch_name,
            [pair.cohort_file_name, pair.course_file_name],
        )
        attached = set((created.get("file_names_to_ids") or {}).keys())
        expected = {pair.cohort_file_name, pair.course_file_name}
        if attached != expected:
            raise RuntimeError(
                f"Batch {batch_name} attached {sorted(attached)}, expected {sorted(expected)}."
            )
        batch_id = str(created.get("batch_id") or "")
    else:
        batch_id = decision.batch_id
    if not batch_id:
        raise RuntimeError(f"Batch {batch_name} did not return a batch_id.")
    batch_id = assert_safe_volume_segment("batch_id", batch_id)
    landing = os.path.join(bronze_dir, "gcs_uploads", batch_id)
    copy_validated_batch_inputs(
        batch_dir=landing,
        bucket_name=bucket,
        cohort_file_name=pair.cohort_file_name,
        course_file_name=pair.course_file_name,
    )
    return {
        "batch_id": batch_id,
        "batch_name": batch_name,
        "bronze_path": landing,
        "status": "BRONZE_MATERIALIZED",
    }


def copy_validated_batch_inputs(
    *,
    batch_dir: str,
    bucket_name: str,
    cohort_file_name: str,
    course_file_name: str,
    storage_client: Optional[object] = None,
) -> str:
    """Copy validated cohort and course objects into ``batch_dir`` when needed."""
    from edvise.utils.databricks import local_fs_path
    from edvise.utils.gcs import (
        DEFAULT_GCS_PREFIX,
        copy_validated_blobs_to_landing,
        write_success_marker,
    )

    if batch_dir_has_inputs(batch_dir, cohort_file_name, course_file_name):
        return batch_dir
    include_paths = [
        f"{DEFAULT_GCS_PREFIX}{cohort_file_name}",
        f"{DEFAULT_GCS_PREFIX}{course_file_name}",
    ]
    os.makedirs(local_fs_path(batch_dir), exist_ok=True)
    copied = copy_validated_blobs_to_landing(
        bucket_name=bucket_name,
        gcs_prefix=DEFAULT_GCS_PREFIX,
        landing_dir=batch_dir,
        include_paths=include_paths,
        max_objects=len(include_paths),
        storage_client=storage_client,  # type: ignore[arg-type]
    )
    if copied != len(include_paths):
        raise FileNotFoundError(
            f"Expected {len(include_paths)} objects in gs://{bucket_name}/"
            f"{DEFAULT_GCS_PREFIX}, copied {copied}."
        )
    write_success_marker(
        batch_dir,
        copied=copied,
        bucket_name=bucket_name,
        gcs_prefix=DEFAULT_GCS_PREFIX,
        storage_layout="gcs_uploads",
        copy_mode="selective",
        include_blob_paths=include_paths,
    )
    return batch_dir
