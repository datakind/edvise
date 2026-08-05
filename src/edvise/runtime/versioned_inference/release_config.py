"""Per-workspace defaults for versioned inference release bundles."""

from __future__ import annotations

RELEASE_BASE_BY_WORKSPACE: dict[str, str] = {
    "dev_sst_02": "/Volumes/dev_sst_02/default/edvise_releases",
    "staging_sst_01": "/Volumes/staging_sst_01/default/edvise_releases",
}


def default_release_base_path(db_workspace: str) -> str:
    """Return the UC volume base for ``edvise_releases`` in ``db_workspace``."""
    key = (db_workspace or "").strip()
    if key in RELEASE_BASE_BY_WORKSPACE:
        return RELEASE_BASE_BY_WORKSPACE[key]
    if not key:
        msg = "db_workspace must be non-empty to resolve release_base_path"
        raise ValueError(msg)
    return f"/Volumes/{key}/default/edvise_releases"


def resolve_release_base_path(db_workspace: str, explicit: str | None = None) -> str:
    """Prefer an explicit job parameter; otherwise derive from workspace catalog."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return default_release_base_path(db_workspace)
