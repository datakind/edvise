"""
Read and write file contents on Unity Catalog volumes via the Databricks Files API.

Used by GenAI mapping repair helpers and the HITL Streamlit app (paths under
``.../genai_mapping/runs/...`` on institution silver volumes).
"""

from __future__ import annotations

import io
from typing import cast

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


def is_unity_catalog_volume_path(path: str) -> bool:
    """True when ``path`` is a non-empty UC ``/Volumes/…`` absolute path."""
    p = str(path or "").strip()
    return bool(p) and p.startswith("/Volumes/")


def _client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


def read_unity_file_bytes(absolute_path: str) -> bytes:
    """
    Read a file at a UC volume (or other Files-API) absolute path, e.g.
    ``/Volumes/catalog/schema/vol/path/manifest_map.json``.
    """
    path = str(absolute_path).strip()
    if not is_unity_catalog_volume_path(path):
        raise ValueError(
            "Path must be a non-empty Unity Catalog /Volumes/… absolute path"
        )
    c = _client()
    out = c.files.download(path)
    if out.contents is None:
        return b""
    return cast(bytes, out.contents.read())


def read_unity_file_text(absolute_path: str, *, encoding: str = "utf-8") -> str:
    return read_unity_file_bytes(absolute_path).decode(encoding)


def write_unity_file_text(
    absolute_path: str, text: str, *, encoding: str = "utf-8", overwrite: bool = True
) -> None:
    path = str(absolute_path).strip()
    if not is_unity_catalog_volume_path(path):
        raise ValueError(
            "Path must be a non-empty Unity Catalog /Volumes/… absolute path"
        )
    c = _client()
    c.files.upload(
        path,
        io.BytesIO(text.encode(encoding)),
        overwrite=overwrite,
    )
