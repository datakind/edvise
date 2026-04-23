"""
Read and write file contents on Unity Catalog volumes via the Databricks Files API
(``/api/2.0/fs/files/...``). Mirrors :class:`databricks.sdk.service.files.FilesAPI`.

The GenAI HITL Streamlit app is deployed as a self-contained directory; it cannot import
``edvise`` at runtime, so this module is local to the app bundle. Typical GenAI HITL paths
are on the institution **silver** volume, under ``.../genai_mapping/runs/...`` (see
``edvise_genai_ia`` / ``edvise_genai_sma`` in ``edvise.genai.mapping.scripts``).
"""

from __future__ import annotations

import io
from typing import cast

from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config


def _client() -> WorkspaceClient:
    return WorkspaceClient(config=Config())


def read_unity_file_bytes(absolute_path: str) -> bytes:
    """
    Read a file at a UC volume (or other Files-API) absolute path, e.g.
    ``/Volumes/catalog/schema/vol/path/identity_grain_hitl.json``.
    """
    path = str(absolute_path).strip()
    if not path or not path.startswith("/Volumes/"):
        raise ValueError("Path must be a non-empty Unity Catalog /Volumes/… absolute path")
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
    if not path or not path.startswith("/Volumes/"):
        raise ValueError("Path must be a non-empty Unity Catalog /Volumes/… absolute path")
    c = _client()
    c.files.upload(
        path,
        io.BytesIO(text.encode(encoding)),
        overwrite=overwrite,
    )
