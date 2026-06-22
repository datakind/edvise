"""
Read and write file contents on Unity Catalog volumes via the Databricks Files API.

Re-exports :mod:`edvise.genai.mapping.shared.unity_volume_files` so the Streamlit app
can keep its existing import path. Typical GenAI HITL paths are on the institution
**silver** volume, under ``.../genai_mapping/runs/...`` (same layout as
``edvise_genai_ia`` / ``edvise_genai_sma``).
"""

from __future__ import annotations

from edvise.genai.mapping.shared.unity_volume_files import (
    is_unity_catalog_volume_path,
    read_unity_file_bytes,
    read_unity_file_text,
    write_unity_file_text,
)

__all__ = [
    "is_unity_catalog_volume_path",
    "read_unity_file_bytes",
    "read_unity_file_text",
    "write_unity_file_text",
]
