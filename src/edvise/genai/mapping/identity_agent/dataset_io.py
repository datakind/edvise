"""
Load bronze tabular files for configured school datasets (Identity Agent / genai cleaning).

CSV reads go through :func:`edvise.dataio.read.from_csv_file` for consistent loading behavior.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from edvise.configs.genai import SchoolMappingConfig, resolve_genai_data_path
from edvise.dataio.read import from_csv_file


def load_school_dataset_dataframe(
    school: SchoolMappingConfig,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Read all configured CSV files for ``dataset_name`` and concatenate.

    Paths are resolved with :func:`~edvise.configs.genai.resolve_genai_data_path`
    against ``school.bronze_volumes_path``. Each file is loaded via
    :func:`~edvise.dataio.read.from_csv_file`.
    """
    if dataset_name not in school.datasets:
        raise KeyError(
            f"Unknown dataset {dataset_name!r} for this school "
            f"(have {list(school.datasets.keys())!r})"
        )
    ds = school.datasets[dataset_name]
    dfs: list[pd.DataFrame] = []
    for fp in ds.files:
        resolved = resolve_genai_data_path(school.bronze_volumes_path, fp)
        p = Path(resolved)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        dfs.append(from_csv_file(str(p), low_memory=False))
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


__all__ = ["load_school_dataset_dataframe"]
