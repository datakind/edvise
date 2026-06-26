"""Apply ES batch GCS ingest bronze dir to GenAI mapping school configs."""

from __future__ import annotations

import logging
from pathlib import Path

from edvise.configs.genai import DatasetConfig, SchoolMappingConfig
from edvise.dataio.batch_dataset_paths import resolve_dataset_file_in_batch_dir

LOGGER = logging.getLogger(__name__)


def apply_bronze_batch_dir_overrides(
    school_config: SchoolMappingConfig,
    *,
    bronze_batch_dir: str | None,
) -> SchoolMappingConfig:
    """
    Point GenAI execute datasets at CSVs under ``bronze_batch_dir``.

    Each configured file in ``inputs.toml`` is resolved by basename (then substring)
    under the batch landing dir from ``batch_gcs_ingest``, e.g.
    ``gcs_uploads/{batch_id}/``.
    """
    batch_dir = (bronze_batch_dir or "").strip()
    if not batch_dir:
        return school_config

    datasets: dict[str, DatasetConfig] = {}
    for ds_name, ds_cfg in school_config.datasets.items():
        resolved_files: list[str] = []
        for configured_path in ds_cfg.files:
            needle = Path(configured_path).name or ds_name
            resolved = resolve_dataset_file_in_batch_dir(batch_dir, needle)
            if resolved is None:
                raise FileNotFoundError(
                    f"GenAI dataset {ds_name!r}: no file matching {needle!r} under batch dir "
                    f"{batch_dir!r} (from inputs.toml path {configured_path!r})."
                )
            resolved_files.append(resolved)
            LOGGER.info(
                "GenAI dataset %r: using batch file %s (inputs.toml had %r)",
                ds_name,
                resolved,
                configured_path,
            )
        datasets[ds_name] = ds_cfg.model_copy(update={"files": resolved_files})

    return school_config.model_copy(update={"datasets": datasets})
