from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edvise.configs.custom import CleaningConfig


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def bronze_volume_path_for_institution(
    institution_id: str,
    *,
    catalog: str,
) -> str:
    """
    Databricks Unity Catalog bronze volume path for an institution.

    Returns ``/Volumes/<catalog>/<institution_id>_bronze/bronze_volume``.
    The UC *catalog* is supplied by the caller (e.g. a notebook variable or job parameter),
    not by edvise defaults.
    """
    cat = catalog.strip()
    if not cat:
        raise ValueError("catalog must be non-empty")
    inst = institution_id.strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    return f"/Volumes/{cat}/{inst}_bronze/bronze_volume"


def resolve_genai_data_path(bronze_volumes_path: Optional[str], file_path: str) -> str:
    """
    Join ``file_path`` to ``bronze_volumes_path`` when the path is relative.

    Absolute ``file_path`` values (e.g. Databricks ``/Volumes/...``) are returned unchanged.
    When ``bronze_volumes_path`` is missing or blank, ``file_path`` is returned as-is.

    Use this for CSV reads and for writing identity-agent cleaned outputs under the same root.
    Materialized hook modules (:attr:`HookSpec.file`) use the same relative layout under
    ``bronze_volumes_path`` (see ``identity_hooks/`` in
    :mod:`edvise.genai.mapping.identity_agent.hitl.hook_generation.paths`).
    """
    if not bronze_volumes_path or not str(bronze_volumes_path).strip():
        return file_path
    p = Path(file_path)
    if p.is_absolute():
        return file_path
    root = Path(bronze_volumes_path.rstrip("/"))
    return str(root / p)


class DatasetConfig(StrictBaseModel):
    files: List[str] = Field(
        ...,
        min_length=1,
        description="One or more file paths for this logical dataset",
    )
    primary_keys: Optional[List[str]] = Field(
        default=None,
        description=(
            "Primary keys for this logical dataset. Omit for identity-only inputs "
            "(filled from IdentityAgent grain contracts via merge_grain_contracts_into_school_config)."
        ),
    )

    @model_validator(mode="after")
    def _primary_keys_nonempty_when_present(self) -> DatasetConfig:
        if self.primary_keys is not None and len(self.primary_keys) < 1:
            raise ValueError("primary_keys, when set, must be non-empty")
        return self


class SchoolMappingConfig(StrictBaseModel):
    institution_id: str
    institution_name: Optional[str] = None
    bronze_volumes_path: Optional[str] = Field(
        default=None,
        description=(
            "Root path on UC/Databricks volumes for relative entries in "
            "``datasets.*.files``. Also the base directory for identity-agent cleaned-data I/O."
        ),
    )
    target_cohort_schema: str = "RawEdviseStudentDataSchema"
    target_course_schema: str = "RawEdviseCourseDataSchema"
    cleaning: Optional[CleaningConfig] = Field(
        default=None,
        description=(
            "Optional CleaningConfig (e.g. student_id_alias). Same semantics as custom "
            "pipeline cleaning in edvise.data_audit.custom_cleaning."
        ),
    )
    datasets: Dict[str, DatasetConfig] = Field(
        ...,
        description="Logical dataset configs keyed by dataset name",
    )
    notes: Optional[str] = None


class MappingProjectConfig(StrictBaseModel):
    """
    Root model for ``pipelines/gen_ai_cleaning/inputs.toml``.

    Load with :func:`edvise.dataio.read.read_config` and ``schema=MappingProjectConfig``.
    Nested tables map naturally, e.g. ``[schools.<id>.cleaning]`` → ``schools.<id>.cleaning``
    as :class:`CleaningConfig` (``student_id_alias``, etc.).
    """

    schools: Dict[str, SchoolMappingConfig]

    @classmethod
    def from_toml_dict(cls, data: Dict[str, dict]) -> "MappingProjectConfig":
        schools = {}
        for school_key, school_value in data.items():
            schools[school_key] = SchoolMappingConfig(
                institution_id=school_key,
                **school_value,
            )
        return cls(schools=schools)


class InstitutionIdSection(StrictBaseModel):
    """``[institution]`` block in per-institution ``inputs.toml`` for IdentityAgent."""

    id: str = Field(..., description="Institution identifier (snake_case)")


class IdentityAgentDatasets(StrictBaseModel):
    """
    ``[datasets]`` in per-institution ``inputs.toml`` — a flat ``files`` map only.

    The bronze volume root is not stored here; :meth:`IdentityAgentInputsConfig.to_school_mapping_config`
    sets :attr:`SchoolMappingConfig.bronze_volumes_path` via :func:`bronze_volume_path_for_institution`
    from ``[institution].id`` and the caller-supplied Unity Catalog name.

    TOML example::

        [datasets.files]
        student = "roster.csv"
    """

    files: Dict[str, Union[str, List[str]]] = Field(
        ...,
        description=(
            "Logical dataset name → CSV path(s), relative to the resolved bronze volume root "
            "when that root is set."
        ),
    )

    @field_validator("files", mode="before")
    @classmethod
    def _files_values_are_str_or_list(cls, v: object) -> object:
        if not isinstance(v, dict):
            raise ValueError("files must be a table mapping dataset names to path(s)")
        for key, val in v.items():
            if isinstance(val, str):
                continue
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                continue
            raise ValueError(
                f"files[{key!r}] must be a string or a list of strings, got {type(val).__name__}"
            )
        return v


class IdentityAgentInputsConfig(StrictBaseModel):
    """
    Per-institution config: ``[institution]`` and ``[datasets.files]``.

    File values may be a single string or a list of strings (e.g. multiple course files).
    Relative paths resolve against :func:`bronze_volume_path_for_institution` once you pass the
    Unity Catalog name to :meth:`to_school_mapping_config`.
    Use absolute paths in ``files`` when reading from outside that layout.

    Load with :func:`edvise.dataio.read.read_config` and ``schema=IdentityAgentInputsConfig``,
    then :meth:`to_school_mapping_config` for :class:`SchoolMappingConfig` (``primary_keys`` unset).
    """

    institution: InstitutionIdSection
    datasets: IdentityAgentDatasets

    def to_school_mapping_config(self, *, uc_catalog: str) -> SchoolMappingConfig:
        """Build :class:`SchoolMappingConfig` with ``DatasetConfig`` entries (files only, no PKs)."""
        ds = self.datasets
        datasets: Dict[str, DatasetConfig] = {}
        for name, spec in ds.files.items():
            paths: List[str] = [spec] if isinstance(spec, str) else list(spec)
            if not paths:
                raise ValueError(
                    f"datasets.files[{name!r}] must list at least one path"
                )
            datasets[name] = DatasetConfig(files=paths, primary_keys=None)
        return SchoolMappingConfig(
            institution_id=self.institution.id,
            datasets=datasets,
            bronze_volumes_path=bronze_volume_path_for_institution(
                self.institution.id, catalog=uc_catalog
            ),
        )
