from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from edvise.configs.custom import CleaningConfig
from edvise.genai.mapping.shared.pipeline_artifacts import (
    resolve_onboard_run_id,
    resolve_pipeline_version,
    versioned_genai_run_root,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


def _require_uc_catalog(catalog: str) -> str:
    cat = str(catalog).strip()
    if not cat:
        raise ValueError(
            "catalog (Databricks UC workspace catalog, e.g. job ``DB_workspace`` / ``--catalog``) "
            "is required to resolve institution volume paths."
        )
    return cat


def bronze_volume_path_for_institution(
    institution_id: str,
    *,
    catalog: str = "",
) -> str:
    """
    Institution bronze UC volume root used to resolve relative dataset paths.

    Returns ``/Volumes/<catalog>/<institution_id>_bronze/bronze_volume`` (same layout as PDP /
    Streamlit HITL helpers).
    """
    inst = institution_id.strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    cat = _require_uc_catalog(catalog)
    return f"/Volumes/{cat}/{inst}_bronze/bronze_volume"


def silver_volume_path_for_institution(institution_id: str, *, catalog: str) -> str:
    """
    Institution silver UC volume root: ``/Volumes/<catalog>/<institution_id>_silver/silver_volume``.

    Aligns with PDP jobs (``--silver_volume_path``).
    """
    inst = institution_id.strip()
    if not inst:
        raise ValueError("institution_id must be non-empty")
    cat = _require_uc_catalog(catalog)
    return f"/Volumes/{cat}/{inst}_silver/silver_volume"


def silver_genai_mapping_root(institution_id: str, *, catalog: str) -> str:
    """GenAI mapping run/active folders: ``…/silver_volume/genai_mapping``."""
    return f"{silver_volume_path_for_institution(institution_id, catalog=catalog)}/genai_mapping"


def ia_inputs_toml_under_bronze(institution_id: str, *, catalog: str = "") -> str:
    """
    Default IdentityAgent ``inputs.toml`` path on the institution bronze volume.

    Layout: ``<bronze_volume_path_for_institution>/genai_mapping/inputs/inputs.toml``.

    ``catalog`` must be the UC workspace catalog (non-empty); see :func:`bronze_volume_path_for_institution`.
    """
    base = Path(bronze_volume_path_for_institution(institution_id, catalog=catalog))
    return str(base / "genai_mapping" / "inputs" / "inputs.toml")


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
    onboard_run_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("onboard_run_id", "pipeline_run_id"),
        description=(
            "Databricks **job run id** for this execution (artifact paths + UC registry). "
            "Resolved by :func:`~edvise.genai.mapping.shared.pipeline_artifacts.resolve_onboard_run_id` "
            "(Spark job conf, then ``DATABRICKS_JOB_RUN_ID``, then ``GENAI_ONBOARD_RUN_ID`` / legacy "
            "``GENAI_PIPELINE_RUN_ID`` for local). "
            "When set with ``bronze_volumes_path``, outputs live under "
            "``genai_pipeline/<onboard_run_id>/`` (institution is implied by the volume root). "
            "Release version is stored in ``genai_pipeline_run.json`` and UC, not in the path."
        ),
    )
    pipeline_version: str = Field(
        default_factory=resolve_pipeline_version,
        description=(
            "Release / **git tag** (e.g. ``0.2.0``). Defaults to env "
            "``GENAI_GIT_TAG`` / ``GIT_TAG`` / installed ``edvise`` package version. "
            "See :func:`~edvise.genai.mapping.shared.pipeline_artifacts.resolve_pipeline_version`."
        ),
    )
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

    def genai_versioned_run_root(self) -> Optional[str]:
        """
        Return the UC volume path for versioned GenAI artifacts, or None if not configured.

        Requires both ``bronze_volumes_path`` and ``onboard_run_id``.
        """
        if not self.bronze_volumes_path or not str(self.bronze_volumes_path).strip():
            return None
        rid = self.onboard_run_id
        if not rid or not str(rid).strip():
            return None
        return str(
            versioned_genai_run_root(
                self.bronze_volumes_path,
                str(rid).strip(),
            )
        )


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
    Relative paths resolve against :func:`bronze_volume_path_for_institution` for the institution
    (``/Volumes/<uc_catalog>/<id>_bronze/bronze_volume``). The ``uc_catalog`` argument to
    :meth:`to_school_mapping_config` sets that catalog segment.
    Use absolute paths in ``files`` when reading from outside that layout.

    Load with :func:`edvise.dataio.read.read_config` and ``schema=IdentityAgentInputsConfig``,
    then :meth:`to_school_mapping_config` for :class:`SchoolMappingConfig` (``primary_keys`` unset).
    """

    institution: InstitutionIdSection
    datasets: IdentityAgentDatasets

    def to_school_mapping_config(
        self,
        *,
        uc_catalog: str,
        onboard_run_id: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ) -> SchoolMappingConfig:
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
        rid = resolve_onboard_run_id(onboard_run_id, create_if_missing=False)
        pv = resolve_pipeline_version(pipeline_version)
        return SchoolMappingConfig(
            institution_id=self.institution.id,
            datasets=datasets,
            bronze_volumes_path=bronze_volume_path_for_institution(
                self.institution.id, catalog=uc_catalog
            ),
            onboard_run_id=rid,
            pipeline_version=pv,
        )
