from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edvise.configs.custom import CleaningConfig


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


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


class IdentityAgentInputsConfig(StrictBaseModel):
    """
    Per-institution inputs: ``[institution]`` (id) and ``[files]`` (dataset → path(s)).

    File values may be a single string or a list of strings (e.g. multiple course files).
    Resolve basenames against your bronze volume root or use absolute paths.

    Load with :func:`edvise.dataio.read.read_config` and ``schema=IdentityAgentInputsConfig``,
    then :meth:`to_school_mapping_config` for :class:`SchoolMappingConfig` (``primary_keys`` unset).
    """

    institution: InstitutionIdSection
    files: Dict[str, Union[str, List[str]]]

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

    def to_school_mapping_config(self) -> SchoolMappingConfig:
        """Build :class:`SchoolMappingConfig` with ``DatasetConfig`` entries (files only, no PKs)."""
        datasets: Dict[str, DatasetConfig] = {}
        for name, spec in self.files.items():
            paths: List[str] = [spec] if isinstance(spec, str) else list(spec)
            if not paths:
                raise ValueError(f"files[{name!r}] must list at least one path")
            datasets[name] = DatasetConfig(files=paths, primary_keys=None)
        return SchoolMappingConfig(
            institution_id=self.institution.id,
            datasets=datasets,
        )
