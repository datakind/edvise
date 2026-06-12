from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from edvise.configs.custom import CleaningConfig
from edvise.genai.mapping.shared.pipeline_artifacts import (
    coerce_pipeline_version,
    default_pipeline_version,
    resolve_onboard_run_id,
)
from edvise.genai.mapping.shared.volume_paths import (
    bronze_volume_path_for_institution,
    resolve_genai_data_path,
    resolve_genai_inputs_toml_path,
    silver_genai_mapping_root,
)

__all__ = [
    "DatasetConfig",
    "IdentityAgentDatasets",
    "IdentityAgentInputsConfig",
    "SchoolMappingConfig",
    "bronze_volume_path_for_institution",
    "resolve_genai_data_path",
    "resolve_genai_inputs_toml_path",
    "silver_genai_mapping_root",
]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DatasetConfig(StrictBaseModel):
    files: List[str] = Field(..., min_length=1)
    primary_keys: Optional[List[str]] = Field(default=None, min_length=1)
    student_id_alias: Optional[str] = None


class SchoolMappingConfig(StrictBaseModel):
    institution_id: str
    institution_name: Optional[str] = None
    onboard_run_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("onboard_run_id", "pipeline_run_id"),
        description="Onboard run folder id; see pipeline_artifacts.resolve_onboard_run_id.",
    )
    pipeline_version: str = Field(default_factory=default_pipeline_version)
    bronze_volumes_path: Optional[str] = None
    cleaning: Optional[CleaningConfig] = None
    datasets: Dict[str, DatasetConfig]


def _validate_dataset_files_table(field_label: str, v: object) -> object:
    if not isinstance(v, dict):
        raise ValueError(
            f"{field_label} must be a table mapping dataset names to path(s)"
        )
    for key, val in v.items():
        if isinstance(val, str):
            continue
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            continue
        raise ValueError(
            f"{field_label}[{key!r}] must be a string or a list of strings, got {type(val).__name__}"
        )
    return v


class IdentityAgentDatasets(StrictBaseModel):
    """``[datasets.onboard_files]`` and optional ``[datasets.execute_files]`` in inputs.toml."""

    onboard_files: Dict[str, Union[str, List[str]]] = Field(..., min_length=1)
    execute_files: Optional[
        Annotated[Dict[str, Union[str, List[str]]], Field(min_length=1)]
    ] = None

    @field_validator("onboard_files", "execute_files", mode="before")
    @classmethod
    def _files_values_are_str_or_list(cls, v: object, info: ValidationInfo) -> object:
        if info.field_name == "execute_files" and v is None:
            return None
        return _validate_dataset_files_table(str(info.field_name), v)


class IdentityAgentInputsConfig(StrictBaseModel):
    """
    Per-institution Identity Agent ``inputs.toml``.

    Load with ``read_config(..., schema=IdentityAgentInputsConfig)``, then
    :meth:`to_school_mapping_config` for :class:`SchoolMappingConfig`.

    Legacy ``[institution] id = …`` is accepted and normalized to ``institution_id``.
    """

    institution_id: str
    datasets: IdentityAgentDatasets

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_institution_table(cls, data: object) -> object:
        """Accept deprecated ``[institution] id = …`` and normalize to ``institution_id``."""
        if not isinstance(data, dict):
            return data
        out = dict(data)
        inst_id = out.get("institution_id")
        legacy = out.get("institution")
        if inst_id is not None and str(inst_id).strip():
            if isinstance(legacy, dict) and legacy.get("id"):
                if str(legacy["id"]).strip() != str(inst_id).strip():
                    raise ValueError(
                        "institution_id conflicts with deprecated [institution].id"
                    )
            out.pop("institution", None)
            return out
        if isinstance(legacy, dict) and legacy.get("id"):
            out["institution_id"] = legacy["id"]
            out.pop("institution", None)
            return out
        return out

    def to_school_mapping_config(
        self,
        *,
        uc_catalog: str,
        pipeline_mode: Literal["onboard", "execute"],
        onboard_run_id: Optional[str] = None,
        pipeline_version: Optional[str] = None,
    ) -> SchoolMappingConfig:
        ds = self.datasets
        if pipeline_mode == "onboard":
            merged_files = dict(ds.onboard_files)
        else:
            if not ds.execute_files:
                raise ValueError(
                    "datasets.execute_files is required when pipeline_mode is 'execute'. "
                    "Add a non-empty [datasets.execute_files] table to inputs.toml (logical "
                    "dataset name → CSV path), or run in onboard mode."
                )
            merged_files = dict(ds.execute_files)

        datasets: Dict[str, DatasetConfig] = {}
        for name, spec in merged_files.items():
            paths: List[str] = [spec] if isinstance(spec, str) else list(spec)
            if not paths:
                raise ValueError(
                    f"datasets.{pipeline_mode}_files[{name!r}] must list at least one path"
                )
            datasets[name] = DatasetConfig(files=paths, primary_keys=None)
        rid = resolve_onboard_run_id(onboard_run_id, create_if_missing=False)
        pv = coerce_pipeline_version(pipeline_version)
        return SchoolMappingConfig(
            institution_id=self.institution_id,
            datasets=datasets,
            bronze_volumes_path=bronze_volume_path_for_institution(
                self.institution_id, catalog=uc_catalog
            ),
            onboard_run_id=rid,
            pipeline_version=pv,
        )
