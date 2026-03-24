from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from edvise.configs.custom import CleaningConfig


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class DatasetConfig(StrictBaseModel):
    primary_keys: Optional[List[str]]= Field(
        ...,
        min_length=1,
        description="Primary keys for this logical dataset",
    )
    files: List[str] = Field(
        ...,
        min_length=1,
        description="One or more file paths for this logical dataset",
    )


class SchoolMappingConfig(StrictBaseModel):
    institution_id: str
    institution_name: Optional[str] = None
    target_cohort_schema: str = "RawEdviseCohortDataSchema"
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
