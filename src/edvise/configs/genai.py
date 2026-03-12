from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


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
    datasets: Dict[str, DatasetConfig] = Field(
        ...,
        description="Logical dataset configs keyed by dataset name",
    )
    notes: Optional[str] = None


class MappingProjectConfig(StrictBaseModel):
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