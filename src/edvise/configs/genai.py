from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from edvise.configs.custom import CleaningConfig


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class KeyCollisionDedupeConfig(StrictBaseModel):
    """
    Declarative key-collision handling (schema-agnostic).

    Applied during schema-contract preprocessing after column normalization,
    term order, and optional ``student_id`` alias rename. See
    :func:`edvise.data_audit.identity_agent.deduplication.apply_key_collision_dedupe_config`.
    """

    key_cols: List[str] = Field(
        ...,
        min_length=1,
        description="Columns defining duplicate groups to resolve.",
    )
    conflict_columns: List[str] = Field(
        default_factory=list,
        description=(
            "Columns that may differ within a duplicate key group; if any vary, "
            "``disambiguate_column`` is suffixed. Omitted or missing columns are "
            "skipped at runtime."
        ),
    )
    disambiguate_column: str = Field(
        ...,
        description="Column to suffix (normally one of ``key_cols``) when conflicts exist.",
    )
    disambiguate_sep: str = Field(default="-")
    drop_full_row_duplicates_first: bool = Field(default=True)
    disambiguate_sort_by: Optional[List[str]] = Field(
        default=None,
        description="Optional sort keys before assigning -1/-2 suffixes within groups.",
    )
    disambiguate_sort_ascending: bool = Field(default=True)
    when_no_conflict_keep: Literal["first", "last"] = "first"
    no_conflict_sort_by: Optional[List[str]] = Field(
        default=None,
        description="When dropping redundant key duplicates, sort by these first.",
    )
    no_conflict_sort_ascending: bool = Field(default=False)


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
    dedupe: Optional[KeyCollisionDedupeConfig] = Field(
        default=None,
        description="Optional key-collision dedupe (identity_agent) for this dataset.",
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