import re
import typing as t

import pydantic as pyd

# allowed primary metrics for h2o
_ALLOWED_PRIMARY_METRICS = {
    "logloss",
    "auc",
    "aucpr",
    "rmse",
    "mae",
    "mean_per_class_error",
}


class CustomProjectConfig(pyd.BaseModel):
    """Configuration schema for SST Custom projects."""

    institution_id: str = pyd.Field(
        ...,
        description=(
            "Unique (ASCII-only) identifier for institution; used in naming things "
            "such as source directories, catalog schemas, keys in shared configs, etc."
        ),
    )
    institution_name: str = pyd.Field(
        ...,
        description=(
            "Readable 'display' name for institution, distinct from the 'id'; "
            "probably just the school's 'official', public name"
        ),
    )

    # shared parameters
    student_id_col: str = "student_id"
    target_col: str = "target"
    split_col: str = "split"
    sample_weight_col: t.Optional[str] = "sample_weight"
    student_group_cols: t.Optional[list[str]] = pyd.Field(
        default=["age_group", "race", "gender"],
        description=(
            "One or more column names in datasets containing student 'groups' "
            "to use for model bias assessment, but *not* as model features"
        ),
    )
    student_group_aliases: dict[str, str] = pyd.Field(
        default_factory=dict,
        description=(
            "Mapping from raw column name (e.g., GENDER_DESC) to "
            "friendly label (e.g., 'Gender') for use in model cards"
        ),
    )
    pred_col: str = "pred"
    pred_prob_col: str = "pred_prob"
    pos_label: t.Optional[bool | str] = True
    random_state: t.Optional[int] = 12345
    pipeline_version: str = "v0.1.2"

    # key artifacts produced by project pipeline
    datasets: "AllDatasetStagesConfig" = pyd.Field(
        description=(
            "Key datasets produced by the pipeline represented here in this config"
        ),
    )
    model: t.Optional["ModelConfig"] = pyd.Field(
        default=None,
        description=(
            "Essential metadata for identifying and loading trained model artifacts, "
            "as produced by the pipeline represented here in this config"
        ),
    )

    # key steps in project pipeline
    preprocessing: t.Optional["PreprocessingConfig"] = None
    modeling: t.Optional["ModelingConfig"] = None
    inference: t.Optional["InferenceConfig"] = None

    @pyd.computed_field  # type: ignore[misc]
    @property
    def non_feature_cols(self) -> list[str]:
        return (
            [self.student_id_col, self.target_col]
            + ([self.split_col] if self.split_col else [])
            + ([self.sample_weight_col] if self.sample_weight_col else [])
            + (self.student_group_cols or [])
        )

    @pyd.field_validator("institution_id", mode="after")
    @classmethod
    def check_institution_id_isascii(cls, value: str) -> str:
        if not re.search(r"^\w+$", value, flags=re.ASCII):
            raise ValueError(f"institution_id='{value}' is not ASCII-only")
        return value

    # NOTE: this is for *pydantic* model -- not ML model -- configuration
    model_config = pyd.ConfigDict(extra="forbid", strict=True)

    @pyd.model_validator(mode="after")
    def check_sample_weight_requires_random_state(self):
        if self.sample_weight_col and self.random_state is None:
            raise ValueError(
                "random_state must be specified if sample_weight_col is provided"
            )
        return self

    @pyd.model_validator(mode="after")
    def validate_student_group_aliases(self) -> "CustomProjectConfig":
        missing = [
            col
            for col in (self.student_group_cols or [])
            if col not in (self.student_group_aliases or {})
        ]
        if missing:
            raise ValueError(f"Missing student_group_aliases for: {missing}")
        return self

    @pyd.model_validator(mode="after")
    def _normalize_and_validate_primary_metric(self) -> "CustomProjectConfig":
        if (
            self.modeling
            and self.modeling.training
            and self.modeling.training.primary_metric
        ):
            pm = self.modeling.training.primary_metric

            # Normalize legacy spelling to H2O spelling
            if pm in {"logloss", "log_loss"}:
                pm = "logloss"

            if pm not in _ALLOWED_PRIMARY_METRICS:
                raise ValueError(
                    f"Unsupported primary_metric '{pm}' for H2O. "
                    f"Allowed: {sorted(_ALLOWED_PRIMARY_METRICS)}"
                )

            self.modeling.training.primary_metric = pm
        return self

    @pyd.model_validator(mode="after")
    def _validate_inference_background_sample(self) -> "CustomProjectConfig":
        if self.inference and self.inference.background_data_sample is not None:
            n = self.inference.background_data_sample
            if not (500 <= n <= 2000):
                raise ValueError(
                    "For H2O, inference.background_data_sample must be between 500 and 2000."
                )
        return self


class DatasetConfig(pyd.BaseModel):
    train_file_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Absolute path to training dataset on disk.",
    )
    predict_file_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Absolute path to prediction/inference dataset on disk.",
    )
    train_table_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Unity Catalog table path for training dataset, e.g., 'catalog.schema.table'.",
    )
    predict_table_path: t.Optional[str] = pyd.Field(
        default=None,
        description="Unity Catalog table path for prediction/inference dataset.",
    )
    file_path: t.Optional[str] = None
    table_path: t.Optional[str] = None

    primary_keys: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Primary keys utilized for data validation, if applicable",
    )
    drop_cols: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Columns to be dropped during pre-processing, if applicable",
    )
    non_null_cols: t.Optional[t.List[str]] = pyd.Field(
        default=None,
        description="Columns to be validated as non-null, if applicable",
    )

    @pyd.model_validator(mode="after")
    def validate_paths(self) -> "DatasetConfig":
        any_paths = [
            self.train_file_path,
            self.predict_file_path,
            self.train_table_path,
            self.predict_table_path,
            self.file_path,  # Legacy, not used in pipeline/DB workflow
            self.table_path,  # Legacy, not used in pipeline/DB workflow
        ]
        if not any(any_paths):
            raise ValueError(
                "At least one dataset path must be specified: "
                "`train_file_path`, `predict_file_path`, "
                "`train_table_path`, `predict_table_path`, "
                "`file_path`, or `table_path`"
            )
        return self

    def get_path(self, mode: t.Literal["train", "predict"]) -> t.Optional[str]:
        """Convenience accessor for the train/predict path."""
        if mode == "train":
            return self.train_file_path or self.train_table_path
        elif mode == "predict":
            return self.predict_file_path or self.predict_table_path
        else:
            raise ValueError(f"Unknown mode: {mode}")


class AllDatasetStagesConfig(pyd.BaseModel):
    bronze: dict[str, DatasetConfig]
    silver: dict[str, DatasetConfig]
    gold: dict[str, DatasetConfig]


class ModelConfig(pyd.BaseModel):
    experiment_id: str
    run_id: str
    calibrate_underpred: t.Optional[bool] = False

    @pyd.computed_field  # type: ignore[misc]
    @property
    def mlflow_model_uri(self) -> str:
        return f"runs:/{self.run_id}/model"


class PreprocessingConfig(pyd.BaseModel):
    cleaning: t.Optional["CleaningConfig"] = None
    selection: "SelectionConfig"
    checkpoint: "CheckpointConfig"
    target: "TargetConfig"
    splits: dict[t.Literal["train", "test", "validate"], float] = pyd.Field(
        default={"train": 0.6, "test": 0.2, "validate": 0.2},
        description=(
            "Mapping of name to fraction of the full datset belonging to a given 'split', "
            "which is a randomized subset used for different parts of the modeling process"
        ),
    )
    sample_class_weight: t.Optional[t.Literal["balanced"] | dict[object, int]] = (
        pyd.Field(
            default=None,
            description=(
                "Weights associated with classes in the form ``{class_label: weight}`` "
                "or 'balanced' to automatically adjust weights inversely proportional "
                "to class frequencies in the input data. "
                "If null (default), then sample weights are not computed."
            ),
        )
    )

    @pyd.field_validator("splits", mode="after")
    @classmethod
    def check_split_fractions(cls, value: dict) -> dict:
        if (sum_fracs := sum(value.values())) != 1.0:
            raise pyd.ValidationError(
                f"split fractions must sum up to 1.0, but input sums up to {sum_fracs}"
            )
        return value


class CleaningConfig(pyd.BaseModel):
    schema_contract_path: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Absolute path on volumes to the schema_contract.json file. "
            "This file contains the frozen multi-dataset schema contract "
            "used for schema enforcement for custom schools. This is needed "
            "for data reliability and to ensure minimal training–inference skew."
        ),
    )
    student_id_alias: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Optional alternate name for the student_id column. "
            "E.g., Zogotech uses 'student_id_randomized_datakind'. "
            "If provided, it will be normalized to 'student_id'."
        ),
    )
    null_tokens: list[str] = pyd.Field(
        default=["(Blank)"],
        description=(
            "Tokens that should be treated as null/NA during cleaning. "
            "These will be replaced with pandas NA before dtype generation."
        ),
    )
    treat_empty_strings_as_null: bool = pyd.Field(
        default=True,
        description=(
            "If True, whitespace-only and empty strings are treated as null values."
        ),
    )
    date_formats: tuple[str, ...] = pyd.Field(
        default=("%m/%d/%Y",),
        description="Candidate date formats to try before falling back to generic parsing.",
    )
    dtype_confidence_threshold: float = pyd.Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum fraction of successfully coerced values required to accept a generated dtype."
        ),
    )
    min_non_null: int = pyd.Field(
        default=10,
        ge=0,
        description=(
            "Minimum number of non-null values required to trust a generated dtype."
        ),
    )
    boolean_map: dict[str, bool] = pyd.Field(
        default_factory=lambda: {
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
            "1": True,
            "0": False,
        },
        description=(
            "Mapping for interpreting string tokens as booleans during dtype generation."
        ),
    )
    forced_dtypes: dict[str, str] = pyd.Field(
        default_factory=dict,
        description=(
            "Optional mapping of normalized column names → forced pandas nullable dtypes "
            "(e.g. {'student_id': 'string', 'term_order': 'Int64'}). "
            "These overrides are applied BEFORE dtype inference across ALL datasets."
        ),
    )
    allow_forced_cast_fallback: bool = pyd.Field(
        default=True,
        description=(
            "If True, failures to cast a forced dtype fall back to inferred dtype with a warning. "
            "If False, such failures raise an exception."
        ),
    )


class CheckpointConfig(pyd.BaseModel):
    name: str = pyd.Field(default="checkpoint")
    params: dict[str, object] = pyd.Field(default_factory=dict)
    unit: t.Literal["credit", "year", "term", "semester"]
    value: int = pyd.Field(
        default=30,
        description=(
            "Number of checkpoint units (e.g. 1 year, 1 term/semester, 30 credits)"
        ),
    )
    optional_desc: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Optional description of the checkpoint beyond the unit and value. "
            "Used to provide further context for the particular institution and model. "
        ),
    )

    @pyd.field_validator("value")
    @classmethod
    def check_value_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be greater than zero.")
        return v

    @pyd.field_validator("name", mode="after")
    @classmethod
    def check_name_is_lowercase(cls, value: str) -> str:
        if value != value.lower():
            raise ValueError(
                f"checkpoint.name='{value}' must be lowercase. "
                "Unity Catalog will lowercase model names automatically. "
                "To ensure consistency across our codebase, we require lowercase names."
            )
        return value


class TargetConfig(pyd.BaseModel):
    name: str = pyd.Field(default="target")
    category: t.Literal["graduation", "retention"]
    params: dict[str, object] = pyd.Field(default_factory=dict)
    unit: t.Literal["credit", "year", "term", "semester", "pct_completion"]
    value: int = pyd.Field(
        default=120,
        description=(
            "Number of target units (e.g. 4 years, 4 terms, 120 credits, 150 completion %)"
        ),
    )
    optional_desc: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Optional description of the target beyond the unit and value. "
            "Used to provide further context for the particular institution and model. "
        ),
    )

    @pyd.field_validator("value")
    @classmethod
    def check_value_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Value must be greater than zero.")
        return v

    @pyd.field_validator("name", mode="after")
    @classmethod
    def check_name_is_lowercase(cls, value: str) -> str:
        if value != value.lower():
            raise ValueError(
                f"target.name='{value}' must be lowercase. "
                "Unity Catalog will lowercase model names automatically. "
                "To ensure consistency across our codebase, we require lowercase names."
            )
        return value


class SelectionConfig(pyd.BaseModel):
    student_criteria: dict[str, object] = pyd.Field(
        default_factory=dict,
        description=(
            "Column name in modeling dataset mapped to one or more values that it must equal "
            "in order for the corresponding student to be considered 'eligible'. "
            "Multiple criteria are combined with a logical 'AND'."
        ),
    )
    student_criteria_aliases: dict[str, str] = pyd.Field(
        default_factory=dict,
        description="Human-readable display names for student_criteria keys",
    )

    @pyd.model_validator(mode="after")
    def validate_criteria_aliases(self) -> "SelectionConfig":
        criteria_keys = self.student_criteria.keys()
        alias_keys = self.student_criteria_aliases or {}

        missing = [k for k in criteria_keys if k not in alias_keys]
        if missing:
            raise ValueError(
                f"Missing display aliases in `student_criteria_aliases` for: {missing}"
            )
        return self


class ModelingConfig(pyd.BaseModel):
    feature_selection: t.Optional["FeatureSelectionConfig"] = None
    training: "TrainingConfig"
    evaluation: t.Optional["EvaluationConfig"] = None
    bias_mitigation: t.Optional["BiasMitigationConfig"] = None


class FeatureSelectionConfig(pyd.BaseModel):
    """
    See Also:
        - :func:`modeling.feature_selection.select_features()`
    """

    force_include_cols: t.Optional[list[str]] = None
    incomplete_threshold: float = 0.5
    low_variance_threshold: float = 0.0
    collinear_threshold: t.Optional[float] = 10.0


class TrainingConfig(pyd.BaseModel):
    """
    References:
        - https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
    """

    exclude_cols: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="One or more column names in dataset to exclude from training.",
    )
    time_col: t.Optional[str] = pyd.Field(
        default=None,
        description=(
            "Column name in dataset used to split train/test/validate sets chronologically, "
            "as an alternative to the randomized assignment in ``split_col`` ."
        ),
    )
    exclude_frameworks: t.Optional[list[str]] = pyd.Field(
        default=None,
        description="List of algorithm frameworks that H2O AutoML excludes from training.",
    )
    primary_metric: str = pyd.Field(
        default="logloss",
        description="Metric used to evaluate and rank model performance.",
    )
    timeout_minutes: t.Optional[int] = pyd.Field(
        default=None,
        description="Maximum time to wait for H2O AutoML trials to complete.",
    )


class EvaluationConfig(pyd.BaseModel):
    topn_runs_included: int = pyd.Field(
        default=3,
        description="Number of top-scoring mlflow runs to include for detailed evaluation",
    )


class BiasMitigationConfig(pyd.BaseModel):
    student_group_col: str = pyd.Field(
        default="student_group",
        description="Column name in dataset to have a custom threshold set.",
    )
    student_group_col_alias: str = pyd.Field(
        default="Student Group",
        description="Human-readable display name for student_group column.",
    )
    student_group: str = pyd.Field(
        default="freshmen",
        description="Value in student_group column that has a custom threshold based on bias considerations.",
    )
    custom_threshold: float = pyd.Field(
        default=0.5,
        description="Threshold for student group based on bias considerations.",
    )


class InferenceConfig(pyd.BaseModel):
    num_top_features: int = pyd.Field(default=5)
    min_prob_pos_label: t.Optional[float] = 0.5
    background_data_sample: t.Optional[int] = 500
    cohort: t.Optional[list[str]] = ["fall 2024-25"]
