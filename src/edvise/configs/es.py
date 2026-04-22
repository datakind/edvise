"""
Edvise Schema (ES) project configuration.

The TOML layout and validation rules match :mod:`edvise.configs.pdp`: top-level
``institution_id``, ``datasets`` (``raw_course``, ``raw_cohort``), optional
``model``, and pipeline sections ``preprocessing``, ``modeling``, and
``inference``. ES-specific behavior comes from raw file schemas, standardizers,
and feature column bundles (see :mod:`edvise.feature_generation`), not from
different keys on this config class.

Example sections:

- ``[datasets]`` — filenames for raw cohort and course uploads.
- ``[preprocessing]`` — features, selection, checkpoint, target, splits.
- ``[modeling]`` / ``[inference]`` — same semantics as PDP when those stages run.
"""

from __future__ import annotations

from edvise.configs.pdp import (
    CheckpointBaseConfig,
    CheckpointFirstAtNumCreditsEarnedConfig,
    CheckpointFirstConfig,
    CheckpointFirstWithinCohortConfig,
    CheckpointNthConfig,
    DatasetsConfig,
    EvaluationConfig,
    FeatureSelectionConfig,
    FeaturesConfig,
    InferenceConfig,
    ModelConfig,
    ModelingConfig,
    PDPProjectConfig,
    PreprocessingConfig,
    SelectionConfig,
    TargetBaseConfig,
    TargetCreditsEarnedConfig,
    TargetGraduationConfig,
    TargetRetentionConfig,
    TrainingConfig,
)


class ESProjectConfig(PDPProjectConfig):
    """
    Configuration schema for pipeline jobs that use the Edvise Schema (ES).

    Inherits all fields and validators from :class:`PDPProjectConfig`. Use the
    same ``config.toml`` structure as a PDP project; set ``--inst_schema es``
    (or pass this class to :func:`~edvise.dataio.read.read_config`) so entrypoints
    load ES-standardized data and ES feature column maps.

    ``student_id_col`` is the person-level key on silver cohort/course tables;
    for raw Edvise uploads that column is :obj:`learner_id`, so the default here
    differs from PDP.
    """

    student_id_col: str = "learner_id"


__all__ = [
    "CheckpointBaseConfig",
    "CheckpointFirstAtNumCreditsEarnedConfig",
    "CheckpointFirstConfig",
    "CheckpointFirstWithinCohortConfig",
    "CheckpointNthConfig",
    "DatasetsConfig",
    "ESProjectConfig",
    "EvaluationConfig",
    "FeatureSelectionConfig",
    "FeaturesConfig",
    "InferenceConfig",
    "ModelConfig",
    "ModelingConfig",
    "PDPProjectConfig",
    "PreprocessingConfig",
    "SelectionConfig",
    "TargetBaseConfig",
    "TargetCreditsEarnedConfig",
    "TargetGraduationConfig",
    "TargetRetentionConfig",
    "TrainingConfig",
]
