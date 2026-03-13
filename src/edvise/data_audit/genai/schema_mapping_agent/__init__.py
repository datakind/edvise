"""SchemaMappingAgent: LLM-driven mapping, transformation, and validation."""

from edvise.data_audit.genai.schema_mapping_agent.join_executor import (
    JoinGraph,
    JoinResolver,
    JoinStep,
    execute_join_graph,
)
from edvise.data_audit.genai.schema_mapping_agent.mapping_schemas import (
    EntityType,
    FieldMappingManifest,
    FieldMappingRecord,
    ReviewStatus,
    TransformationMap,
)
from edvise.data_audit.genai.schema_mapping_agent.preprocessing import (
    build_schema_contract_from_config,
)
from edvise.data_audit.genai.schema_mapping_agent.transformation_executor import (
    ExecutionError,
    ExecutionGapError,
    ExecutionResult,
    execute_transformation_map,
)

__all__ = [
    # Join executor
    "JoinGraph",
    "JoinResolver",
    "JoinStep",
    "execute_join_graph",
    # Mapping schemas
    "EntityType",
    "FieldMappingManifest",
    "FieldMappingRecord",
    "ReviewStatus",
    "TransformationMap",
    # Preprocessing
    "build_schema_contract_from_config",
    # Transformation executor
    "ExecutionError",
    "ExecutionGapError",
    "ExecutionResult",
    "execute_transformation_map",
]
