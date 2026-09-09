"""
Execute

Load approved artifacts -> execute transformation map -> Pandera -> write outputs
"""

from __future__ import annotations

import json
import logging
from typing import Any

from edvise.genai.mapping.shared.active_promotion import (
    update_genai_active_registry_execute,
)
from edvise.genai.mapping.schema_mapping_agent.grain_resolution import (
    execute_transformation_map_for_sma_execute_mode,
)

from .helpers import (
    _load_cleaned_dataframes,
    _load_enriched_contract,
    _run_pandera_validation,
    _write_output_data,
)
from .paths import SMAPaths

LOGGER = logging.getLogger("edvise_sma")


def run_execute(
    institution_id: str,
    paths: SMAPaths,
    spark_session: Any,
    *,
    execute_run_id: str,
) -> None:
    from edvise.genai.mapping.schema_mapping_agent.manifest.schemas import (
        FieldMappingManifest,
        MappingManifestEnvelope,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.dedupe_plans import (
        dedupe_transformation_plans_in_wrapper,
    )
    from edvise.genai.mapping.schema_mapping_agent.transformation.schemas import (
        TransformationMap,
    )
    from edvise.data_audit.schemas.raw_edvise_student import RawEdviseStudentDataSchema
    from edvise.data_audit.schemas.raw_edvise_course import RawEdviseCourseDataSchema

    LOGGER.info(
        "[execute] Loading approved artifacts from active/ for %s", institution_id
    )

    # Validate active artifacts exist
    for p in (
        paths.active_manifest_map,
        paths.active_transformation_map,
        paths.active_enriched_schema_contract,
    ):
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing active artifact: {p}. "
                "Has this institution been onboarded and activated?"
            )

    # Load approved artifacts
    manifest_data = json.loads(paths.active_manifest_map.read_text())
    transformation_data = json.loads(paths.active_transformation_map.read_text())
    dedupe_transformation_plans_in_wrapper(transformation_data, log=LOGGER)
    enriched_contract = _load_enriched_contract(paths.active_enriched_schema_contract)

    # Load cleaned dataframes — written by edvise_ia execute mode in this run
    dataframes = _load_cleaned_dataframes(paths.ia_cleaned_datasets, enriched_contract)

    # Execute transformation maps
    LOGGER.info("[execute] Executing transformation maps")
    institution_id_from_tm = transformation_data.get("institution_id", institution_id)

    cohort_map_data = {
        **transformation_data["transformation_maps"]["cohort"],
        "institution_id": institution_id_from_tm,
    }
    course_map_data = {
        **transformation_data["transformation_maps"]["course"],
        "institution_id": institution_id_from_tm,
    }

    envelope = MappingManifestEnvelope.model_validate(manifest_data)
    cohort_manifest = FieldMappingManifest.model_validate(
        manifest_data["manifests"]["cohort"]
    )
    course_manifest = FieldMappingManifest.model_validate(
        manifest_data["manifests"]["course"]
    )
    cohort_map = TransformationMap.model_validate(cohort_map_data)
    course_map = TransformationMap.model_validate(course_map_data)

    cohort_result = execute_transformation_map_for_sma_execute_mode(
        transformation_map=cohort_map,
        manifest=cohort_manifest,
        dataframes=dataframes,
        schema=RawEdviseStudentDataSchema,
        spark_session=spark_session,
        institution_id=institution_id_from_tm,
        enriched_contract=enriched_contract,
        manifest_map_path=paths.active_manifest_map,
        grain_hitl_path=paths.run_root / "cohort_sma_grain_hitl.json",
        active_grain_resolution_root=paths.active_root,
        hook_modules_root=paths.active_root,
    )
    course_result = execute_transformation_map_for_sma_execute_mode(
        transformation_map=course_map,
        manifest=course_manifest,
        dataframes=dataframes,
        schema=RawEdviseCourseDataSchema,
        spark_session=spark_session,
        institution_id=institution_id_from_tm,
        enriched_contract=enriched_contract,
        manifest_map_path=paths.active_manifest_map,
        grain_hitl_path=paths.run_root / "course_sma_grain_hitl.json",
        active_grain_resolution_root=paths.active_root,
        hook_modules_root=paths.active_root,
    )

    # Pandera validation (report only)
    LOGGER.info("[execute] Pandera validation")
    _run_pandera_validation(
        cohort_result, course_result, report_path=paths.pandera_validation_errors
    )

    # Write output data
    _write_output_data(paths.output_data, cohort_result, course_result)
    LOGGER.info("[execute] Updating genai_active_registry execute pointer")
    update_genai_active_registry_execute(
        paths.active_root,
        execute_run_id=execute_run_id,
    )
    LOGGER.info("[execute] Complete. Exiting.")
