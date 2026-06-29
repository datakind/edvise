from edvise.scripts.pdp_synthetic_cleanup import (
    SILVER_VOLUME_PRESERVE_DIRS,
    silver_volume_child_basename,
)


def test_silver_volume_child_basename() -> None:
    assert (
        silver_volume_child_basename(
            "/Volumes/dev_sst_02/synthetic_integration_es_silver/silver_volume/genai_mapping"
        )
        == "genai_mapping"
    )
    assert (
        silver_volume_child_basename(
            "/Volumes/dev_sst_02/synthetic_integration_es_silver/silver_volume/abc-uuid/"
        )
        == "abc-uuid"
    )


def test_genai_mapping_is_preserved_during_silver_cleanup() -> None:
    assert "genai_mapping" in SILVER_VOLUME_PRESERVE_DIRS
