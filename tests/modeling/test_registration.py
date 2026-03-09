import pytest
from unittest.mock import Mock
import mlflow
from edvise.modeling.registration import (
    register_mlflow_model,
    get_model_name,
    get_model_name_from_config,
    pdp_get_model_name,
    normalize_degree,
    format_enrollment_intensity_time_limits,
)


@pytest.fixture
def mock_client():
    client = Mock(spec=mlflow.tracking.MlflowClient)
    return client


def test_registers_new_model_and_sets_tag(mock_client):
    run_id = "abc123"
    model_name = "my_model"
    institution_id = "inst"
    catalog = "main"
    version = Mock()
    version.version = 5

    # Simulate: model doesn't exist → create it
    mock_client.create_registered_model.side_effect = None
    mock_client.get_run.return_value.data.tags = {}

    mock_client.create_model_version.return_value = version

    register_mlflow_model(
        model_name=model_name,
        institution_id=institution_id,
        run_id=run_id,
        catalog=catalog,
        mlflow_client=mock_client,
    )

    model_path = f"{catalog}.{institution_id}_gold.{model_name}"
    model_uri = f"runs:/{run_id}/model"

    mock_client.create_registered_model.assert_called_once_with(name=model_path)
    mock_client.get_run.assert_called_once_with(run_id)
    mock_client.create_model_version.assert_called_once_with(
        name=model_path,
        source=model_uri,
        run_id=run_id,
    )
    mock_client.set_tag.assert_called_once_with(run_id, "model_registered", "true")
    mock_client.set_registered_model_alias.assert_called_once_with(
        model_path, "Staging", version.version
    )


def test_skips_if_tag_indicates_already_registered(mock_client):
    mock_client.get_run.return_value.data.tags = {"model_registered": "true"}

    register_mlflow_model(
        model_name="my_model",
        institution_id="inst",
        run_id="abc123",
        catalog="main",
        mlflow_client=mock_client,
    )

    mock_client.create_model_version.assert_not_called()
    mock_client.set_tag.assert_not_called()
    mock_client.set_registered_model_alias.assert_not_called()


def test_handles_existing_registered_model_gracefully(mock_client):
    mock_client.create_registered_model.side_effect = mlflow.exceptions.MlflowException(
        "RESOURCE_ALREADY_EXISTS"
    )
    mock_client.get_run.return_value.data.tags = {}
    mock_client.create_model_version.return_value.version = 1

    register_mlflow_model(
        model_name="m",
        institution_id="inst",
        run_id="run1",
        catalog="main",
        mlflow_client=mock_client,
    )

    mock_client.create_model_version.assert_called()


def test_raises_if_tag_check_fails(mock_client):
    mock_client.get_run.side_effect = mlflow.exceptions.MlflowException("Bad request")

    with pytest.raises(mlflow.exceptions.MlflowException):
        register_mlflow_model(
            model_name="m",
            institution_id="inst",
            run_id="bad_run",
            catalog="main",
            mlflow_client=mock_client,
        )


def test_skips_setting_alias_if_none(mock_client):
    mock_client.get_run.return_value.data.tags = {}
    mock_client.create_model_version.return_value.version = 1

    register_mlflow_model(
        model_name="m",
        institution_id="inst",
        run_id="run1",
        catalog="main",
        model_alias=None,
        mlflow_client=mock_client,
    )

    mock_client.set_registered_model_alias.assert_not_called()


# Tests for normalize_degree helper function
class TestNormalizeDegree:
    """Tests for normalize_degree function used in get_model_name"""

    def test_removes_degree_suffix_uppercase(self):
        result = normalize_degree("ASSOCIATE'S DEGREE")
        assert result == "Associates"

    def test_removes_degree_suffix_lowercase(self):
        result = normalize_degree("bachelor's degree")
        assert result == "Bachelors"

    def test_removes_degree_suffix_mixed_case(self):
        result = normalize_degree("Master's Degree")
        assert result == "Masters"

    def test_removes_degree_with_extra_spaces(self):
        result = normalize_degree("DOCTORAL  DEGREE  ")
        assert result == "Doctoral"

    def test_handles_text_without_degree_suffix(self):
        result = normalize_degree("CERTIFICATE")
        assert result == "Certificate"

    def test_preserves_apostrophe_in_title_case(self):
        result = normalize_degree("BACHELOR'S DEGREE")
        assert result == "Bachelors"


# Tests for format_enrollment_intensity_time_limits helper function
class TestFormatEnrollmentIntensityTimeLimits:
    """Tests for format_enrollment_intensity_time_limits function used in get_model_name"""

    def test_full_time_only_years(self):
        intensity_time_limits = {"FULL-TIME": [3.0, "year"]}
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "3Y FT"

    def test_part_time_only_years(self):
        intensity_time_limits = {"PART-TIME": [6.0, "year"]}
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "6Y PT"

    def test_both_full_time_and_part_time_years(self):
        """Test typical config template format: 3 years FT, 6 years PT"""
        intensity_time_limits = {
            "FULL-TIME": [3.0, "year"],
            "PART-TIME": [6.0, "year"],
        }
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "3Y FT, 6Y PT"

    def test_decimal_duration(self):
        intensity_time_limits = {"FULL-TIME": [2.5, "year"]}
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "2.5Y FT"

    def test_terms_unit(self):
        """Test term-based time limits (alternative to years)"""
        intensity_time_limits = {
            "FULL-TIME": [4.0, "term"],
            "PART-TIME": [8.0, "term"],
        }
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "4T FT, 8T PT"

    def test_respects_order_full_time_first(self):
        """FULL-TIME should always appear first regardless of dict order"""
        intensity_time_limits = {
            "PART-TIME": [6.0, "year"],
            "FULL-TIME": [3.0, "year"],
        }
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "3Y FT, 6Y PT"

    def test_integer_duration_no_decimal_point(self):
        """3.0 should display as '3' not '3.0'"""
        intensity_time_limits = {"FULL-TIME": [3.0, "year"]}
        result = format_enrollment_intensity_time_limits(
            intensity_time_limits=intensity_time_limits, style="compact"
        )
        assert result == "3Y FT"


# Tests for get_model_name function (simple version)
class TestGetModelName:
    """Tests for the simple get_model_name function that takes string parameters."""

    def test_basic_model_name_format(self):
        """Test basic format: institution_id_target_checkpoint"""
        result = get_model_name(
            institution_id="test_inst",
            target="retention_into_year_2",
            checkpoint="first_term",
        )
        assert result == "test_inst_retention_into_year_2_first_term"

    def test_model_name_with_extra_info(self):
        """Test format with extra_info: institution_id_target_checkpoint_extra_info"""
        result = get_model_name(
            institution_id="test_inst",
            target="graduation_in_3y",
            checkpoint="checkpoint_2",
            extra_info="pilot",
        )
        assert result == "test_inst_graduation_in_3y_checkpoint_2_pilot"

    def test_model_name_without_extra_info(self):
        """Test that extra_info is optional (None omits suffix)"""
        result = get_model_name(
            institution_id="my_school",
            target="custom_target",
            checkpoint="custom_checkpoint",
            extra_info=None,
        )
        assert result == "my_school_custom_target_custom_checkpoint"

    def test_model_name_without_institution_id(self):
        """Test format when institution_id is omitted (target only or target + checkpoint)"""
        result = get_model_name(
            target="retention_into_year_2_associates",
            checkpoint="",
        )
        assert result == "retention_into_year_2_associates"

    def test_model_name_target_and_checkpoint_only(self):
        """Test format with target and checkpoint, no institution_id"""
        result = get_model_name(
            target="graduation_in_3y_ft_6y_pt",
            checkpoint="checkpoint_2_core_terms",
        )
        assert result == "graduation_in_3y_ft_6y_pt_checkpoint_2_core_terms"

    def test_model_name_target_only_no_checkpoint(self):
        """Test format with only target (empty checkpoint)"""
        result = get_model_name(
            target="retention_into_year_2_all_degrees",
            checkpoint="",
        )
        assert result == "retention_into_year_2_all_degrees"

    def test_model_name_produces_lowercase_underscore_separated(self):
        """Test that output is lowercase with underscores (Unity Catalog compatible)"""
        result = get_model_name(
            institution_id="my_inst",
            target="credits_30_in_3y_ft_6y_pt",
            checkpoint="checkpoint_2_core_terms",
        )
        assert result == "my_inst_credits_30_in_3y_ft_6y_pt_checkpoint_2_core_terms"
        assert " " not in result
        assert result == result.lower()


# Tests for get_model_name_from_config (try PDP first, fallback to custom)
class TestGetModelNameFromConfig:
    """Tests for get_model_name_from_config: tries PDP logic first, falls back to target.name/checkpoint.name."""

    def test_pdp_graduation_uses_pdp_logic(self):
        """PDP-style config should use pdp_get_model_name and produce graduation model name."""
        preprocessing = type("Preprocessing", (), {})()
        preprocessing.target = {
            "type_": "graduation",
            "intensity_time_limits": {
                "FULL-TIME": [3.0, "year"],
                "PART-TIME": [6.0, "year"],
            },
        }
        preprocessing.checkpoint = {
            "type_": "nth",
            "n": 1,
            "exclude_non_core_terms": True,
        }
        preprocessing.selection = type("Selection", (), {"student_criteria": {}})()

        result = get_model_name_from_config(
            preprocessing=preprocessing,
            institution_id="test_inst",
        )
        assert result == "graduation_in_3y_ft_6y_pt_checkpoint_2_core_terms"

    def test_custom_fallback_uses_target_checkpoint_name(self):
        """Custom-style config (no type_ or unsupported) should fall back to target.name, checkpoint.name."""
        preprocessing = type("Preprocessing", (), {})()
        preprocessing.target = type("Target", (), {"name": "my_custom_target"})()
        preprocessing.checkpoint = type("Checkpoint", (), {"name": "my_checkpoint"})()
        preprocessing.selection = None

        result = get_model_name_from_config(
            preprocessing=preprocessing,
            institution_id="my_school",
        )
        assert result == "my_school_my_custom_target_my_checkpoint"

    def test_fallback_when_pdp_raises_unsupported_target_type(self):
        """When target type is unsupported, fall back to .name."""
        preprocessing = type("Preprocessing", (), {})()
        preprocessing.target = {
            "type_": "unsupported_type",
            "name": "fallback_target",
        }
        preprocessing.checkpoint = {"name": "fallback_checkpoint"}
        preprocessing.selection = type("Selection", (), {"student_criteria": {}})()

        result = get_model_name_from_config(
            preprocessing=preprocessing,
            institution_id="inst",
        )
        assert result == "inst_fallback_target_fallback_checkpoint"


# Tests for pdp_get_model_name function
class TestPDPGetModelName:
    """
    Comprehensive tests for pdp_get_model_name based on actual config templates.

    Config templates tested:
    - config-RETENTION_TEMPLATE.toml
    - config-GRADUATION_TEMPLATE.toml
    - config-CREDITS_EARNED_TEMPLATE.toml
    """

    # === RETENTION TARGET TESTS (based on config-RETENTION_TEMPLATE.toml) ===

    def test_retention_with_associate_degree(self):
        """Retention template default: Associate's degree"""
        target = {
            "type_": "retention",
            "max_academic_year": "2024",
        }
        checkpoint = {
            "type_": "first_within_cohort",
            "exclude_non_core_terms": True,
            "exclude_pre_cohort_terms": True,
        }
        student_criteria = {
            "enrollment_type": "FIRST-TIME",
            "credential_type_sought_year_1": "ASSOCIATE'S DEGREE",
            "cohort_term": ["FALL", "SPRING"],
        }

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "retention_into_year_2_associates"

    def test_retention_with_bachelor_degree(self):
        """Retention with Bachelor's degree variant"""
        target = {
            "type_": "retention",
            "max_academic_year": "2024",
        }
        checkpoint = {
            "type_": "first_within_cohort",
        }
        student_criteria = {
            "credential_type_sought_year_1": "BACHELOR'S DEGREE",
        }

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "retention_into_year_2_bachelors"

    def test_retention_without_credential_type(self):
        """Retention with no credential type specified -> 'All Degrees'"""
        target = {
            "type_": "retention",
            "max_academic_year": "2024",
        }
        checkpoint = {
            "type_": "first_within_cohort",
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "retention_into_year_2_all_degrees"

    def test_retention_with_extra_info(self):
        """Retention with optional extra_info parameter"""
        target = {
            "type_": "retention",
            "max_academic_year": "2024",
        }
        checkpoint = {
            "type_": "first_within_cohort",
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
            extra_info="pilot",
        )
        assert result == "retention_into_year_2_all_degrees_pilot"

    # === GRADUATION TARGET TESTS (based on config-GRADUATION_TEMPLATE.toml) ===

    def test_graduation_nth_checkpoint_core_terms(self):
        """
        Graduation template Option 1: nth checkpoint with core terms only
        Matches config-GRADUATION_TEMPLATE.toml
        """
        target = {
            "type_": "graduation",
            "intensity_time_limits": {
                "FULL-TIME": [3.0, "year"],
                "PART-TIME": [6.0, "year"],
            },
            "years_to_degree_col": "first_year_to_associates_at_cohort_inst",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 1,
            "term_is_core_col": "term_is_core",
            "exclude_non_core_terms": True,
            "exclude_pre_cohort_terms": True,
        }
        student_criteria = {
            "enrollment_type": "FIRST-TIME",
            "credential_type_sought_year_1": "ASSOCIATE'S DEGREE",
        }

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_3y_ft_6y_pt_checkpoint_2_core_terms"

    def test_graduation_nth_checkpoint_total_terms(self):
        """Graduation with nth checkpoint counting all terms (not just core)"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"FULL-TIME": [4.0, "year"]},
            "years_to_degree_col": "first_year_to_bachelors_at_cohort_inst",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 2,
            "exclude_non_core_terms": False,
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_4y_ft_checkpoint_3_total_terms"

    def test_graduation_first_at_num_credits_checkpoint(self):
        """
        Graduation template Option 2: credits-based checkpoint
        Matches config-GRADUATION_TEMPLATE.toml Option 2
        """
        target = {
            "type_": "graduation",
            "intensity_time_limits": {
                "FULL-TIME": [3.0, "year"],
                "PART-TIME": [6.0, "year"],
            },
            "years_to_degree_col": "first_year_to_associates_at_cohort_inst",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "first_at_num_credits_earned",
            "min_num_credits": 30.0,
            "num_credits_col": "cumsum_num_credits_earned",
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_3y_ft_6y_pt_checkpoint_30_credits"

    def test_graduation_first_student_terms_checkpoint(self):
        """Graduation with first term checkpoint"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"FULL-TIME": [2.0, "year"]},
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_2y_ft_checkpoint_first_term"

    def test_graduation_first_student_terms_within_cohort_checkpoint(self):
        """Graduation with first cohort term checkpoint"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"PART-TIME": [6.0, "year"]},
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms_within_cohort"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_6y_pt_checkpoint_first_cohort_term"

    def test_graduation_with_term_based_time_limits(self):
        """Graduation using terms instead of years"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {
                "FULL-TIME": [6.0, "term"],
                "PART-TIME": [12.0, "term"],
            },
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "graduation_in_6t_ft_12t_pt_checkpoint_first_term"

    # === CREDITS_EARNED TARGET TESTS (based on config-CREDITS_EARNED_TEMPLATE.toml) ===

    def test_credits_earned_template_default_config(self):
        """
        Credits earned template default configuration
        Matches config-CREDITS_EARNED_TEMPLATE.toml exactly
        """
        target = {
            "type_": "credits_earned",
            "min_num_credits": 60.0,
            "intensity_time_limits": {
                "FULL-TIME": [3.0, "year"],
                "PART-TIME": [6.0, "year"],
            },
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "first_at_num_credits_earned",
            "min_num_credits": 30.0,
            "num_credits_col": "cumsum_num_credits_earned",
        }
        student_criteria = {
            "enrollment_type": "FIRST-TIME",
            "credential_type_sought_year_1": "ASSOCIATE'S DEGREE",
        }

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "60_credits_in_3y_ft_6y_pt_checkpoint_30_credits"

    def test_credits_earned_with_nth_checkpoint_core_terms(self):
        """Credits earned with nth core terms checkpoint"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 60,
            "intensity_time_limits": {"FULL-TIME": [2.0, "year"]},
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 1,
            "exclude_non_core_terms": True,
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "60_credits_in_2y_ft_checkpoint_2_core_terms"

    def test_credits_earned_with_nth_checkpoint_total_terms(self):
        """Credits earned with nth total terms checkpoint"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 30,
            "intensity_time_limits": {
                "FULL-TIME": [1.0, "year"],
                "PART-TIME": [2.0, "year"],
            },
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 0,
            "exclude_non_core_terms": False,
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "30_credits_in_1y_ft_2y_pt_checkpoint_1_total_terms"

    def test_credits_earned_with_first_student_terms_checkpoint(self):
        """Credits earned with first term checkpoint"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 15,
            "intensity_time_limits": {"FULL-TIME": [1.0, "term"]},
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "15_credits_in_1t_ft_checkpoint_first_term"

    def test_credits_earned_with_first_student_terms_within_cohort_checkpoint(self):
        """Credits earned with first cohort term checkpoint"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 45,
            "intensity_time_limits": {"PART-TIME": [3.0, "year"]},
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms_within_cohort"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "45_credits_in_3y_pt_checkpoint_first_cohort_term"

    def test_credits_earned_with_term_based_time_limits(self):
        """Credits earned using terms instead of years"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 24,
            "intensity_time_limits": {
                "FULL-TIME": [4.0, "term"],
                "PART-TIME": [8.0, "term"],
            },
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        assert result == "24_credits_in_4t_ft_8t_pt_checkpoint_first_term"

    # === EDGE CASES AND ADDITIONAL SCENARIOS ===

    def test_decimal_credits_as_integer(self):
        """Test that 60.0 displays as-is per Python's str()"""
        target = {
            "type_": "credits_earned",
            "min_num_credits": 60.0,
            "intensity_time_limits": {"FULL-TIME": [3.0, "year"]},
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        # Python's str(60.0) = "60.0", str(3.0) when int(3.0) == 3.0 displays as "3"
        assert result == "60_credits_in_3y_ft_checkpoint_first_term"

    def test_zero_n_checkpoint(self):
        """Test n=0 which results in checkpoint at position 1"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"FULL-TIME": [2.0, "year"]},
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 0,
            "exclude_non_core_terms": True,
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        # n=0 means first term, so displays as "1 Core Terms"
        assert result == "graduation_in_2y_ft_checkpoint_1_core_terms"

    def test_large_n_checkpoint(self):
        """Test large n value (e.g., for long programs)"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"FULL-TIME": [6.0, "year"]},
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {
            "type_": "nth",
            "n": 11,
            "exclude_non_core_terms": False,
        }
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
        )
        # n=11 means 12th term
        assert result == "graduation_in_6y_ft_checkpoint_12_total_terms"

    def test_graduation_with_extra_info(self):
        """Test extra_info parameter with graduation"""
        target = {
            "type_": "graduation",
            "intensity_time_limits": {"FULL-TIME": [4.0, "year"]},
            "years_to_degree_col": "first_year_to_degree",
            "num_terms_in_year": 4,
            "max_term_rank": "infer",
        }
        checkpoint = {"type_": "first_student_terms"}
        student_criteria = {}

        result = pdp_get_model_name(
            target=target,
            checkpoint=checkpoint,
            student_criteria=student_criteria,
            extra_info="experimental_v2",
        )
        assert result == "graduation_in_4y_ft_checkpoint_first_term_experimental_v2"
