# Standard library imports
import logging
import re
import typing as t
from typing import cast
from urllib.parse import quote

# Third-party imports
import requests

LOGGER = logging.getLogger(__name__)


def get_base_url(DB_workspace: str) -> str:
    """
    Map DB_workspace to the appropriate API base URL.

    Args:
        DB_workspace: The Databricks workspace identifier (e.g., 'dev_sst_02', 'staging_sst_01')

    Returns:
        Base URL for the API

    Raises:
        ValueError: If DB_workspace is not recognized
    """
    workspace_lower = DB_workspace.lower().strip()

    # Map workspace to base URL based on pattern matching
    # Dev workspaces (dev_sst_02, etc.)
    if workspace_lower.startswith("dev"):
        return "https://dev-sst.datakind.org"
    # Staging workspaces (staging_sst_01, etc.)
    elif workspace_lower.startswith("staging"):
        return "https://staging-sst.datakind.org"
    else:
        raise ValueError(
            f"Unknown DB_workspace '{DB_workspace}'. Must start with 'dev' or 'staging'"
        )


def get_access_tokens(api_key: str, DB_workspace: str) -> t.Any:
    if not api_key or not isinstance(api_key, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "api_key must be a non-empty string",
        }

    session = requests.Session()

    # Retrieve API token
    token_headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    base_url = get_base_url(DB_workspace)
    access_token_url = f"{base_url}/api/v1/token-from-api-key"
    token_resp = session.post(access_token_url, headers=token_headers, timeout=15)
    token_resp.raise_for_status()

    try:
        token_json = token_resp.json()
    except ValueError as e:
        raise ValueError(f"Token endpoint returned non-JSON: {token_resp.text}") from e

    access_token = (
        token_json.get("access_token") if isinstance(token_json, dict) else None
    )
    if not access_token:
        raise KeyError(f"No 'access_token' in token response: {token_json}")

    return access_token


def create_custom_model(
    inst_id: str,
    model_name: str,
    api_key: str,
    valid: bool,
    DB_workspace: str,
) -> t.Any:
    "Retrieve access token and log custom job ids on the GCP Cloud SQL JobTable"

    if not inst_id or not isinstance(inst_id, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "inst_id must be a non-empty string",
        }
    if not model_name or not isinstance(model_name, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "model_name must be a non-empty string",
        }
    if not api_key or not isinstance(api_key, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "api_key must be a non-empty string",
        }
    if not isinstance(valid, bool):
        return {"ok": False, "stage": "validation", "error": "valid must be a boolean"}

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Log custom jobs in JobTable
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "name": model_name,
        "valid": valid,
        "schema_configs": [
            [
                {
                    "schema_type": "STUDENT",
                    "optional": False,
                    "multiple_allowed": False,
                },
                {
                    "schema_type": "SEMESTER",
                    "optional": True,
                    "multiple_allowed": False,
                },
                {"schema_type": "COURSE", "optional": False, "multiple_allowed": False},
            ]
        ],
    }

    base_url = get_base_url(DB_workspace)
    create_model_endpoint_url = f"{base_url}/api/v1/{inst_id}/models/"
    resp = session.post(
        create_model_endpoint_url, json=payload, headers=custom_model_headers
    )
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_custom_institution_exist(inst_id: str, api_key: str, DB_workspace: str) -> t.Any:
    if not inst_id or not isinstance(inst_id, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "inst_id must be a non-empty string",
        }
    if not api_key or not isinstance(api_key, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "api_key must be a non-empty string",
        }

    session = requests.Session()

    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Verify institution exists
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    base_url = get_base_url(DB_workspace)
    read_inst_endpoint_url = f"{base_url}/api/v1/institutions/{inst_id}"
    resp = session.get(read_inst_endpoint_url, headers=custom_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_custom_model_exist(inst_id: str, model_name: str, api_key: str, DB_workspace: str) -> t.Any:
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Verify institution exists
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    base_url = get_base_url(DB_workspace)
    read_model_endpoint_url = f"{base_url}/api/v1/institutions/{inst_id}/models/{model_name}"
    resp = session.get(read_model_endpoint_url, headers=custom_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


# Compiled regex patterns for reverse transformation (performance optimization)
_REVERSE_REPLACEMENTS = {
    "ctc": "community technical college",
    "cc": "community college",
    "st": "of science and technology",
    "uni": "university",
    "col": "college",
}

# Pre-compile regex patterns for word boundary matching
_COMPILED_REVERSE_PATTERNS = {
    abbrev: re.compile(r"\b" + re.escape(abbrev) + r"\b")
    for abbrev in _REVERSE_REPLACEMENTS.keys()
}


def _validate_databricks_name_format(databricks_name: str) -> None:
    """
    Validate that databricks name matches expected format.

    Args:
        databricks_name: Name to validate

    Raises:
        ValueError: If name is empty or contains invalid characters
    """
    if not isinstance(databricks_name, str) or not databricks_name.strip():
        raise ValueError("databricks_name must be a non-empty string")

    pattern = "^[a-z0-9_]*$"
    if not re.match(pattern, databricks_name):
        raise ValueError(
            f"Invalid databricks name format '{databricks_name}'. "
            "Must contain only lowercase letters, numbers, and underscores."
        )


def _reverse_abbreviation_replacements(name: str) -> str:
    """
    Reverse abbreviation replacements in the name.

    Handles the ambiguous "st" abbreviation:
    - If "st" appears as the first word, it's kept as "st" (abbreviation for Saint)
      and will be capitalized to "St" by title() case
    - Otherwise, "st" is treated as "of science and technology"

    Args:
        name: Name with underscores replaced by spaces

    Returns:
        Name with abbreviations expanded to full forms
    """
    # Split into words to handle "st" at the beginning specially
    words = name.split()

    # Keep "st" at the beginning as-is (will be capitalized to "St" by title() case)
    # Don't expand it to "saint" - preserve the abbreviation

    # Replace "st" in remaining positions with "of science and technology"
    for i in range(len(words)):
        if words[i] == "st" and i > 0:  # Only replace if not the first word
            words[i] = "of science and technology"

    # Rejoin and apply other abbreviation replacements
    name = " ".join(words)

    # Apply other abbreviation replacements (excluding "st" which we handled above)
    for abbrev, full_form in _REVERSE_REPLACEMENTS.items():
        if abbrev != "st":  # Skip "st" as we handled it above
            pattern = _COMPILED_REVERSE_PATTERNS[abbrev]
            name = pattern.sub(full_form, name)

    return name


def reverse_databricksify_inst_name(databricks_name: str) -> str:
    """
    Reverse the databricksify transformation to get back the original institution name.

    This function attempts to reverse the transformation done by databricksify_inst_name.
    Since the transformation is lossy (multiple original names can map to the same
    databricks name), this function produces the most likely original name.

    Args:
        databricks_name: The databricks-transformed institution name (e.g., "motlow_state_cc")
            Case inconsistencies are normalized (input is lowercased before processing).

    Returns:
        The reversed institution name with proper capitalization (e.g., "Motlow State Community College")

    Raises:
        ValueError: If the databricks name contains invalid characters
    """
    # Normalize to lowercase to handle case inconsistencies
    # (databricksify_inst_name always produces lowercase output)
    databricks_name = databricks_name.lower()
    _validate_databricks_name_format(databricks_name)

    # Step 1: Replace underscores with spaces
    name = databricks_name.replace("_", " ")

    # Step 2: Reverse the abbreviation replacements
    # The original replacements were done in this order (most specific first):
    # 1. "community technical college" → "ctc"
    # 2. "community college" → "cc"
    # 3. "of science and technology" → "st"
    # 4. "university" → "uni"
    # 5. "college" → "col"
    name = _reverse_abbreviation_replacements(name)

    # Step 3: Capitalize appropriately (title case)
    return name.title()


def _fetch_institution_by_name(normalized_name: str, access_token: str, DB_workspace: str) -> t.Any:
    """
    Fetch institution data from API by normalized name.

    Args:
        normalized_name: Institution name normalized to lowercase
        access_token: Bearer token for authentication
        DB_workspace: The Databricks workspace identifier

    Returns:
        JSON response data from API

    Raises:
        requests.HTTPError: If the API request fails
        ValueError: If the response is not valid JSON
    """
    session = requests.Session()
    institution_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    # URL-encode the institution name to handle spaces, special chars, unicode, etc.
    encoded_name = quote(normalized_name, safe="")
    base_url = get_base_url(DB_workspace)
    institution_endpoint_url = f"{base_url}/api/v1/institutions/name/{encoded_name}"
    resp = session.get(
        institution_endpoint_url, headers=institution_headers, timeout=15
    )
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError as e:
        LOGGER.error(
            f"Institution endpoint returned non-JSON for name '{normalized_name}': "
            f"{resp.text[:200]}"
        )
        raise ValueError(
            f"Institution endpoint returned non-JSON for name '{normalized_name}': "
            f"{resp.text[:200]}"
        ) from e


def _validate_and_transform_institution_name(
    institution_name: str, is_databricks_name: bool
) -> tuple[str, dict[str, t.Any] | None]:
    """
    Validate and optionally transform institution name.

    Args:
        institution_name: The institution name to validate/transform
        is_databricks_name: Whether the name is in databricks format

    Returns:
        Tuple of (transformed_name, error_dict). If error_dict is not None,
        validation failed and error_dict should be returned to caller.

    Raises:
        ValueError: If databricks name format is invalid (only if is_databricks_name=True)
    """
    # Validate institution_name
    if not isinstance(institution_name, str) or not institution_name.strip():
        return (
            "",
            {
                "ok": False,
                "stage": "validation",
                "error": "institution_name must be a non-empty string",
            },
        )

    # Validate and transform databricks name if needed
    if is_databricks_name:
        try:
            institution_name = reverse_databricksify_inst_name(institution_name.strip())
        except ValueError as e:
            LOGGER.error(
                f"Invalid databricks name format for institution lookup: "
                f"'{institution_name}'. Error: {str(e)}"
            )
            return (
                "",
                {
                    "ok": False,
                    "stage": "validation",
                    "error": f"Invalid databricks name format: {str(e)}",
                },
            )

    return (institution_name.strip(), None)


def _parse_institution_response(institution_data: t.Any, institution_name: str) -> str:
    """
    Parse institution ID from API response.

    Args:
        institution_data: JSON response from institution API
        institution_name: Original institution name for error context

    Returns:
        Institution ID string

    Raises:
        KeyError: If inst_id is missing from response
    """
    inst_id = (
        institution_data.get("inst_id") if isinstance(institution_data, dict) else None
    )
    if not inst_id:
        LOGGER.error(
            f"No 'inst_id' in institution response for name '{institution_name}': "
            f"{institution_data}"
        )
        raise KeyError(
            f"No 'inst_id' in institution response for name '{institution_name}': "
            f"{institution_data}"
        )
    # Type check: ensure inst_id is a string
    if not isinstance(inst_id, str):
        LOGGER.error(
            f"inst_id is not a string for name '{institution_name}': "
            f"type={type(inst_id)}, value={inst_id}"
        )
        raise TypeError(
            f"inst_id must be a string for name '{institution_name}', "
            f"got {type(inst_id).__name__}: {inst_id}"
        )
    # Type cast to satisfy mypy - we've verified it's a string above
    return cast(str, inst_id)


def get_institution_id_by_name(
    institution_name: str, api_key: str, DB_workspace: str, is_databricks_name: bool = False
) -> t.Any:
    """
    Retrieve institution ID by institution name from the API.

    Makes a GET request to the API endpoint that looks up an institution
    by its human-readable name and returns the corresponding institution ID.
    The API performs case-insensitive matching, so the name is normalized
    to lowercase before querying.

    Args:
        institution_name: The name of the institution to look up. If is_databricks_name
            is True, this should be the databricks-transformed name (e.g., "motlow_state_cc").
            Otherwise, it should be the original institution name. Case is normalized
            to lowercase before querying (the API endpoint is case-insensitive).
        api_key: API key required for authentication
        is_databricks_name: If True, institution_name will be reverse-transformed from
            databricks format to original format before querying the API

    Returns:
        Institution ID (str) if found, or error dictionary if validation fails

    Raises:
        requests.HTTPError: If the API request fails (e.g., 404 if institution not found)
        KeyError: If the response doesn't contain 'inst_id'
        ValueError: If the response is not valid JSON or if databricks name is invalid
    """
    # Validate api_key
    if not isinstance(api_key, str) or not api_key.strip():
        return {
            "ok": False,
            "stage": "validation",
            "error": "api_key must be a non-empty string",
        }

    # Validate and transform institution name
    institution_name, validation_error = _validate_and_transform_institution_name(
        institution_name, is_databricks_name
    )
    if validation_error is not None:
        return validation_error

    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Look up institution by name
    # Normalize to lowercase - the API endpoint performs case-insensitive matching
    # by comparing lowercase(name) == lowercase(input), so we normalize here for consistency
    normalized_name = institution_name.strip().lower()

    institution_data = _fetch_institution_by_name(normalized_name, access_token, DB_workspace)
    return _parse_institution_response(institution_data, normalized_name)


def log_custom_job(
    inst_id: str, job_run_id: str, model_name: str, api_key: str, DB_workspace: str
) -> t.Any:
    "Retrieve access token and log custom job ids on the GCP Cloud SQL JobTable"
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    if not isinstance(job_run_id, str) or not job_run_id.strip():
        raise ValueError("job_run_id must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Log custom jobs in JobTable
    custom_job_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    base_url = get_base_url(DB_workspace)
    custom_job_endpoint_url = f"{base_url}/api/v1/{inst_id}/add-custom-school-job/{job_run_id}?model_name={model_name}"
    resp = session.post(custom_job_endpoint_url, headers=custom_job_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text
