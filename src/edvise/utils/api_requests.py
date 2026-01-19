import requests
import typing as t
from urllib.parse import quote


def get_access_tokens(api_key: str) -> t.Any:
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

    access_token_url = "https://staging-sst.datakind.org/api/v1/token-from-api-key"
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
    access_token = get_access_tokens(api_key=api_key)

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

    create_model_endpoint_url = (
        f"https://staging-sst.datakind.org/api/v1/{inst_id}/models/"
    )
    resp = session.post(
        create_model_endpoint_url, json=payload, headers=custom_model_headers
    )
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_custom_institution_exist(inst_id: str, api_key: str) -> t.Any:
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

    access_token = get_access_tokens(api_key=api_key)

    # Verify institution exists
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    read_inst_endpoint_url = (
        f"https://staging-sst.datakind.org/api/v1/institutions/{inst_id}"
    )
    resp = session.get(read_inst_endpoint_url, headers=custom_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_custom_model_exist(inst_id: str, model_name: str, api_key: str) -> t.Any:
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key)

    # Verify institution exists
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    read_model_endpoint_url = f"https://staging-sst.datakind.org/api/v1/institutions/{inst_id}/models/{model_name}"
    resp = session.get(read_model_endpoint_url, headers=custom_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def get_institution_id_by_name(institution_name: str, api_key: str) -> t.Any:
    """
    Retrieve institution ID by institution name from the API.
    
    Makes a GET request to the API endpoint that looks up an institution
    by its human-readable name and returns the corresponding institution ID.
    
    Args:
        institution_name: The name of the institution to look up
        api_key: API key required for authentication
    
    Returns:
        Institution ID (str) if found, or error dictionary if validation fails
    
    Raises:
        requests.HTTPError: If the API request fails (e.g., 404 if institution not found)
        KeyError: If the response doesn't contain 'inst_id'
        ValueError: If the response is not valid JSON
    """
    if not institution_name or not isinstance(institution_name, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "institution_name must be a non-empty string",
        }
    if not api_key or not isinstance(api_key, str):
        return {
            "ok": False,
            "stage": "validation",
            "error": "api_key must be a non-empty string",
        }

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key)

    # Look up institution by name
    # URL-encode the institution name to handle spaces, special chars, unicode, etc.
    encoded_name = quote(institution_name.strip(), safe="")
    institution_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    institution_endpoint_url = (
        f"https://staging-sst.datakind.org/api/v1/institutions/name/{encoded_name}"
    )
    resp = session.get(institution_endpoint_url, headers=institution_headers, timeout=15)
    resp.raise_for_status()

    try:
        institution_data = resp.json()
    except ValueError as e:
        raise ValueError(
            f"Institution endpoint returned non-JSON: {resp.text}"
        ) from e

    inst_id = (
        institution_data.get("inst_id")
        if isinstance(institution_data, dict)
        else None
    )
    if not inst_id:
        raise KeyError(
            f"No 'inst_id' in institution response: {institution_data}"
        )

    return inst_id


def log_custom_job(
    inst_id: str, job_run_id: str, model_name: str, api_key: str
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
    access_token = get_access_tokens(api_key=api_key)

    # Log custom jobs in JobTable
    custom_job_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    custom_job_endpoint_url = f"https://staging-sst.datakind.org/api/v1/{inst_id}/add-custom-school-job/{job_run_id}?model_name={model_name}"
    resp = session.post(custom_job_endpoint_url, headers=custom_job_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text
