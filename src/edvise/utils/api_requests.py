import requests
import typing as t

def get_access_tokens(api_key: str):
    if not api_key or not isinstance(api_key, str):
        return {"ok": False, "stage": "validation", "error": "api_key must be a non-empty string"}
    
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

    access_token = token_json.get("access_token") if isinstance(token_json, dict) else None
    if not access_token:
        raise KeyError(f"No 'access_token' in token response: {token_json}")

    return access_token
    
def create_custom_model(
        inst_id: str, model_name: str, api_key: str, valid: bool,
) -> t.Any:
    "Retrieve access token and log custom job ids on the GCP Cloud SQL JobTable"

    if not inst_id or not isinstance(inst_id, str):
        return {"ok": False, "stage": "validation", "error": "inst_id must be a non-empty string"}
    if not model_name or not isinstance(model_name, str):
        return {"ok": False, "stage": "validation", "error": "model_name must be a non-empty string"}
    if not api_key or not isinstance(api_key, str):
        return {"ok": False, "stage": "validation", "error": "api_key must be a non-empty string"}
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
                {"schema_type": "STUDENT",  "optional": False, "multiple_allowed": False},
                {"schema_type": "SEMESTER", "optional": True,  "multiple_allowed": False},
                {"schema_type": "COURSE",   "optional": False, "multiple_allowed": False},
            ]
        ]
    }

    create_model_endpoint_url = f"https://staging-sst.datakind.org/api/v1/{inst_id}/models/"
    resp = session.post(create_model_endpoint_url, json=payload, headers=custom_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text

def validate_custom_institution_exist(
        inst_id: str, api_key: str
) -> t.Any:
    if not inst_id or not isinstance(inst_id, str):
        return {"ok": False, "stage": "validation", "error": "inst_id must be a non-empty string"}
    if not api_key or not isinstance(api_key, str):
        return {"ok": False, "stage": "validation", "error": "api_key must be a non-empty string"}

    session = requests.Session()

    access_token = get_access_tokens(api_key=api_key)

    # Verify institution exists
    custom_model_headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    read_inst_endpoint_url = f"https://staging-sst.datakind.org/api/v1/institutions/{inst_id}"
    resp = session.get(read_inst_endpoint_url, headers=custom_model_headers)
    resp.raise_for_status()
    
    try:
        return resp.json()
    except ValueError:
        return resp.text

def validate_custom_model_exist(
        inst_id: str, model_name: str, api_key: str
) -> t.Any:
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

def log_custom_job(
    inst_id: str, job_run_id: str, model_name: str, api_key: str
) -> t.Any:
    "Retrieve access token and log custom job ids on the GCP Cloud SQL JobTable"
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
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