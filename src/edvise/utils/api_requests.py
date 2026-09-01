# Standard library imports
import logging
import os
import typing as t
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import quote, urljoin

from google.auth.transport.requests import Request
from google.oauth2 import id_token

# Third-party imports
import requests

LOGGER = logging.getLogger(__name__)


def fetch_iap_token(iap_audience: str) -> str:
    # Uses ADC (same auth mechanism as google.cloud.storage.Client()).
    return cast(str, id_token.fetch_id_token(Request(), iap_audience))


def get_iap_audience() -> str:
    aud = os.getenv("SST_IAP_AUDIENCE")
    if not aud:
        raise RuntimeError(
            "Missing SST_IAP_AUDIENCE env var (should be the IAP OAuth client ID / audience, "
            "like '...apps.googleusercontent.com')."
        )
    return aud


def iap_proxy_auth_headers() -> dict[str, str]:
    """
    IAP identity token for ``Proxy-Authorization``.

    Use this header for IAP so ``Authorization`` can carry the Edvise app JWT.
    Returns ``{}`` when ``SST_IAP_AUDIENCE`` is unset (local / non-IAP).
    """
    if not os.getenv("SST_IAP_AUDIENCE", "").strip():
        return {}
    return {
        "Proxy-Authorization": f"Bearer {fetch_iap_token(get_iap_audience())}",
    }


def edvise_request_headers(access_token: str, **extra: str) -> dict[str, str]:
    """App JWT on ``Authorization`` plus IAP on ``Proxy-Authorization`` when configured."""
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
        **iap_proxy_auth_headers(),
    }
    headers.update({k: v for k, v in extra.items() if v is not None})
    return headers


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


def get_access_tokens(api_key: str, DB_workspace: str) -> str:
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    api_key = api_key.strip()
    base_url = get_base_url(DB_workspace)
    url = f"{base_url}/api/v1/token-from-api-key"

    headers = {
        "X-API-KEY": api_key,  # consumed by your app
        "Accept": "application/json",
        # IAP on Proxy-Authorization so Authorization can hold the app JWT later
        **iap_proxy_auth_headers(),
    }

    resp = requests.post(url, headers=headers, timeout=15)

    # Helpful debug if IAP blocks you again
    if resp.status_code == 401 and "Invalid IAP credentials" in (resp.text or ""):
        raise RuntimeError(
            "Blocked by IAP (Invalid IAP credentials). "
            "Either SST_IAP_AUDIENCE is wrong, or the Databricks identity lacks IAP access."
        )

    resp.raise_for_status()
    token_json = resp.json()
    access_token = token_json.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise KeyError(f"No 'access_token' in token response: {token_json}")
    return access_token


def create_legacy_model(
    inst_id: str,
    model_name: str,
    api_key: str,
    valid: bool,
    DB_workspace: str,
) -> t.Any:
    "Retrieve access token and log legacy job ids on the GCP Cloud SQL JobTable"

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

    # Log legacy jobs in JobTable
    legacy_model_headers = edvise_request_headers(
        access_token, **{"Content-Type": "application/json"}
    )

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
        create_model_endpoint_url, json=payload, headers=legacy_model_headers
    )
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_legacy_institution_exist(
    inst_id: str, api_key: str, DB_workspace: str
) -> t.Any:
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
    legacy_model_headers = edvise_request_headers(
        access_token, **{"Content-Type": "application/json"}
    )

    base_url = get_base_url(DB_workspace)
    read_inst_endpoint_url = f"{base_url}/api/v1/institutions/{inst_id}"
    resp = session.get(read_inst_endpoint_url, headers=legacy_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def validate_legacy_model_exist(
    inst_id: str, model_name: str, api_key: str, DB_workspace: str
) -> t.Any:
    if not isinstance(inst_id, str) or not inst_id.strip():
        raise ValueError("inst_id must be a non-empty string")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")

    session = requests.Session()
    access_token = get_access_tokens(api_key=api_key, DB_workspace=DB_workspace)

    # Verify institution exists
    legacy_model_headers = edvise_request_headers(
        access_token, **{"Content-Type": "application/json"}
    )

    base_url = get_base_url(DB_workspace)
    read_model_endpoint_url = (
        f"{base_url}/api/v1/institutions/{inst_id}/models/{model_name}"
    )
    resp = session.get(read_model_endpoint_url, headers=legacy_model_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


def _fetch_institution_by_name(
    normalized_name: str, access_token: str, DB_workspace: str
) -> t.Any:
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
    institution_headers = edvise_request_headers(access_token)

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
            from edvise.utils.institution_naming import reverse_databricksify_inst_name

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
    institution_name: str,
    api_key: str,
    DB_workspace: str,
    is_databricks_name: bool = False,
) -> t.Any:
    """
    Retrieve institution ID by institution name from the API.

    Makes a GET request to the API endpoint that looks up an institution
    by its human-readable name and returns the corresponding institution ID.
    The API performs case-insensitive matching, so the name is normalized
    to lowercase before querying.

    Args:
        institution_name: The name of the institution to look up. If is_databricks_name
            is True, this should be the databricks-transformed name (e.g., "fixture_alpha_state_cc").
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

    institution_data = _fetch_institution_by_name(
        normalized_name, access_token, DB_workspace
    )
    return _parse_institution_response(institution_data, normalized_name)


def log_legacy_job(
    inst_id: str, job_run_id: str, model_name: str, api_key: str, DB_workspace: str
) -> t.Any:
    "Retrieve access token and log legacy job ids on the GCP Cloud SQL JobTable"
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

    # Log legacy jobs in JobTable
    legacy_job_headers = edvise_request_headers(
        access_token, **{"Content-Type": "application/json"}
    )

    base_url = get_base_url(DB_workspace)
    legacy_job_endpoint_url = f"{base_url}/api/v1/{inst_id}/add-custom-school-job/{job_run_id}?model_name={model_name}"
    resp = session.post(legacy_job_endpoint_url, headers=legacy_job_headers)
    resp.raise_for_status()

    try:
        return resp.json()
    except ValueError:
        return resp.text


# ---------------------------
# Edvise API Client (with caching and auto-refresh)
# ---------------------------


@dataclass
class EdviseAPIClient:
    """
    API client for Edvise API with bearer token management.

    Features:
    - Automatic bearer token fetching and refresh
    - Token caching within a session
    - Institution lookup caching
    - Automatic retry on 401 (unauthorized) errors

    Example:
        >>> client = EdviseAPIClient(
        ...     api_key="your-api-key",
        ...     base_url="https://staging-sst.datakind.org",
        ...     token_endpoint="/api/v1/token-from-api-key",
        ...     institution_lookup_path="/api/v1/institutions/pdp-id/{pdp_id}"
        ... )
        >>> institution = fetch_institution_by_pdp_id(client, "12345")
    """

    api_key: str
    base_url: str
    token_endpoint: str
    institution_lookup_path: str
    session: requests.Session = field(default_factory=requests.Session)
    bearer_token: str | None = None
    institution_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize API client configuration."""
        self.api_key = self.api_key.strip()
        if not self.api_key:
            raise ValueError("Empty Edvise API key.")

        self.base_url = self.base_url.rstrip("/")
        self.token_endpoint = self.token_endpoint.strip()
        self.institution_lookup_path = self.institution_lookup_path.strip()

        self.session.headers.update({"accept": "application/json"})


def _fetch_bearer_token_for_client(client: EdviseAPIClient) -> str:
    """
    Fetch Edvise bearer token from API key.

    Sends ``X-API-KEY`` and, when ``SST_IAP_AUDIENCE`` is set, an IAP token on
    ``Proxy-Authorization`` (so ``Authorization`` stays free for the app JWT).
    """
    token_url = (
        client.token_endpoint
        if client.token_endpoint.startswith(("http://", "https://"))
        else urljoin(f"{client.base_url}/", client.token_endpoint)
    )
    # Do not send a stale app JWT on the token-exchange call.
    client.session.headers.pop("Authorization", None)
    headers: dict[str, str] = {
        "accept": "application/json",
        "X-API-KEY": client.api_key,
        **iap_proxy_auth_headers(),
    }

    resp = client.session.post(token_url, headers=headers, timeout=30)
    body_preview = (resp.text or "")[:300]
    if resp.status_code == 401:
        if "Invalid IAP credentials" in body_preview:
            aud = os.getenv("SST_IAP_AUDIENCE", "").strip()
            hint = (
                "SST_IAP_AUDIENCE is unset on this cluster."
                if not aud
                else (
                    f"SST_IAP_AUDIENCE is set but IAP rejected the token "
                    f"(audience={aud!r}). Check the OAuth client ID and that the "
                    "cluster google_service_account has roles/iap.httpsResourceAccessor."
                )
            )
            raise PermissionError(
                f"Blocked by IAP calling token endpoint. {hint} "
                f"url={token_url} body={body_preview!r}"
            )
        raise PermissionError(
            "Unauthorized calling token endpoint (invalid X-API-KEY or credentials). "
            f"url={token_url} body={body_preview!r}"
        )
    resp.raise_for_status()

    data = resp.json()
    for k in ["access_token", "token", "bearer_token", "jwt"]:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    raise ValueError(
        "Token endpoint response missing expected token field. "
        f"Keys={list(data.keys())}"
    )


def _ensure_auth(client: EdviseAPIClient) -> None:
    """Ensure client has a valid Edvise bearer token and a fresh IAP proxy header."""
    if client.bearer_token is None:
        _refresh_auth(client)
    else:
        # Keep Edvise JWT on Authorization; refresh IAP for this request.
        client.session.headers.update(iap_proxy_auth_headers())


def _refresh_auth(client: EdviseAPIClient) -> None:
    """Refresh Edvise bearer token and set both auth headers on the session."""
    client.bearer_token = _fetch_bearer_token_for_client(client)
    client.session.headers["Authorization"] = f"Bearer {client.bearer_token}"
    client.session.headers.update(iap_proxy_auth_headers())


def fetch_institution_by_pdp_id(client: EdviseAPIClient, pdp_id: str) -> dict[str, Any]:
    """
    Resolve institution for PDP id using Edvise API.

    Cached within run. Automatically refreshes token on 401 errors.

    Args:
        client: EdviseAPIClient instance
        pdp_id: Institution PDP ID to look up

    Returns:
        Institution data dictionary from API

    Raises:
        ValueError: If institution PDP ID not found (404) or other API errors
        requests.HTTPError: For HTTP errors other than 401/404

    Example:
        >>> client = EdviseAPIClient(...)
        >>> inst = fetch_institution_by_pdp_id(client, "12345")
        >>> print(inst["name"])
        'Example University'
    """
    pid = str(pdp_id).strip()
    if pid in client.institution_cache:
        return client.institution_cache[pid]

    _ensure_auth(client)

    url = client.base_url + client.institution_lookup_path.format(pdp_id=pid)
    resp = client.session.get(url, timeout=30)

    if resp.status_code == 401:
        _refresh_auth(client)
        resp = client.session.get(url, timeout=30)

    if resp.status_code == 404:
        raise ValueError(f"Institution PDP ID not found in SST staging: {pid}")

    resp.raise_for_status()
    data = cast(dict[str, Any], resp.json())
    client.institution_cache[pid] = data
    return data
