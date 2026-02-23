from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class SstApiClient:
    api_key: str
    base_url: str
    token_endpoint: str
    institution_lookup_path: str
    session: requests.Session = field(default_factory=requests.Session)
    bearer_token: str | None = None
    institution_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.api_key = self.api_key.strip()
        if not self.api_key:
            raise ValueError("Empty SST API key.")

        self.base_url = self.base_url.rstrip("/")
        self.token_endpoint = self.token_endpoint.strip()
        self.institution_lookup_path = self.institution_lookup_path.strip()

        self.session.headers.update({"accept": "application/json"})


def fetch_bearer_token(client: SstApiClient) -> str:
    """
    Fetch bearer token from API key using X-API-KEY header.
    Assumes token endpoint returns JSON containing one of: access_token, token, bearer_token, jwt.
    """
    resp = client.session.post(
        client.token_endpoint,
        headers={"accept": "application/json", "X-API-KEY": client.api_key},
        timeout=30,
    )
    if resp.status_code == 401:
        raise PermissionError(
            "Unauthorized calling token endpoint (check X-API-KEY secret)."
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


def ensure_auth(client: SstApiClient) -> None:
    if client.bearer_token is None:
        refresh_auth(client)


def refresh_auth(client: SstApiClient) -> None:
    client.bearer_token = fetch_bearer_token(client)
    client.session.headers.update({"Authorization": f"Bearer {client.bearer_token}"})


def fetch_institution_by_pdp_id(client: SstApiClient, pdp_id: str) -> dict[str, Any]:
    """
    Resolve institution for PDP id. Cached within run.
    Refresh token once on 401.
    """
    pid = str(pdp_id).strip()
    if pid in client.institution_cache:
        return client.institution_cache[pid]

    ensure_auth(client)

    url = client.base_url + client.institution_lookup_path.format(pdp_id=pid)
    resp = client.session.get(url, timeout=30)

    if resp.status_code == 401:
        refresh_auth(client)
        resp = client.session.get(url, timeout=30)

    if resp.status_code == 404:
        raise ValueError(f"Institution PDP ID not found in SST staging: {pid}")

    resp.raise_for_status()
    data = resp.json()
    client.institution_cache[pid] = data
    return data
