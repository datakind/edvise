import requests


def log_custom_job(
    inst_id: str, job_run_id: str, model_name: str, api_key: str
) -> t.Any:
    "Retrieve access token and log custom job ids on the GCP Cloud SQL JobTable"

    # Retrieve API token
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json",
    }

    access_token_url = "https://staging-sst.datakind.org/api/v1/token-from-api-key"
    r = requests.post(access_token_url, headers=headers)
    access_token = r.json()["access_token"]

    # Log custom jobs in JobTable
    custom_job_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    custom_job_endpoint_url = f"https://staging-sst.datakind.org/api/v1/{inst_id}/add-custom-school-job/{job_run_id}?model_name={model_name}"
    endpoint_result = requests.post(custom_job_endpoint_url, headers=custom_job_headers)
    return_response = (
        endpoint_result.json()
        if endpoint_result.json()
        else {"Check API Staging logs to debug this error"}
    )
    return return_response
