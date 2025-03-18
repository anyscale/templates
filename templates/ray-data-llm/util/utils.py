import requests


def is_on_gcp_cloud() -> bool:
    """Detects if the cluster is running on GCP."""
    try:
        resp = requests.get("http://metadata.google.internal")
        return resp.headers["Metadata-Flavor"] == "Google"
    except:  # noqa: E722
        return False
