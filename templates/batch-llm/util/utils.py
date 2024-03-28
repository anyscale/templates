import os
import random
import requests
import string


def generate_output_path(output_path_prefix: str, model_id: str) -> str:
    """
    Constructs unique output path to write data out.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username[:5]
    else:
        username = "".join(
            random.choice(string.ascii_lowercase) for _ in range(5)
        )
    suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    return f"{output_path_prefix}/{model_id}:{username}:{suffix}"


def _on_gcp_cloud() -> bool:
    """Detects if the cluster is running on GCP."""
    try:
        resp = requests.get("http://metadata.google.internal")
    except:  # noqa: E722
        return False
    return resp.headers["Metadata-Flavor"] == "Google"


def get_a10g_or_equivalent_accelerator_type() -> str:
    """Returns an accelerator type string for an A10G (or equivalent) GPU.

    Equivalence is determined by the amount of GPU memory in this case.
    GCP doesn't provide instance types with A10G GPUs, so we request L4 GPUs
    instead.
    """
    return "L4" if _on_gcp_cloud() else "A10G"
