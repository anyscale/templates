import os
import random
import requests
import string

import huggingface_hub
from transformers import AutoConfig

HF_TOKEN_CACHE_PATH = "/home/ray/.cache/huggingface/token"


def generate_output_path(output_path_prefix: str, model_id: str) -> str:
    """
    Constructs unique output path to write data out.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username[:5]
    else:
        username = "".join(
            random.choices(string.ascii_lowercase, k=5)
        )
    suffix = "".join(random.choices(string.ascii_lowercase, k=5))
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


def prompt_for_hugging_face_token_dataset(hf_model: str) -> str:
    """Prompts the user for Hugging Face token if required by the model.
    Returns the token as a string. If a token is not required by the model,
    returns an empty string."""

    try:
        # Try loading model config to check if Hugging Face token is required.
        AutoConfig.from_pretrained(hf_model)
        # If config is loaded successfully, no need for token.
        return ""
    except OSError as e:
        if "You are trying to access a gated repo." in str(e):
            # Model requires HF token to access. Get the token, either from
            # cached token or from user input.
            if not os.path.isfile(HF_TOKEN_CACHE_PATH):
                print("No cached Hugging Face token found. Starting authentication on VS Code overlay...")
                # Starts authentication through VSCode overlay.
                # Token saved to `HF_TOKEN_CACHE_PATH`
                huggingface_hub.interpreter_login()

            return read_hugging_face_token_from_cache()
        else:
            # Some other error occurred, raise it.
            raise e


def read_hugging_face_token_from_cache() -> str:
    try:
        with open(HF_TOKEN_CACHE_PATH, "r") as file:
            tkn = file.read()
        print(f"Successfully read cached token at {HF_TOKEN_CACHE_PATH}.")
        return tkn
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find Hugging Face token cached at {HF_TOKEN_CACHE_PATH}."
        )
