import os
import requests

import huggingface_hub

HF_TOKEN_CACHE_PATH = "/mnt/local_storage/data/cache/huggingface/token"
HF_TOKEN_LOCAL_PATH = "huggingface_token.txt"

def is_on_gcp_cloud() -> bool:
    """Detects if the cluster is running on GCP."""
    try:
        resp = requests.get("http://metadata.google.internal")
        return resp.headers["Metadata-Flavor"] == "Google"
    except:  # noqa: E722
        return False


def prompt_for_hugging_face_token(hf_model: str) -> str:
    """Prompts the user for Hugging Face token if required by the model.
    Returns the token as a string. If a token is not required by the model,
    returns an empty string."""

    url = f"https://huggingface.co/api/models/{hf_model}"

    response = requests.get(url)
    response_json = response.json()
    is_private = response_json.get("private", False)
    # possible gated response values: [False, "auto", "manual"]
    is_gated = response_json.get("gated", False) in ("auto", "manual")
    if response.status_code == 200 and not (is_private or is_gated):
        return ""
    elif response.status_code == 401 or (is_private or is_gated):
        if os.path.isfile(HF_TOKEN_LOCAL_PATH):
            return read_hugging_face_token_from_cache(HF_TOKEN_LOCAL_PATH)
        if not os.path.isfile(HF_TOKEN_CACHE_PATH):
            print("No cached Hugging Face token found. Starting authentication on VS Code overlay...")
            # Starts authentication through VSCode overlay.
            # Token saved to `HF_TOKEN_CACHE_PATH`
            huggingface_hub.interpreter_login()

        tkn = read_hugging_face_token_from_cache(HF_TOKEN_CACHE_PATH)
        # Write the token to a local file, so it can be used by Ray job.
        with open(HF_TOKEN_LOCAL_PATH, "w") as file:
            file.write(tkn)
        return tkn
    else:
        raise Exception(f"Failed to access the model. Status code: {response.status_code}")


def read_hugging_face_token_from_cache(path) -> str:
    try:
        with open(path, "r") as file:
            tkn = file.read()
        print(f"Successfully read cached token at {path}.")
        return tkn
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find Hugging Face token cached at {path}."
        )
