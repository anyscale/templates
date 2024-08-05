import logging
import os
import random
import string

HF_TOKEN_CACHE_PATH = "/mnt/local_storage/data/cache/huggingface/token"
HF_TOKEN_LOCAL_PATH = "huggingface_token.txt"


def init_logger():
    """Get the root logger"""
    return logging.getLogger()


def generate_output_path(output_path_prefix: str, model_id: str) -> str:
    """
    Constructs unique output path to write data out.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username[:5]
    else:
        username = "".join(random.choices(string.ascii_lowercase, k=5))
    suffix = "".join(random.choices(string.ascii_lowercase, k=5))
    return f"{output_path_prefix}/{model_id}:{username}:{suffix}"
