import logging
import os
import random
import string
import time
from typing import Any, Dict, List

import openai
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

HF_TOKEN_CACHE_PATH = "/mnt/local_storage/data/cache/huggingface/token"
HF_TOKEN_LOCAL_PATH = "huggingface_token.txt"

NUM_RETRIES = 5
SLEEP_INTERVAL_BETWEEN_RETRIES = 10
ERROR_OUTPUT = "$$RUNTIME_ERROR$$"


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


def get_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]] = None,
    temperature: float = 0.0,
    max_tokens: int = 256,
) -> "ChatCompletion":
    """
    Gets completion from the OpenAI ChatCompletion API for the provided OpenAI client and model.

    Employs a simple retry strategy in case of rate limiting.
    """
    # Simple way to handle rate limit errors with retries
    for _ in range(NUM_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except openai.RateLimitError as e:
            # This will capture rate limiting errors
            print(f"Error: {e}")
            time.sleep(SLEEP_INTERVAL_BETWEEN_RETRIES)
    # Error response
    return ERROR_OUTPUT
