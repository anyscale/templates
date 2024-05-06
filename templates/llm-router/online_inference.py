import pandas as pd
import json
import ray
from typing import Dict, Any, List
import copy
import openai
import time
import ray


@ray.remote(num_cpus=0)
def get_llm_response(
    base_url: str,
    api_key: str,
    llm: str,
    temperature: float,
    max_tokens: int,
    pidx: int,
    messages: List[Dict[str, str]],
    max_retries=1,
    retry_interval=60,
) -> Dict[int, str]:
    """
    Use OpenAI's API to request completions from a specified LLM and manages request retries upon failures.
    """
    retry_count = 0
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    while retry_count <= max_retries:
        try:
            response = client.chat.completions.create(
                model=llm,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (pidx, response.choices[0].message.content)
        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return (pidx, "")


def generate_batch_responses(
    base_url: str,
    api_key: str,
    llm: str,
    queries: Dict[int, Any],
    max_concurrent_queries: int,
    temperature: float,
    max_tokens: int,
    verbose: bool = False,
) -> Dict[int, str]:
    """
    This function manages online batch inference of queries using a specified LLM, tracking progress and handling responses.
    """
    print(f"Starting batch inference on {len(queries)} queries...")
    queue = copy.copy(queries)
    in_progress, responses = [], []

    start_time = time.time()
    while queue or in_progress:
        if len(in_progress) < max_concurrent_queries and queue:
            pidx, messages = queue.popitem()
            in_progress.append(
                get_llm_response.remote(
                    base_url, api_key, llm, temperature, max_tokens, pidx, messages
                )
            )
        ready, in_progress = ray.wait(in_progress, timeout=0.5)
        if verbose:
            print(
                f"# queries un-processed: {len(queue)}, in-progress: {len(in_progress)}, ready: {len(ready)}"
            )
        if ready:
            responses.extend(ray.get(ready))

    print(f"Done in {time.time() - start_time:.2f}sec.")
    return dict(responses)

