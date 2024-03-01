import os
import random
import string

def generate_output_path(output_path_prefix: str, model_id: str) -> str:
    """
    Constructs unique output path to write data out.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username[:5]
    else:
        username = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    return f"{output_path_prefix}/{model_id}:{username}:{suffix}"
