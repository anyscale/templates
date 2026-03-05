import time
from ray import serve
from typing import Any


@serve.deployment(num_replicas=3)
class TextProcessor:
    """A simple text processing deployment that can be scaled externally."""

    def __init__(self):
        self.request_count = 0

    def __call__(self, text: Any) -> dict:
        # Simulate text processing work
        time.sleep(0.1)
        self.request_count += 1
        return {
            "request_count": self.request_count,
        }


app = TextProcessor.bind()
