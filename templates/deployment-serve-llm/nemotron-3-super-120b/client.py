# client.py
from urllib.parse import urljoin

from openai import OpenAI

API_KEY = "FAKE_KEY"
BASE_URL = "http://localhost:8000"

client = OpenAI(base_url=urljoin(BASE_URL, "v1"), api_key=API_KEY)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super",
    messages=[
        {
            "role": "user",
            "content": "What is the sum of all even numbers between 1 and 100?",
        }
    ],
    # NVIDIA recommends these sampling defaults for all tasks.
    temperature=1.0,
    top_p=0.95,
)

# Reasoning trace is separated from the final answer by the "nemotron_v3" parser.
# This model exposes the trace in the `reasoning` field (older vLLM builds use
# `reasoning_content`); read via model_extra since the typed OpenAI SDK doesn't
# surface either field directly.
message = response.choices[0].message
extra = getattr(message, "model_extra", None) or {}
reasoning = extra.get("reasoning") or extra.get("reasoning_content")
print(f"Reasoning: \n{reasoning}\n\n")
print(f"Answer: \n{message.content}")
