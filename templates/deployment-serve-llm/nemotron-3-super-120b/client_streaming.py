# client_streaming.py
from urllib.parse import urljoin

from openai import OpenAI

API_KEY = "FAKE_KEY"
BASE_URL = "http://localhost:8000"

client = OpenAI(base_url=urljoin(BASE_URL, "v1"), api_key=API_KEY)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super",
    messages=[{"role": "user", "content": "Explain why the sky is blue."}],
    temperature=1.0,
    top_p=0.95,
    stream=True,
)

for chunk in response:
    delta = chunk.choices[0].delta
    # Stream the reasoning trace first. Newer vLLM builds expose it as
    # `reasoning` while older ones use `reasoning_content`; read via model_extra
    # so both work (the typed OpenAI SDK doesn't surface these fields directly).
    extra = getattr(delta, "model_extra", None) or {}
    reasoning = extra.get("reasoning") or extra.get("reasoning_content")
    if reasoning:
        print(reasoning, end="", flush=True)
    # Then stream the final answer.
    if delta.content:
        print(delta.content, end="", flush=True)
