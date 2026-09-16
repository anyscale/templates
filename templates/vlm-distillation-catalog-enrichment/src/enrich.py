"""
GPU enrichment stage — Qwen2.5-VL-3B-Instruct via vLLM.

Two paths, mirroring `notebooks/demo_walkthrough.ipynb`:

  1. NaiveVLMEnricher (callable class)
     Single-actor approach: each GPU actor does HTTP fetch + decode + VLM
     generate inline. The GPU sits idle while images download. The "before".

  2. build_heterogeneous_processor (ray.data.llm)
     Decoupled approach: messages carry the image as an OpenAI-spec base64
     data URL (Arrow-native, no PIL → pickle fallback across stages).
     CPU stage upstream produces ``image_bytes``; this stage runs only the
     VLM via ray.data.llm + vLLM. The GPU stays saturated. The "after".
"""
import base64
import io

from PIL import Image


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

ENRICHMENT_PROMPT = """\
You are a product catalog enrichment assistant. Given a product image and \
the merchant-supplied title, output a JSON object with exactly these keys:

  category:    one short string (e.g. "Wireless Earbuds")
  attributes:  a list of 3 short attribute strings
  search_tags: a list of 5 short search keywords
  description: a single sentence (<= 30 words)

Title: {title}

Return ONLY the JSON object, no commentary.\
"""


def build_messages(image: Image.Image, title: str) -> list:
    """Naive variant: PIL.Image inline. Used inside a single actor only —
    the message dict never crosses Ray Data block boundaries, so PIL is fine."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ENRICHMENT_PROMPT.format(title=title)},
            ],
        }
    ]


def build_messages_url(data_url: str, title: str) -> list:
    """Hetero variant: image as an OpenAI-spec base64 data URL string.

    Strings serialize zero-copy through Arrow across operator boundaries —
    no PIL → pickle fallback warnings, no per-stage memory blow-up.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": ENRICHMENT_PROMPT.format(title=title)},
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Path 1 — NAIVE: one actor does fetch + decode + generate
# ---------------------------------------------------------------------------

class NaiveVLMEnricher:
    """One GPU actor: HTTP fetch + PIL decode + vLLM generate, all inline."""

    def __init__(self, model_id: str = MODEL_ID):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_id,
            trust_remote_code=True,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 1},
            dtype="float16",
            gpu_memory_utilization=0.85,
        )
        self.sampling = SamplingParams(max_tokens=256, temperature=0.0)
        self.tokenizer = self.llm.get_tokenizer()

    def __call__(self, batch: dict) -> dict:
        import requests

        prompts, kept = [], []
        for i in range(len(batch["product_id"])):
            try:
                resp = requests.get(
                    batch["image_url"][i],
                    timeout=5.0,
                    headers={"User-Agent": "vlm-distillation-catalog-enrichment/1.0"},
                )
                if resp.status_code != 200:
                    continue
                img = (
                    Image.open(io.BytesIO(resp.content))
                    .convert("RGB")
                    .resize((384, 384))
                )
            except Exception:
                continue

            prompt = self.tokenizer.apply_chat_template(
                build_messages(img, batch["title"][i]),
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append({"prompt": prompt, "multi_modal_data": {"image": img}})
            kept.append(i)

        if not prompts:
            return {k: [] for k in ("product_id", "title", "raw_output")}

        outputs = self.llm.generate(prompts, self.sampling)
        return {
            "product_id": [batch["product_id"][i] for i in kept],
            "title": [batch["title"][i] for i in kept],
            "raw_output": [o.outputs[0].text for o in outputs],
        }


# ---------------------------------------------------------------------------
# Path 2 — HETEROGENEOUS: ray.data.llm processor, CPU pool feeds it bytes
# ---------------------------------------------------------------------------

def build_heterogeneous_processor(num_gpus: int = 2, batch_size: int = 8, model_id: str = MODEL_ID):
    """Returns a ray.data.llm processor that consumes ``image_bytes`` from the
    upstream CPU stage (see ``src.preprocess.fetch_and_decode``).

    Expected input columns:  product_id, title, image_bytes
    Output columns:          product_id, title, raw_output
    """
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

    config = vLLMEngineProcessorConfig(
        model_source=model_id,
        engine_kwargs={
            "trust_remote_code": True,
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1},
            "dtype": "float16",
            "gpu_memory_utilization": 0.85,
        },
        concurrency=num_gpus,
        batch_size=batch_size,
        # vLLM 0.20+ moved TokensPrompt up from vllm.inputs.data to vllm.inputs.
        # Ray 2.55's batch LLM stage still imports the old path, so each worker
        # patches it on startup. Job-level ray.init also sets this hook for
        # belt-and-suspenders.
        runtime_env={"worker_process_setup_hook": "src._vllm_compat.patch"},
    )

    def preprocess(row):
        b64 = base64.b64encode(row["image_bytes"]).decode("ascii")
        data_url = f"data:image/jpeg;base64,{b64}"
        return dict(
            messages=build_messages_url(data_url, row["title"]),
            sampling_params=dict(max_tokens=256, temperature=0.0),
        )

    def postprocess(row):
        return {
            "product_id": row["product_id"],
            "title": row["title"],
            "raw_output": row.get("generated_text", "") or "",
        }

    return build_llm_processor(config, preprocess=preprocess, postprocess=postprocess)
