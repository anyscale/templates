"""
Ray Data — Multimodal Catalog Pipeline: Qwen2.5-VL Enrichment + SigLIP 2 Embeddings

One Ray Data graph that produces, for every product, BOTH:
  - the structured-attribute JSON a generative VLM emits (category / attributes /
    tags / description), and
  - dense image AND text vectors in a shared multimodal space (1152-dim,
    L2-normalized) ready to push into FAISS / OpenSearch k-NN / Vespa.

This is the shape of a real Walmart-class catalog-understanding pipeline:
  - Generative VLM enrichment fills the search index (categories, tags, descriptions).
  - Dual-tower image+text encoder fills the vector index (visual search,
    out-of-stock substitution, listing dedup, "more like this", multimodal recall).
  Same images, two parallel outputs, one Ray Data run.

Pipeline shape:
  LOAD → PREPROCESS → URL_CHECK → CPU_FETCH
       → VLM_ENRICH (GPU pool, ray.data.llm)            ← Qwen2.5-VL-3B
       → CPU_PROCESS (CPU pool, SigLIP processor + tok) ← consumes image_bytes + raw_output
       → GPU_EMBED (GPU pool, pure inference)           ← SigLIP image+text features
       → WRITE

Each comment block below says WHERE the stage runs (CPU or GPU) and links to
the relevant docs.

Run directly on the workspace cluster:
  python scripts/run_enc_vlm_batch_emb_enrich_3b.py

Or as an Anyscale Job (re-use vlm_32b_job_config.yaml — same g6.12xlarge fits;
flip cpu-workers max_nodes back to a positive number — this pipeline WANTS the
CPU pool):
  anyscale job submit --config-file vlm_32b_job_config.yaml \\
    --entrypoint "python scripts/run_enc_vlm_batch_emb_enrich_3b.py" \\
    --env HF_TOKEN=$HF_TOKEN

NOTE: this is a batch Ray Data pipeline (datasets in → parquet out). It is
NOT a Ray Serve app and cannot be run via `serve run`. To expose both models
as HTTP endpoints — Qwen on /v1/chat/completions, SigLIP on /embed — use a
separate Serve file (not built here). See scripts/run_vlm_online_enrich_3b.py
for the OpenAI-compatible Serve idiom.

──────────────────────────────────────────────────────────
Why two GPU stages, not one — and why the CPU split matters
──────────────────────────────────────────────────────────
Qwen2.5-VL is a *generative* VLM (autoregressive token decode, hidden states
not contrastively aligned for retrieval, hundreds of ms/image). SigLIP 2 is
a contrastive dual-tower encoder (image + text into the same 1152-dim space,
~thousands of items/sec on one L4). Same images, complementary outputs.

The CPU/GPU split for the SigLIP path mirrors the Anyscale cross-modal-search
pattern: a CPU `map` actor (Process) does the image processor + text
tokenizer; a GPU `map_batches` actor (Embed) does PURE inference on
pre-tensorized inputs. The L4 spends every cycle on forward-pass FLOPs, never
on PIL or tokenization. Each stage has its own concurrency knob.

  Blog: https://www.anyscale.com/blog/cross-modal-search-for-e-commerce-building-and-scaling-a-cross-modal-image-retrieval-app
  Code: https://github.com/anyscale/cross-modal-search-ecommerce-project
  SigLIP 2: https://huggingface.co/google/siglip2-so400m-patch14-384

Domain-tuned alternative (better for ecommerce-specific recall):
  Marqo/marqo-ecommerce-embeddings-L  — same dual-tower shape, drop-in swap.

──────────────────────────────────────────────────────────
What gets optimized for IO and GPU at multi-node scale
──────────────────────────────────────────────────────────
  1. Single image fetch, two consumers. CPU_FETCH decodes, resizes to
     512×512, attaches raw JPEG bytes. The VLM stage (via base64 data URL —
     no HTTP refetch) and the SigLIP CPU_PROCESS stage (direct PIL decode)
     both read those same bytes.
  2. Resize to a size both models like. 512×512 sits below Qwen2.5-VL's
     max_pixels cap (~633×633) so its dynamic-resolution processor uses the
     image as-is, and SigLIP's processor downscales to 384 internally.
  3. Cap Qwen2.5-VL vision tokens at ~512/image via mm_processor_kwargs.
     Halves prefill work — the dominant cost on the VLM stage.
  4. SigLIP text tower embeds the *enriched* description, not just the
     title. We pull "description" out of the VLM's raw_output JSON in
     ProcessSigLIP and concat with the title before tokenization. Sparse
     merchant titles → richer text vectors → better retrieval recall.
  5. Process and Embed are separate Ray Data stages. Each has its own
     concurrency knob (CPU pool vs GPU pool); the GPU actor never touches
     PIL, tokenizers, or Python transforms.
  6. Streaming executor pipelines all stages. SigLIP_GPU runs downstream of
     VLM (the heavy bottleneck) — when VLM is busy, SigLIP idles between
     batches. Both GPU pools autoscale concurrency independently.
  7. Stable SHA-1(title|image_url) row IDs + CheckpointConfig. Job-level
     resumability across worker death or cancel-resubmit. Random UUIDs
     would regenerate per submission and the checkpoint would never match.
"""

import os, sys, json, io, base64, hashlib
from typing import Optional

import numpy as np
import ray
import requests
from huggingface_hub import HfFileSystem
from ray.data.llm import vLLMEngineProcessorConfig, build_processor
from ray.data.checkpoint import CheckpointConfig


# Repo root — so `src._vllm_compat` resolves on the driver and inside Ray workers.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ──────────────────────────────────────────────────────────
# Knobs — the only things you usually tune
# ──────────────────────────────────────────────────────────
CATEGORY = "Electronics"
N_ROWS = 10_000          # bump to 10_000 / 100_000 when going beyond a smoke run
SEED = 42

# VLM (generative) — Qwen2.5-VL-3B fits on a single L4 at bf16.
VLM_MODEL_SOURCE = "Qwen/Qwen2.5-VL-3B-Instruct"
VLM_TENSOR_PARALLEL_SIZE = 1
VLM_PIPELINE_PARALLEL_SIZE = 1
VLM_MAX_MODEL_LEN = 2048           # image (~512 tok) + prompt (~150) + gen (≤160) ≈ 822
VLM_BATCH_SIZE = 48
VLM_CONCURRENCY = 2                # set (min, max) for autoscaling, e.g. (1, 4)

# Embedding (contrastive dual-tower) — SigLIP 2 so400m, 1152-dim shared space.
EMB_MODEL_SOURCE = "google/siglip2-so400m-patch14-384"
EMB_BATCH_SIZE = 32                # bs=64 fits comfortably on L4 24GB at bf16; 32 is conservative.
EMB_CONCURRENCY = 2                # one actor per L4. Same autoscale story as VLM_CONCURRENCY.

# CPU FETCH (HTTP + PIL decode + resize). Network IO bound.
IMAGE_RESIZE = 512
FETCH_TIMEOUT_S = 5.0
FETCH_CONCURRENCY = 16

# CPU PROCESS (SigLIP image processor + text tokenizer). CPU bound.
PROCESS_CONCURRENCY = 8

BASE_DIR = "/mnt/shared_storage/walmart-notebooks"
HF_PATH = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_{CATEGORY}"
CACHE_PATH = f"{BASE_DIR}/catalog_{CATEGORY}_{N_ROWS}.parquet"
CHECKPOINT_PATH = f"{BASE_DIR}/enc_vlm_emb_enrich_{N_ROWS}_checkpoint"
OUTPUT_PATH = f"{BASE_DIR}/enc_vlm_enriched_with_embeddings_{N_ROWS}.parquet"


# ──────────────────────────────────────────────────────────
# Optional LoRA adapter (mirrors run_vlm_online_search_3b.py)
# ──────────────────────────────────────────────────────────
# If the adapter dir exists, route VLM enrichment through the fine-tuned
# adapter via vLLM's LoRA multiplexing; otherwise fall back to the base 3B
# model. Override the location with QWEN_LORA_ADAPTER_DIR.
#
# ray.data.llm batch wiring:
#   - dynamic_lora_loading_path (top-level on vLLMEngineProcessorConfig) is the
#     S3 prefix that *contains* adapter subfolders.
#   - Per-row "model" = "<adapter_name>" triggers a LoRARequest; vLLM then
#     pulls "<dynamic_lora_loading_path>/<adapter_name>/" on first use and
#     caches the loaded adapter for the rest of the run.
#   - dynamic_lora_loading_path requires a cloud URI (s3://, gs://, ...). When
#     the adapter dir is local (under /mnt), we sync it to the workspace's
#     ANYSCALE_ARTIFACT_STORAGE bucket once and use that S3 prefix.
LORA_ADAPTER_DIR = os.environ.get(
    "QWEN_LORA_ADAPTER_DIR",
    "s3://anyscale-production-data-cld-g54aiirwj1s8t9ktgzikqur41k/org_967t9ah1lbk1yqf1zau6a1v247/cld_g54aiirwj1s8t9ktgzikqur41k/artifact_storage/loras/qwen25vl_3b_enrichment_lora_1000",
)
LORA_MAX_RANK = 16  # must be ≥ LORA_R used during training (run_vlm_ft_enrich_3b.py)


def _ensure_lora_remote(local_dir: str) -> Optional[str]:
    """Sync a local LoRA dir to the workspace artifact bucket and return the
    parent S3 prefix. If `local_dir` is already a cloud URI, return its
    parent. Returns None if we can't make a usable cloud prefix.
    """
    if local_dir.startswith(("s3://", "gs://", "abfss://", "azure://")):
        return os.path.dirname(local_dir.rstrip("/"))

    artifact_root = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
    if not artifact_root:
        print(
            f"[vlm] LoRA dir {local_dir!r} is local and "
            "ANYSCALE_ARTIFACT_STORAGE is not set; can't auto-sync to cloud. "
            "Either set QWEN_LORA_ADAPTER_DIR to an s3:// path or run on an "
            "Anyscale workspace with artifact storage configured."
        )
        return None

    adapter_name = os.path.basename(local_dir.rstrip("/"))
    s3_prefix = f"{artifact_root.rstrip('/')}/loras"
    s3_dest = f"{s3_prefix}/{adapter_name}"
    import subprocess
    print(f"[vlm] syncing LoRA {local_dir} → {s3_dest}")
    res = subprocess.run(
        ["aws", "s3", "sync", local_dir, s3_dest],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print(f"[vlm] aws s3 sync failed:\n{res.stderr}")
        return None
    return s3_prefix


_lora_dir = LORA_ADAPTER_DIR.rstrip("/")
_lora_ready = (
    os.path.isdir(_lora_dir)
    and os.path.exists(os.path.join(_lora_dir, "adapter_config.json"))
) or _lora_dir.startswith(("s3://", "gs://", "abfss://", "azure://"))

LORA_REMOTE_PREFIX: Optional[str] = None
LORA_ADAPTER_NAME: Optional[str] = None
if _lora_ready:
    LORA_REMOTE_PREFIX = _ensure_lora_remote(_lora_dir)
    _lora_ready = LORA_REMOTE_PREFIX is not None

if _lora_ready:
    LORA_ADAPTER_NAME = os.path.basename(_lora_dir.rstrip("/"))
    print(f"[vlm] LoRA adapter ready at {LORA_REMOTE_PREFIX}/{LORA_ADAPTER_NAME}; "
          f"routing batch enrichment requests with model='{LORA_ADAPTER_NAME}'")
else:
    print(f"[vlm] no usable LoRA adapter; falling back to base model='{VLM_MODEL_SOURCE}'")


# ──────────────────────────────────────────────────────────
# STAGE 1 — LOAD + STAGE 2 — PREPROCESS  (CPU)
# ──────────────────────────────────────────────────────────
# Pulls Amazon-Reviews-2023 metadata from HF, drops rows missing title/image,
# normalizes one row per product, HEAD-checks the URL, caches as parquet.
# Runs entirely on CPU; Ray Data block parallelism does the work.
# https://docs.ray.io/en/latest/data/loading-data.html

def _extract_image_url(images_field):
    if not images_field or not isinstance(images_field, dict):
        return None
    for key in ("large", "hi_res", "thumb"):
        urls = images_field.get(key)
        if urls is not None and len(urls) > 0:
            return urls[0]
    return None


def _has_title_and_image(row):
    title = row.get("title")
    if not (title and title.strip()):
        return False
    return _extract_image_url(row.get("images")) is not None


def _coerce_description(desc_field):
    if isinstance(desc_field, list):
        return " ".join(str(x) for x in desc_field if x).strip()
    if isinstance(desc_field, str):
        return desc_field.strip()
    return ""


def _url_is_reachable(row):
    try:
        return requests.head(row["image_url"], timeout=FETCH_TIMEOUT_S, allow_redirects=True).ok
    except Exception:
        return False


def _normalize_amazon_row(row):
    title = row["title"].strip()[:512]
    image_url = _extract_image_url(row["images"])
    row_id = hashlib.sha1(f"{title}|{image_url}".encode("utf-8")).hexdigest()[:16]
    return {
        "id": row_id,
        "product_id": row.get("parent_asin") or row.get("asin") or "",
        "title": title,
        "description": _coerce_description(row.get("description"))[:1024],
        "image_url": image_url,
        "source": "amazon-reviews-2023",
    }


def build_catalog():
    # Cache check: /mnt/shared_storage/walmart-notebooks is per-user persistent, so once we've
    # built the catalog at this (CATEGORY, N_ROWS) we never need to redo the
    # HF read + URL HEAD checks. Saves ~10 min/run for 10K rows.
    if os.path.exists(CACHE_PATH):
        cached = ray.data.read_parquet(CACHE_PATH)
        print(f"[load hf] reusing cache at {CACHE_PATH}")
        return cached

    print(f"[load hf] Loading {HF_PATH} ...")
    ds = ray.data.read_parquet(
        HF_PATH,
        file_extensions=["parquet"],
        filesystem=HfFileSystem(),
    )
    ds = ds.limit(N_ROWS)
    ds = ds.filter(_has_title_and_image)
    ds = ds.map(_normalize_amazon_row)
    ds = ds.filter(_url_is_reachable)
    ds = ds.random_shuffle(seed=SEED)
    # mode="overwrite": write_parquet defaults to APPEND (writes new files into
    # the directory without removing existing ones). If a stale partial run
    # left files behind, the next run would silently mix old + new rows on read
    # — exactly the 19,991-row "two runs accumulated" bug we hit before the
    # cache_check above was added.
    ds.write_parquet(CACHE_PATH, mode="overwrite")
    # CRITICAL: count from the materialized parquet, NOT the lazy `ds`. Calling
    # ds.count() here would re-execute the entire upstream plan (HF read +
    # filters + URL HEAD checks + shuffle) from scratch — wasted ~10 min on
    # the first 10K-row run before this fix. read_parquet().count() reads
    # parquet row-group metadata, near-instant.
    cached = ray.data.read_parquet(CACHE_PATH)
    print(f"[load hf] cached {cached.count()} rows → {CACHE_PATH}")
    return cached


# ──────────────────────────────────────────────────────────
# STAGE 3 — CPU FETCH  (CPU, autoscaled actor pool)
# ──────────────────────────────────────────────────────────
# One HTTP round-trip per image. Bytes attached to the row so BOTH downstream
# GPU stages can consume them without re-fetching:
#   - VLM stage gets a base64 data URL (built in vlm_preprocess) — vLLM's
#     prepare_multimodal_stage decodes the base64 in-place, no HTTP.
#   - SigLIP CPU_PROCESS decodes the bytes directly via PIL.

def fetch_and_resize(batch):
    from PIL import Image
    out_id, out_pid, out_title, out_desc, out_url, out_src, out_bytes = (
        [], [], [], [], [], [], [],
    )
    for i in range(len(batch["id"])):
        try:
            r = requests.get(
                batch["image_url"][i],
                timeout=FETCH_TIMEOUT_S,
                headers={"User-Agent": "anyscale-demo/1.0"},
            )
            if r.status_code != 200:
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGB").resize(
                (IMAGE_RESIZE, IMAGE_RESIZE)
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            jpeg_bytes = buf.getvalue()
        except Exception:
            continue
        out_id.append(batch["id"][i])
        out_pid.append(batch["product_id"][i])
        out_title.append(batch["title"][i])
        out_desc.append(batch["description"][i])
        out_url.append(batch["image_url"][i])
        out_src.append(batch["source"][i])
        out_bytes.append(jpeg_bytes)
    return {
        "id": out_id,
        "product_id": out_pid,
        "title": out_title,
        "description": out_desc,
        "image_url": out_url,
        "source": out_src,
        "image_bytes": out_bytes,
    }


# ──────────────────────────────────────────────────────────
# STAGE 4 — VLM ENRICH  (GPU, ray.data.llm processor)
# ──────────────────────────────────────────────────────────
# build_processor builds a multi-stage sub-pipeline under the hood:
#
#   PREPROCESS (CPU) → PREPARE_IMAGES (CPU) → ChatTemplate (CPU)
#       → Tokenize (CPU) → vLLM Engine (GPU) → Detokenize (CPU) → POSTPROCESS (CPU)
#
# Because we already have image_bytes on the row, we feed vLLM a base64 data
# URL — prepare_multimodal_stage decodes the base64 in-place instead of doing
# an HTTP fetch. One fetch upstream, zero refetches.
#
# CRITICAL: vlm_preprocess and vlm_postprocess must round-trip image_bytes
# through this stage so the downstream SigLIP CPU_PROCESS can read them.
# vLLM only forwards columns we name; everything else is dropped after detokenize.
# https://docs.ray.io/en/latest/data/working-with-llms.html
# https://docs.ray.io/en/latest/data/working-with-llms.html#multimodal

VLM_PROMPT = """\
You are a product catalog enrichment assistant. Given a product image and \
the merchant-supplied title, output a JSON object with exactly these keys:

  category:    one short string (e.g. "Wireless Earbuds")
  attributes:  a list of 3 short attribute strings
  search_tags: a list of 5 short search keywords
  description: a single sentence (<= 30 words)

Title: {title}

Return ONLY the JSON object, no commentary.\
"""


def vlm_preprocess(row):
    b64 = base64.b64encode(row["image_bytes"]).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    out = {
        "id": row["id"],
        # Round-trip these so the SigLIP CPU_PROCESS stage downstream can pick
        # them up. vLLM's processor only forwards columns we name in postprocess;
        # everything else is dropped after detokenize.
        "image_bytes": row["image_bytes"],
        "product_id": row["product_id"],
        "title": row["title"],
        "image_url": row["image_url"],
        "source": row["source"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": VLM_PROMPT.format(title=row["title"])},
                ],
            }
        ],
        # max_tokens=160: 4-key JSON is ~120 tok typical; 160 is a safe ceiling
        # and shrinks per-seq KV reservation vs 256.
        "sampling_params": {"max_tokens": 160, "temperature": 0.0},
    }
    # When a LoRA adapter is loaded, ray.data.llm routes the request through
    # it iff the row's "model" field differs from the processor's
    # `model_source` (the base). Set it to the adapter name so vLLM downloads
    # `<dynamic_lora_loading_path>/<adapter_name>/` once and reuses it.
    if LORA_ADAPTER_NAME is not None:
        out["model"] = LORA_ADAPTER_NAME
    return out


def vlm_postprocess(row):
    return {
        "id": row["id"],
        "product_id": row["product_id"],
        "title": row["title"],
        "image_url": row["image_url"],
        "image_bytes": row["image_bytes"],
        "source": row["source"],
        "raw_output": row["generated_text"],
    }


def build_vlm_processor():
    engine_kwargs = {
        "tensor_parallel_size": VLM_TENSOR_PARALLEL_SIZE,
        "pipeline_parallel_size": VLM_PIPELINE_PARALLEL_SIZE,
        "max_model_len": VLM_MAX_MODEL_LEN,
        "trust_remote_code": True,
        # Chunked prefill: image-token prefills are huge (~500 tok/image
        # after the cap below). Chunking lets prefill interleave with
        # ongoing decode steps in the same forward pass — big throughput
        # win for multimodal.
        "enable_chunked_prefill": True,
        "max_num_batched_tokens": 8192,
        "limit_mm_per_prompt": {"image": 1},
        # Cap Qwen2.5-VL's dynamic-resolution processor. Token count
        # ≈ pixels / (28×28). 512×(28²) = 401,408 px (~633×633) → ~512
        # vision tokens / image. Halves prefill vs uncapped.
        "mm_processor_kwargs": {
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 512 * 28 * 28,
        },
        "gpu_memory_utilization": 0.85,
    }
    extra_kwargs = {}
    if _lora_ready:
        engine_kwargs.update({
            "enable_lora": True,
            "max_lora_rank": LORA_MAX_RANK,
            "max_loras": 1,
        })
        # Top-level vLLMEngineProcessorConfig field — the cloud prefix that
        # contains adapter subfolders. Per-row "model" picks which subfolder.
        extra_kwargs["dynamic_lora_loading_path"] = LORA_REMOTE_PREFIX

    config = vLLMEngineProcessorConfig(
        model_source=VLM_MODEL_SOURCE,
        engine_kwargs=engine_kwargs,
        batch_size=VLM_BATCH_SIZE,
        concurrency=VLM_CONCURRENCY,
        prepare_multimodal_stage={"enabled": True},
        # vLLM 0.20+ moved TokensPrompt; Ray 2.55's batch LLM stage still
        # imports the old path. Patch lives in src/_vllm_compat.py.
        runtime_env={"worker_process_setup_hook": "src._vllm_compat.patch"},
        **extra_kwargs,
    )
    return build_processor(config, preprocess=vlm_preprocess, postprocess=vlm_postprocess)


# ──────────────────────────────────────────────────────────
# STAGE 5 — CPU PROCESS  (CPU, autoscaled actor pool)
# ──────────────────────────────────────────────────────────
# SigLIP image processor (PIL → pixel_values) + SigLIP tokenizer (enriched
# text → input_ids). One CPU actor since both operate on the same row.
# Decoupled from the GPU stage on purpose: the GPU actor receives ready-to-go
# tensors and spends all its cycles on forward-pass FLOPs.
#
# One-pipeline payoff: we tokenize "title + VLM description" instead of just
# title. Sparse merchant titles get enriched by the VLM upstream → richer
# text vectors → better retrieval recall on descriptive queries.
# https://huggingface.co/google/siglip2-so400m-patch14-384

class ProcessSigLIP:
    """CPU-side image+text preprocessing.

    Input columns:  image_bytes (jpeg), title, raw_output (VLM JSON string)
    Output columns: pixel_values, input_ids
                    (image_bytes dropped — no need to ship raw bytes to GPU;
                     raw_output kept so it lands in the final parquet)
    """

    def __init__(self, model_id: str):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)

    @staticmethod
    def _compose_text(title: str, raw_output: str) -> str:
        # Embed title + VLM description in the text tower. The VLM enrichment
        # makes the embedding stage's text side richer, which improves recall
        # for descriptive queries ("noise-cancelling over-ear") that wouldn't
        # match a sparse merchant title. This is the one-pipeline payoff.
        try:
            obj = json.loads(raw_output)
            desc = obj.get("description") or ""
            return f"{title}. {desc}".strip(". ").strip() if desc else title
        except Exception:
            return title

    def __call__(self, row: dict) -> dict:
        from PIL import Image

        img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
        img_inputs = self.processor(images=img, return_tensors="pt")
        # SigLIP 2 was trained with fixed 64-token text inputs; padding="max_length"
        # is what the model expects, NOT padding=True. Match it or quality drops.
        # NOTE: SigLIP's tokenizer always pads to max_length and the model treats
        # every token as valid — the processor does NOT return an attention_mask,
        # and get_text_features doesn't need one. Don't try to grab one.
        text = self._compose_text(row["title"], row.get("raw_output") or "")
        txt_inputs = self.processor(
            text=text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        row["pixel_values"] = img_inputs["pixel_values"][0].numpy()       # (3, 384, 384)
        row["input_ids"] = txt_inputs["input_ids"][0].numpy()             # (64,)
        # Drop raw bytes — the GPU stage doesn't need them, and shipping them
        # across the streaming executor is just memory pressure.
        row.pop("image_bytes", None)
        return row


# ──────────────────────────────────────────────────────────
# STAGE 6 — GPU EMBED  (GPU, actor pool, num_gpus=1)
# ──────────────────────────────────────────────────────────
# Pure inference. No PIL, no tokenizer, no Python transforms — just two
# forward passes on pre-tensorized inputs. Loads model ONCE per actor on init.
#
#   - Image tower: pixel_values → get_image_features → 1152-dim vector
#   - Text  tower: input_ids    → get_text_features  → 1152-dim vector
#     (SigLIP doesn't take an attention_mask — fixed-length padded inputs)
#
# Both vectors L2-normalized so cosine similarity = dot product downstream.

class EmbedSigLIP:
    """GPU forward-pass actor. Pure inference."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16: SigLIP 2 trains and ships in bf16. ~1.8 GB weights.
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        ).to(self.device).eval()
        print(f"  [embed] loaded {model_id} on {self.device} (bf16)")

    def __call__(self, batch: dict) -> dict:
        torch = self._torch

        pixel_values = torch.tensor(
            np.stack(batch["pixel_values"]),
            device=self.device,
            dtype=torch.bfloat16,
        )
        input_ids = torch.tensor(np.stack(batch["input_ids"]), device=self.device)

        with torch.inference_mode():
            img_out = self.model.get_image_features(pixel_values=pixel_values)
            txt_out = self.model.get_text_features(input_ids=input_ids)

        # SigLIP2 returns BaseModelOutputWithPooling on some transformers
        # versions; older SigLIP returned a tensor. Unwrap defensively.
        if hasattr(img_out, "pooler_output"):
            img_out = img_out.pooler_output
        if hasattr(txt_out, "pooler_output"):
            txt_out = txt_out.pooler_output

        img_feat = torch.nn.functional.normalize(img_out, dim=-1).float().cpu().numpy()
        txt_feat = torch.nn.functional.normalize(txt_out, dim=-1).float().cpu().numpy()

        batch["image_embedding"] = list(img_feat)
        batch["text_embedding"] = list(txt_feat)
        # Drop intermediate tensors — we don't want them in the output parquet.
        for k in ("pixel_values", "input_ids"):
            batch.pop(k, None)
        return batch


# ──────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────

def main():
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "worker_process_setup_hook": "src._vllm_compat.patch",
        },
    )
    print("Cluster resources:", json.dumps(ray.cluster_resources(), indent=2))

    # STAGE 1+2 — load + preprocess
    # NB: CheckpointConfig is set AFTER build_catalog (further down) on
    # purpose. Setting it before would cause build_catalog's write_parquet to
    # populate the checkpoint with all catalog row IDs — and then the
    # downstream inference pipeline would think those IDs are already done
    # and skip 99%+ of the rows. (Caught a 1-row-out-of-9998 run this way.)
    # The catalog has its own resume mechanism: the cache parquet itself.
    ds = build_catalog()

    # Job-level checkpointing for the inference pipeline ONLY — resumes after
    # worker death / cancel-resubmit. https://docs.anyscale.com/runtime/data
    ctx = ray.data.DataContext.get_current()
    ctx.checkpoint_config = CheckpointConfig(
        id_column="id",
        checkpoint_path=CHECKPOINT_PATH,
        delete_checkpoint_on_success=False,
    )

    # STAGE 3 — CPU fetch + decode + resize
    print(f"\n[fetch+resize] CPU pool, concurrency={FETCH_CONCURRENCY}, target {IMAGE_RESIZE}px")
    ds = ds.map_batches(
        fetch_and_resize,
        batch_size=16,
        concurrency=FETCH_CONCURRENCY,
        batch_format="numpy",
    )

    # STAGE 4 — VLM enrichment via ray.data.llm
    print(f"\n[vlm enrich] GPU pool, concurrency={VLM_CONCURRENCY}, model={VLM_MODEL_SOURCE}")
    ds = build_vlm_processor()(ds)

    # STAGE 5 — CPU process (SigLIP image processor + text tokenizer)
    print(f"\n[siglip process] CPU pool, concurrency={PROCESS_CONCURRENCY}, model={EMB_MODEL_SOURCE}")
    ds = ds.map(
        ProcessSigLIP,
        fn_constructor_kwargs={"model_id": EMB_MODEL_SOURCE},
        num_cpus=1,
        concurrency=PROCESS_CONCURRENCY,
    )

    # STAGE 6 — GPU embed (pure inference)
    print(f"\n[siglip embed] GPU pool, concurrency={EMB_CONCURRENCY}, batch={EMB_BATCH_SIZE}")
    ds = ds.map_batches(
        EmbedSigLIP,
        fn_constructor_kwargs={"model_id": EMB_MODEL_SOURCE},
        batch_size=EMB_BATCH_SIZE,
        num_gpus=1,
        concurrency=EMB_CONCURRENCY,
        batch_format="numpy",
    )

    # STAGE 7 — write parquet (the sink that triggers checkpointing)
    # materialize() pins the result blocks in the object store so both
    # write_parquet and the preview .take() below consume the same in-memory
    # dataset — no disk round-trip, no re-execution of the GPU pipeline.
    # https://docs.ray.io/en/latest/data/saving-data.html
    ds = ds.materialize()
    ds.write_parquet(OUTPUT_PATH)
    print(f"\n[done] wrote enriched + embedded catalog to {OUTPUT_PATH}")

    # ── PREVIEW — sample rows + in-driver retrieval demo ──
    # Stand-in for what FAISS / OpenSearch k-NN / Vespa do at scale.
    #
    # SigLIP raw-cosine regime (the model applies its own learned
    # logit_scale ≈ 100 / bias ≈ -10 internally, so raw cos clusters near 0):
    #   image → image  : relevant pairs at ~0.6+
    #   text  → image  : relevant pairs at ~0.10–0.20; <0.05 = noise
    print(f"\n[schema]\n{ds.schema()}")
    print(f"\n[count] {ds.count()} rows")
    print(f"\n[row 0]\n{ds.take(1)[0]}")
    print(f"\n[checkpoint] manifest at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────
# CHEAT SHEET — Where each stage runs
# ──────────────────────────────────────────────────────────
#
#  Stage                 Runs on    Scales via                      Bottleneck
#  ────────────────────  ─────────  ──────────────────────────────  ──────────
#  LOAD (read_parquet)   CPU        Ray Data block parallelism      HF CDN
#  PREPROCESS            CPU        block parallelism               cheap
#  URL_CHECK             CPU        block parallelism               net IO
#  CPU_FETCH             CPU pool   FETCH_CONCURRENCY               net IO
#  VLM_ENRICH            GPU pool   VLM_CONCURRENCY (TP×PP=1 each)  L4 FLOPs (heavy)
#  CPU_PROCESS           CPU pool   PROCESS_CONCURRENCY              CPU
#  GPU_EMBED             GPU pool   EMB_CONCURRENCY (1 GPU each)    L4 FLOPs (light)
#  WRITE (parquet)       CPU        block parallelism                disk
#
#  GPU footprint at defaults (g6.12xlarge, 4× L4 24 GB):
#    VLM_CONCURRENCY=2 × (TP=1, PP=1) → 2 L4s
#    EMB_CONCURRENCY=2 × 1 GPU       → 2 L4s
#    Total                            = 4 L4s ✓ (one node)
#
#  Multi-node scale-up:
#    - Bump VLM_CONCURRENCY / EMB_CONCURRENCY to (min, max) tuples for autoscale.
#    - Each GPU replica is independent; Ray Data shards blocks across them.
#    - GPU_EMBED is the cheaper stage (~thousands of items/sec/L4); usually
#      idles between VLM batches in the streaming executor — that's the
#      pipeline doing the right thing, not a bug. Don't optimize a stage
#      that's already idle.
#
#  Output schema (one row per product):
#    id, product_id, title, image_url, source,
#    raw_output                  ← VLM enrichment JSON string
#    image_embedding[1152]       ← SigLIP image tower (L2-normalized bf16→fp32)
#    text_embedding[1152]        ← SigLIP text tower  (L2-normalized bf16→fp32)
#
#  Drop-in alternatives you can swap with one line:
#    EMB_MODEL_SOURCE = "Marqo/marqo-ecommerce-embeddings-L"   # ecommerce-tuned
#    VLM_MODEL_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"          # better quality, more GPU
#
#  This is a BATCH pipeline. To expose both models as HTTP endpoints:
#    - Build a separate Serve app (e.g. scripts/run_vlm_online_enrich_emb_dual.py)
#    - Use ray.serve.llm.LLMConfig + build_openai_app for Qwen
#    - Add a Serve deployment for SigLIP that takes (image_url|image_bytes, text)
#      and returns {"image_embedding": [...], "text_embedding": [...]}
