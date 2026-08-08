"""
Stage 1 — Teacher batch enrichment with Qwen2.5-VL-7B.

Reads an Amazon-Reviews-2023 product subset, prompts Qwen2.5-VL-7B with the
image + title, and writes a parquet of {category, attributes, search_tags,
description} JSON per row. The output is the labeled corpus that Stage 2
(run_distill_student_lora.py) distills into a 3B student.

Pipeline:
  LOAD → PREPROCESS → PREPARE_IMAGES → INFER (vLLM) → POSTPROCESS → WRITE

Each replica runs on a single L4 GPU (TP=1). With CONCURRENCY=4 you saturate
all four GPUs on a g6.12xlarge node, which gives roughly 4× the throughput of
a single TP=4 replica without any all-reduce overhead.

Run directly on the workspace cluster:
  python scripts/run_teacher_batch_label.py

Or as an Anyscale Job:
  anyscale job submit --config-file job_config.yaml --env HF_TOKEN=$HF_TOKEN
"""

import os, sys, json, hashlib
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
CATEGORY = os.environ.get("CATEGORY", "Electronics")
N_ROWS = int(os.environ.get("N_ROWS", 10_000))
SEED = int(os.environ.get("SEED", 42))

# Qwen2.5-VL ships in 3B / 7B / 32B / 72B. 7B in bf16 (~14GB) fits on a
# single L4 (24GB), so we shard via replicas — one model copy per GPU —
# rather than tensor parallelism.
MODEL_SOURCE = "Qwen/Qwen2.5-VL-7B-Instruct"

# tensor_parallel_size: shard weights across N GPUs on ONE node (intra-node).
# pipeline_parallel_size: split layers across M nodes (inter-node).
# Each replica claims TP × PP GPUs total. The 7B fits on one L4, so TP=1 +
# CONCURRENCY=4 beats TP=4 + CONCURRENCY=1: every GPU runs an independent
# replica every step, with no all-reduce overhead between them.
# https://docs.vllm.ai/en/stable/serving/distributed_serving.html
TENSOR_PARALLEL_SIZE = 1
PIPELINE_PARALLEL_SIZE = 1

MAX_MODEL_LEN = 2048    # prompt + image tokens + generation; raise if truncating.
                         # With max_pixels capped (see mm_processor_kwargs below),
                         # image ≈ 512 tok + prompt ≈ 150 tok + gen ≤ 160 tok ≈ 822.
                         # 2048 leaves headroom; halves per-seq KV reservation vs 4096.
BATCH_SIZE = 16          # rows fed to each replica per step; smaller for VLMs
                         # because image tokens explode the per-row token count.
CONCURRENCY = 4          # one replica per L4 on a g6.12xlarge (4× L4) node.
                         # Set (min, max) for autoscaling, e.g. (1, 4).

BASE_DIR = "/mnt/cluster_storage/vlm-distillation-catalog-enrichment"   # per-user persistent NFS — survives cluster
                                  # teardown AND is private to the submitting user.
                                  # Use /mnt/cluster_storage/vlm-distillation-catalog-enrichment instead if teammates
                                  # in the same project need to read these outputs.
HF_PATH = f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw_meta_{CATEGORY}"
CACHE_PATH = f"{BASE_DIR}/catalog_{CATEGORY}_{N_ROWS}.parquet"
CHECKPOINT_PATH = f"{BASE_DIR}/teacher_7b_{N_ROWS}_checkpoint"
OUTPUT_PATH = f"{BASE_DIR}/teacher_7b_enriched_{N_ROWS}.parquet"


# ──────────────────────────────────────────────────────────
# STAGE 1 — LOAD + STAGE 2 — PREPROCESS  (CPU)
# ──────────────────────────────────────────────────────────
# Pulls Amazon-Reviews-2023 metadata from HF, drops rows missing title/image,
# normalizes one row per product, and caches as parquet on cluster storage.
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
    # Pre-fetch HEAD check — Amazon delists product images periodically, and
    # one 404 inside prepare_multimodal_stage aborts the whole run (the stage
    # has no row-level fault tolerance). Cheaper to drop here than retry-fight
    # vLLM. Parallelized via Ray Data block parallelism on the CPU pool.
    try:
        return requests.head(row["image_url"], timeout=5, allow_redirects=True).ok
    except Exception:
        return False


def _normalize_amazon_row_to_image(row):
    # Stable per-(product, image) ID: SHA-1 of title + image_url. Deterministic
    # across runs so the Ray Data CheckpointConfig can actually resume —
    # uuid.uuid4() regenerated fresh IDs each submission, making the checkpoint
    # match zero rows. Hashing on (title, image_url) instead of product_id keeps
    # the ID unique if we ever fan a single product out into one row per image.
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
    print(f"[load hf] Loading data from huggingface hub: {HF_PATH}...")
    ds = ray.data.read_parquet(
        HF_PATH,
        file_extensions=["parquet"],
        filesystem=HfFileSystem(),
    )
    print("[load hf] count:", ds.count())
    print("[load hf] original schema:", ds.schema())

    ds = ds.limit(N_ROWS)
    ds = ds.filter(_has_title_and_image)
    ds = ds.map(_normalize_amazon_row_to_image)
    ds = ds.filter(_url_is_reachable)
    ds = ds.random_shuffle(seed=SEED)
    ds.write_parquet(CACHE_PATH)
    return ray.data.read_parquet(CACHE_PATH)


# ──────────────────────────────────────────────────────────
# STAGE 3 — VLM INFERENCE pipeline
# ──────────────────────────────────────────────────────────
# build_processor builds a multi-stage sub-pipeline under the hood:
#
#   PREPROCESS (CPU) →  PREPARE_IMAGES (CPU) → ChatTemplate (CPU)
#       → Tokenize (CPU) → vLLM Engine (GPU) → Detokenize (CPU) → POSTPROCESS (CPU)
#
# Each inner stage is its own auto-scaled actor pool. The PREPARE_IMAGES
# stage is the multimodal-specific one — it fetches every image_url and
# decodes to a PIL.Image so the GPU stage doesn't pay download latency.
# https://docs.ray.io/en/latest/data/working-with-llms.html
# https://docs.ray.io/en/latest/data/working-with-llms.html#multimodal

PROMPT = """\
You are a product catalog enrichment assistant. Given a product image and \
the merchant-supplied title, output a JSON object with exactly these keys:

  category:    one short string (e.g. "Wireless Earbuds")
  attributes:  a list of 3 short attribute strings
  search_tags: a list of 5 short search keywords
  description: a single sentence (<= 30 words)

Title: {title}

Return ONLY the JSON object, no commentary.\
"""


def build_messages_url(url, title):
    """OpenAI-spec multimodal message: image_url block + text block.

    Ray's prepare_multimodal_stage will fetch the URL on a CPU worker and
    replace the content block with the decoded image before it hits the GPU.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": PROMPT.format(title=title)},
            ],
        }
    ]


# ── PREPROCESS — (CPU, autoscaled actor pool) ──
# Builds the OpenAI-style messages + per-row sampling params. Keep this lean:
# anything you return travels through every downstream stage.
# https://docs.ray.io/en/latest/data/working-with-llms.html#preprocess
def vlm_preprocess(row):
    return {
        "id": row["id"],
        "messages": build_messages_url(row["image_url"], row["title"]),
        # max_tokens=160: output is a 4-key JSON (~120 tok typical, ~150 worst-
        # case). 160 is a safe ceiling and shrinks per-seq KV reservation vs 256.
        "sampling_params": {"max_tokens": 160, "temperature": 0.0},
    }


# ── POSTPROCESS — (CPU, autoscaled actor pool) ──
# Project to just the columns we want in the output parquet. Don't `**row`
# here — it'd carry vLLM-internal columns (prompt_token_ids, MediaWithBytes,
# generated_tokens, …) into the sink and bloat the file. Keep `id` so the
# checkpoint id_column lookup still works.
# https://docs.ray.io/en/latest/data/working-with-llms.html#postprocess
def vlm_postprocess(row):
    return {
        "id": row["id"],
        "product_id": row["product_id"],
        "title": row["title"],
        "image_url": row["image_url"],
        "source": row["source"],
        "raw_output": row["generated_text"],
    }


def run_inference(ds):
    # ── Job-level checkpointing (Anyscale-only feature) ──
    # Lets the job resume from where it left off if a worker dies or you
    # cancel + resubmit. id_column must be a stable string per row.
    # delete_checkpoint_on_success=False keeps the checkpoint after a clean
    # finish — handy during development; flip to True in production to free
    # the storage automatically.
    # https://docs.anyscale.com/runtime/data
    ctx = ray.data.DataContext.get_current()
    ctx.checkpoint_config = CheckpointConfig(
        id_column="id",
        checkpoint_path=CHECKPOINT_PATH,
        delete_checkpoint_on_success=False,
    )

    config = vLLMEngineProcessorConfig(
        model_source=MODEL_SOURCE,
        engine_kwargs={
            # TP shards the model across GPUs on ONE node (intra-node, NCCL).
            # PP splits LAYERS across nodes (inter-node, slower link OK).
            # Total GPUs per replica = TP × PP.
            "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
            "pipeline_parallel_size": PIPELINE_PARALLEL_SIZE,
            "max_model_len": MAX_MODEL_LEN,
            "trust_remote_code": True,
            # Chunked prefill: break long prefills (image tokens are huge —
            # ~1k+ per image for Qwen2.5-VL) into chunks that interleave
            # with ongoing decode steps in the same forward pass. Without
            # this, one image's prefill stalls every other request in the
            # running batch. Big throughput win for multimodal.
            "enable_chunked_prefill": True,
            # Per-step compute budget — total tokens vLLM processes in one
            # forward pass (prefill chunks + decodes combined). When chunked
            # prefill is on, this also caps the chunk size. 8192 is a good
            # middle ground for image-heavy prompts; raise for higher
            # throughput / lower for tighter per-step latency.
            "max_num_batched_tokens": 8192,
            # NB: continuous batching is the default vLLM scheduler — you're
            # already getting it (admit-on-arrival, finished-requests free
            # their slot mid-step). No flag needed.
            # Cap images per prompt so vLLM's mm scheduler can size buffers.
            # We send exactly 1 image per row in this pipeline.
            "limit_mm_per_prompt": {"image": 1},
            # Cap Qwen2.5-VL's dynamic-resolution image processor. Token count
            # ≈ pixels / (28×28). At Amazon's `large` image size (~1024²) the
            # processor emits ~1,300 vision tokens per row; capping at
            # 512×(28²) = 401,408 px (~633×633) bounds it to ~512 tokens →
            # roughly halves prefill work with no visible quality loss for
            # centered product photos. Floor of 256×(28²) avoids tiny
            # thumbnails getting upsampled into wasted tokens.
            "mm_processor_kwargs": {
                "min_pixels": 256 * 28 * 28,
                "max_pixels": 512 * 28 * 28,
            },
            # "should_continue_on_error": True,
            #   ↑ row-level fault tolerance: skip a bad image / corrupt
            #   prompt instead of failing the whole batch. Off here so dev
            #   bugs surface loudly; turn on for prod.
        },
        batch_size=BATCH_SIZE,
        concurrency=CONCURRENCY,
        # Multimodal-specific stage. Without this, the engine receives raw
        # image_url strings and the GPU actor would be doing HTTP fetches —
        # huge GPU-idle time. With it, fetch + decode is offloaded to CPU
        # workers and pipelined alongside inference.
        prepare_multimodal_stage={"enabled": True},
        # Setup hook re-run on every engine actor: vLLM 0.20+ moved
        # TokensPrompt; Ray 2.55's batch LLM stage still imports the old
        # path. Patch lives in src/_vllm_compat.py.
        runtime_env={"worker_process_setup_hook": "src._vllm_compat.patch"},
    )

    vlm_processor = build_processor(
        config,
        preprocess=vlm_preprocess,
        postprocess=vlm_postprocess,
    )
    return vlm_processor(ds)


# ──────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────

def main():
    # ray.init connects to the cluster Anyscale provisioned. We only add the
    # vllm setup hook here — the Job's runtime_env already supplies working_dir
    # (auto-uploaded from the yaml's `working_dir: .`) and env_vars (HF_TOKEN
    # from yaml). Passing those again here triggers a "Failed to merge runtime
    # env" conflict, since Ray won't merge overlapping keys.
    #
    # HF_TOKEN: NOT threaded through here. huggingface_hub / transformers /
    # datasets auto-pick from os.environ. Same pattern as the official
    # Megatron example:
    # https://github.com/anyscale/examples/blob/main/megatron_training/llm_sft_ray_train_megatron.py
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "worker_process_setup_hook": "src._vllm_compat.patch",
        },
    )
    print("Cluster resources:", json.dumps(ray.cluster_resources(), indent=2))

    ds = build_catalog()    # STAGE 1+2 — load + preprocess
    ds = run_inference(ds)  # STAGE 3   — multimodal inference

    # ── STAGE 4 — WRITE  (CPU, the "sink" that triggers checkpointing) ──
    # https://docs.ray.io/en/latest/data/saving-data.html
    ds.write_parquet(OUTPUT_PATH)
    print(f"[done] wrote enriched output to {OUTPUT_PATH}")

    # ── PREVIEW — print a few enriched rows to the job log ──
    # Reads from the just-written parquet (NOT the lazy `ds`) so we don't
    # re-execute the GPU pipeline for the preview. If the cluster's
    # /mnt/cluster_storage/vlm-distillation-catalog-enrichment gets blown away after the job, at least the
    # sample rows live forever in the captured job logs.
    print("\n[preview] sample enriched rows:")
    for row in ray.data.read_parquet(OUTPUT_PATH).take(limit=4):
        print(json.dumps(row, indent=2, default=str))

    print(ds.stats())


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────
# CHEAT SHEET — Where each stage runs
# ──────────────────────────────────────────────────────────
#
#  Stage                 Runs on    Scales via
#  ────────────────────  ─────────  ──────────────────────────
#  LOAD (read_parquet)   CPU        Ray Data block parallelism
#  PREPROCESS            CPU        Autoscaled actor pool
#  PREPARE_IMAGES        CPU  ┐     Multimodal-only: HTTP fetch + PIL decode.
#                                   Set prepare_multimodal_stage={"enabled":True}.
#  ChatTemplate          CPU  │
#  Tokenize              CPU  ├──   Built-in stages inside build_processor;
#  vLLM Engine           GPU  │     each is its own actor pool.
#  Detokenize            CPU  │
#  POSTPROCESS           CPU  ┘
#  WRITE (parquet)       CPU        Ray Data block parallelism
#
#  Sizing the engine replicas:
#    GPUs per replica = tensor_parallel_size × pipeline_parallel_size
#    total replicas   = concurrency  (or autoscales between (min, max))
#    total GPUs used  = (TP × PP) × concurrency
#
#  This script: TP=1, PP=1, concurrency=4 → 4 GPUs total → fits on one
#  g6.12xlarge (4× L4), one 7B replica per GPU.
#
#  Common scale-ups:
#    7B-8B on 1 node:          TP=1 PP=1, concurrency=N (one replica per GPU)
#    32B-70B on 1 node:        TP=4 PP=1, concurrency=1 (or (1, num_nodes))
#    70B+ multi-node:          TP=4 PP=2, concurrency=1 (one replica spans 2 nodes)
