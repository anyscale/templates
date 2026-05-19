"""
Ray Train — Qwen2.5-VL-3B Catalog Enrichment SFT (FSDP + LoRA)

Distills the 32B teacher's enrichment JSON into the 3B student so you ship
~32B-quality structured catalog output at 3B inference cost. Reads the
parquet that scripts/run_vlm_batch_enrich_32b.py wrote and uses each row's
`raw_output` (the teacher's JSON string) as the SFT target.

This closes the third loop of the demo arc:
  1) Batch enrichment      (run_enc_vlm_batch_emb_enrich_3b.py)  ← inference
  2) Online search         (run_vlm_online_search_3b.py)         ← inference
  3) Post-training (SFT)   (this file)                           ← training

The output is a LoRA adapter that drops into either of the inference scripts
as a model swap — same Qwen2.5-VL-3B base, fine-tuned weights.

Pipeline shape:
  LOAD teacher parquet → CPU FETCH images + SPLIT → BUILD SFT EXAMPLES (CPU pool)
       → SFT TRAIN  (Ray Train + FSDP + LoRA on 4× L4)
       → WRITE adapter

Each comment block below says WHERE the stage runs (CPU pool, GPU pool, or
inside the Ray Train workers) and links the relevant docs.

Run on the workspace cluster (after the 32B teacher parquet exists):
  python scripts/run_finetune_vlm_enrichment_3b.py

Or as an Anyscale Job (re-uses the 4× L4 fleet from the 32B job config —
training fleet is the same shape):
  anyscale job submit --config-file vlm_32b_job_config.yaml \\
    --entrypoint "python scripts/run_finetune_vlm_enrichment_3b.py" \\
    --env HF_TOKEN=$HF_TOKEN

──────────────────────────────────────────────────────────
Why distill the 32B teacher into a 3B student
──────────────────────────────────────────────────────────
Same prompt, same 4-key JSON schema, but a fraction of the inference cost.
The student is bounded by teacher quality — it will not exceed 32B output —
but will close most of the gap on a focused task (structured JSON over a
narrow product category) at 3B model size. This is the same recipe Anthropic
and OpenAI use internally to ship Haiku-class models from Opus-class teachers,
applied to one customer's catalog.

The category-specialization is a feature, not a bug. The teacher parquet was
produced on `Electronics` (CATEGORY in run_vlm_batch_enrich_32b.py); the
fine-tuned adapter will be category-specialized. production-scale catalogs
deploy this as a fleet of per-category adapters served from one base model
checkpoint — clean operational pattern, cheap to retrain per category.

──────────────────────────────────────────────────────────
What's optimized for L4 24GB and the 4× L4 node shape
──────────────────────────────────────────────────────────
  1. LoRA on the LLM, vision tower frozen. Trainable params drop to ~1% of
     the model. The vision encoder is already strong for product photography
     out-of-the-box and freezing it sidesteps multimodal training instability.
  2. FSDP FULL_SHARD with bf16 mixed precision. 3B in bf16 is ~6 GB; sharded
     across 4 GPUs that's ~1.5 GB/GPU for params. Plenty of headroom for
     activations on L4 24 GB.
  3. Gradient checkpointing on the LLM. Halves activation memory for the
     long image-token prefix at the cost of one extra forward per step —
     standard VLM SFT recipe.
  4. Per-device batch size of 1 with grad accumulation. Image-token prefixes
     (~512 tokens at our pixel cap) make per-device batches expensive in
     activation memory; bs=1 + grad_accum 16 → effective batch of 64.
  5. Cached training parquet with image_bytes pre-fetched. Epochs 2..N read
     from cache and never go back to HTTP. Same trick as the inference
     scripts, applied to the SFT path.
  6. Loss masking on the user prefix. Standard SFT — only the assistant's
     JSON tokens contribute to the loss, image and prompt tokens are -100.
  7. Pixel cap matches the inference-time cap (max_pixels = 512·28²) so
     the fine-tuned weights see exactly the same visual feature distribution
     they will see at deployment.
"""

import os, sys, json, hashlib, io

import numpy as np
import ray
import requests
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.torch import TorchTrainer


# Repo root — so `src._vllm_compat` resolves on the driver and Ray workers.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ──────────────────────────────────────────────────────────
# Knobs — the only things you usually tune
# ──────────────────────────────────────────────────────────
# Teacher parquet always reads from the 10k file (the source of truth produced
# by stage 1). Training subset is N_ROWS — env-overridable so the same script
# + job config can submit a small smoke job (N_ROWS=1000, ~45 min) or the
# full run (N_ROWS=10000, ~6 hr) without code edits.
TEACHER_N_ROWS = 10_000
N_ROWS = int(os.environ.get("N_ROWS", TEACHER_N_ROWS))
SEED = int(os.environ.get("SEED", 42))

BASE_DIR = "/mnt/cluster_storage/vlm-distillation-catalog-enrichment"
TEACHER_PARQUET     = f"{BASE_DIR}/vlm_enriched_32b_{TEACHER_N_ROWS}.parquet"
SFT_CACHE_PATH      = f"{BASE_DIR}/sft_cache_{N_ROWS}.parquet"
ADAPTER_OUTPUT_DIR  = f"{BASE_DIR}/qwen25vl_3b_enrichment_lora_{N_ROWS}"
TRAIN_RUN_DIR       = f"{BASE_DIR}/qwen25vl_3b_enrichment_runs_{N_ROWS}"

STUDENT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# One Ray Train worker per L4 — matches the g6.12xlarge shape (4× L4 24GB).
NUM_WORKERS = 4

# SFT hyperparameters. LoRA tolerates a higher LR than full FT because only
# adapter params move; 1e-4 is the standard safe default for r=16.
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1   # bs=1 + grad accum keeps activation memory in check
# Eval batch can be larger than train: no backward → no activation tape, no
# grad checkpointing tax, no optimizer state. With seq=2048 + bf16 model on
# L4 24GB, bs=4 fits with room. Cuts val pass wall-clock by ~4x.
EVAL_PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 16       # effective batch = 1 × 4 workers × 16 = 64
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
MAX_SEQ_LEN = 1024          # ~512 visual + ~200 prompt + ≤160 target ≈ 870
                            # tokens worst-case; 1024 is a safe ceiling. Lower
                            # than batch-enrich-32b's 2048 max_model_len to
                            # halve the cross-entropy logits allocation
                            # (batch × seq × ~152K vocab × bf16) which OOMed
                            # the loss step on 4× L4. Inference (online
                            # search) still uses max_model_len=2048 — the
                            # model's positional encoding is unchanged, only
                            # the SFT context window is shorter.

# Mid-epoch checkpoint cadence. 0 = disabled (epoch-only checkpoints).
# Each step checkpoint costs one FSDP FULL_STATE_DICT gather + ~74MB write,
# roughly 5–10s of pause. At 50, that's ~3 step-ckpts per epoch on the 10k-row
# config (156 grad steps), <1% wall-clock overhead, and lets you resume
# closer to a crash than epoch boundaries allow.
SAVE_EVERY_N_STEPS = 50

# LoRA config. r=16 is the sweet spot for 3B-scale models on a focused task;
# bump to 32 for harder tasks, drop to 8 for compute-tight runs.
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",      # MLP
]

# Image processing — same caps the inference-time scripts use, so the model
# trains on exactly the visual feature distribution it will see at serve time.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28
IMAGE_RESIZE = 512          # pre-resize to a fixed square for stable batch shape

# Deterministic train/val/test split by SHA-1(id) — stable across re-runs.
TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
# remainder is test

# CPU FETCH (HTTP + PIL decode + resize). Network IO bound.
FETCH_TIMEOUT_S = 5.0
FETCH_CONCURRENCY = 16

# CPU SFT EXAMPLE BUILDER (chat template + processor). CPU bound.
BUILD_CONCURRENCY = 8


# ──────────────────────────────────────────────────────────
# STAGE 1 — BUILD SFT TRAINING CACHE  (CPU)
# ──────────────────────────────────────────────────────────
# Reads the 32B teacher parquet, drops rows whose teacher output isn't valid
# JSON (bad supervision), assigns a deterministic train/val/test split,
# fetches every image once, and writes a self-contained training parquet at
# SFT_CACHE_PATH. Once cached, every epoch reads from this file — no HTTP.
# https://docs.ray.io/en/latest/data/loading-data.html

def _split_bucket(row_id: str) -> str:
    """Deterministic split by hashing the stable row id."""
    h = int(hashlib.sha1(row_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if h < TRAIN_FRAC:
        return "train"
    if h < TRAIN_FRAC + VAL_FRAC:
        return "val"
    return "test"


def _strip_code_fence(text: str) -> str:
    """The 32B teacher sometimes wraps JSON in ```json ... ``` fences. Strip
    them so json.loads succeeds. Same recovery the serve app does."""
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:].lstrip("\n")
    return text.strip()


def _teacher_output_is_valid(row):
    """Drop rows where the teacher output isn't a complete JSON object with
    all four expected keys. Bad targets pollute the loss."""
    try:
        obj = json.loads(_strip_code_fence(row["raw_output"]))
        return all(k in obj for k in ("category", "attributes", "search_tags", "description"))
    except Exception:
        return False


def fetch_and_resize_images(batch):
    """One HTTP round-trip per image. Resize to a fixed square so all rows
    produce identically-shaped processor outputs downstream — that matters
    because Ray Data's `iter_batches(numpy)` stacks rows into a single array
    per column."""
    from PIL import Image
    keep = ("id", "product_id", "title", "image_url", "raw_output", "split")
    out = {k: [] for k in keep}
    out["image_bytes"] = []
    for i in range(len(batch["id"])):
        try:
            r = requests.get(
                batch["image_url"][i],
                timeout=FETCH_TIMEOUT_S,
                headers={"User-Agent": "anyscale-finetune/1.0"},
            )
            if r.status_code != 200:
                continue
            img = Image.open(io.BytesIO(r.content)).convert("RGB").resize(
                (IMAGE_RESIZE, IMAGE_RESIZE)
            )
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            jpeg = buf.getvalue()
        except Exception:
            continue
        for k in keep:
            out[k].append(batch[k][i])
        out["image_bytes"].append(jpeg)
    return out


def build_sft_cache():
    if os.path.exists(SFT_CACHE_PATH):
        cached = ray.data.read_parquet(SFT_CACHE_PATH)
        print(f"[cache] reusing {SFT_CACHE_PATH}")
        return cached

    if not os.path.exists(TEACHER_PARQUET):
        raise FileNotFoundError(
            f"Teacher parquet not found at {TEACHER_PARQUET!r}. Run "
            f"scripts/run_vlm_batch_enrich_32b.py first to produce it."
        )

    print(f"[cache] reading teacher parquet {TEACHER_PARQUET}")
    ds = ray.data.read_parquet(TEACHER_PARQUET)
    ds = ds.filter(_teacher_output_is_valid)
    if N_ROWS < TEACHER_N_ROWS:
        # Subset for smoke/quick runs. .limit() before image fetch so we don't
        # do 10× more network IO than we need.
        ds = ds.limit(N_ROWS)
        print(f"[cache] limiting to first {N_ROWS} rows (subset of {TEACHER_N_ROWS})")

    def _attach_split(row):
        row["split"] = _split_bucket(row["id"])
        return row

    ds = ds.map(_attach_split)
    ds = ds.map_batches(
        fetch_and_resize_images,
        batch_size=16,
        concurrency=FETCH_CONCURRENCY,
        batch_format="numpy",
    )
    ds.write_parquet(SFT_CACHE_PATH)

    cached = ray.data.read_parquet(SFT_CACHE_PATH)
    print(f"[cache] wrote {cached.count()} rows → {SFT_CACHE_PATH}")
    return cached


# ──────────────────────────────────────────────────────────
# STAGE 2 — BUILD SFT EXAMPLES  (CPU, autoscaled actor pool)
# ──────────────────────────────────────────────────────────
# One AutoProcessor instance per CPU actor. Each call returns a fully
# tokenized, padded SFT example with loss masked on the user prefix so only
# the assistant's JSON tokens contribute to the loss. Output shapes are
# deterministic — fixed image size + fixed max_length — which is what lets
# Ray Data's `iter_batches(numpy)` stack rows for the trainer.

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


class BuildSFTExample:
    """Tokenize one (image, title, teacher_json) triple into an SFT example.

    Output columns: input_ids, labels, attention_mask, pixel_values,
                    image_grid_thw, split

    Loss masking: tokens up to the end of the user prefix get label=-100 so
    they don't contribute to cross-entropy. Pad tokens are also masked.
    """

    def __init__(self, model_id: str, max_length: int,
                 min_pixels: int, max_pixels: int):
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_pixels, max_pixels=max_pixels
        )
        self.max_length = max_length
        # Qwen tokenizer has eos but not always pad — fall back to eos for
        # padding (with the attention mask doing the right thing).
        self.pad_id = (
            self.processor.tokenizer.pad_token_id
            or self.processor.tokenizer.eos_token_id
        )

    def __call__(self, row: dict) -> dict:
        from PIL import Image

        img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")

        user_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": VLM_PROMPT.format(title=row["title"])},
            ],
        }
        # Strip the teacher's code fences before training on the JSON. Means
        # the student learns to emit clean JSON, not fenced JSON — matches the
        # robust-parser the serve app wraps the output in anyway.
        target = _strip_code_fence(row["raw_output"])
        asst_msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": target}],
        }

        # User-only template + generation prompt so we know where the
        # assistant tokens begin (everything before that gets -100).
        user_text = self.processor.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )
        full_text = self.processor.apply_chat_template(
            [user_msg, asst_msg], tokenize=False
        )

        user_inputs = self.processor(
            text=[user_text], images=[img],
            padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        full_inputs = self.processor(
            text=[full_text], images=[img],
            padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )

        # Boundary = number of non-pad tokens in the user-only prompt.
        user_len = int((user_inputs["input_ids"][0] != self.pad_id).sum().item())

        input_ids = full_inputs["input_ids"][0]
        attention_mask = full_inputs["attention_mask"][0]
        labels = input_ids.clone()
        labels[:user_len] = -100                              # mask user prefix
        labels[input_ids == self.pad_id] = -100               # mask padding

        return {
            "input_ids":      input_ids.numpy(),
            "labels":         labels.numpy(),
            "attention_mask": attention_mask.numpy(),
            "pixel_values":   full_inputs["pixel_values"].numpy(),
            "image_grid_thw": full_inputs["image_grid_thw"][0].numpy(),
            "split":          row["split"],
        }


# ──────────────────────────────────────────────────────────
# STAGE 3 — TRAIN LOOP  (Ray Train worker, FSDP + LoRA)
# ──────────────────────────────────────────────────────────
# One process per GPU. Each worker loads the full Qwen2.5-VL-3B in bf16,
# applies LoRA via PEFT, FSDP-wraps the result, and runs SGD on the data
# shard Ray Train hands it. After training, rank 0 gathers the full state
# and saves the LoRA adapter (config + weights only) to ADAPTER_OUTPUT_DIR.
# https://docs.ray.io/en/latest/train/getting-started-pytorch.html
# https://huggingface.co/docs/peft/main/en/accelerate/fsdp

def train_loop_per_worker(config: dict):
    import functools
    import random
    import tempfile
    import torch
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision, ShardingStrategy,
        StateDictType, FullStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import (
        Qwen2_5_VLForConditionalGeneration,
        get_cosine_schedule_with_warmup,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLDecoderLayer,
    )
    from peft import LoraConfig, get_peft_model
    import ray.train as train

    # ── Distributed context ──
    rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()
    device = torch.device(f"cuda:{local_rank}")

    # ── Seed propagation ──
    # torch + cuda get a per-rank offset so dropout differs across ranks
    # (otherwise FSDP's regularization is undermined — every rank dropping
    # the same units is just a uniform scale-down). random/numpy stay
    # rank-uniform; they're consumed by CPU-side ops that should be
    # deterministic across ranks for the same logical sample.
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)

    if rank == 0:
        print(f"[train] world_size={world_size}, model={config['model_id']}, "
              f"effective_batch={config['per_device_bs'] * world_size * config['grad_accum']}, "
              f"seed={seed}")

    # ── Load student + apply LoRA (BEFORE FSDP wrap) ──
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config["model_id"], torch_dtype=torch.bfloat16,
    )

    # Freeze the vision encoder. Vision-tower weights are already strong on
    # product photography out-of-the-box; full-multimodal SFT is unstable for
    # short runs. The LLM-only LoRA path is the standard VLM SFT recipe.
    for p in model.visual.parameters():
        p.requires_grad = False

    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["lora_target_modules"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    if rank == 0:
        model.print_trainable_parameters()

    # FSDP's FlatParameter requires uniform dtype within each wrap unit, but
    # PEFT creates LoRA weights (lora_A, lora_B) in fp32 while the base is
    # bf16. Cast trainable adapter params to bf16 so each decoder block has
    # uniform dtype. Stable for r=16 SFT at this scale; if you need fp32
    # LoRA, switch FSDP to HSDP/no_shard or use a transformer_auto_wrap_policy
    # that pulls adapters into their own wrap unit.
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.float32:
            p.data = p.data.to(torch.bfloat16)

    # Required when using gradient checkpointing on a PEFT model: input grads
    # need to flow back through the frozen base layers.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable({"use_reentrant": False})

    # ── FSDP wrap ──
    # transformer_auto_wrap_policy keyed on Qwen2_5_VLDecoderLayer is the
    # PEFT-compatible policy: each decoder layer is its own FSDP unit while
    # lm_head, embed_tokens, and the (frozen) vision tower stay replicated.
    # Why not size_based: PEFT's tuner wrapper calls the base model via
    # `self.model.forward(...)` (peft/tuners/tuners_utils.py:330), bypassing
    # __call__ and therefore any FSDP pre-forward unshard hook installed on
    # an inner-LLM-level wrap. The symptom is a `size mismatch ... vec (N/2)`
    # error inside lm_head where N is the lm_head's full param count.
    # Wrapping at the decoder-layer level avoids that path entirely.
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2_5_VLDecoderLayer},
    )
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,   # required for PEFT param-name preservation
    )

    # ── Optimizer + cosine LR schedule ──
    optimizer = torch.optim.AdamW(
        [p for p in fsdp_model.parameters() if p.requires_grad],
        lr=config["lr"], weight_decay=config["weight_decay"],
    )

    steps_per_epoch = max(1, config["train_size"] // (
        config["per_device_bs"] * world_size * config["grad_accum"]
    ))
    total_steps = steps_per_epoch * config["num_epochs"]
    warmup_steps = max(1, int(total_steps * config["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    # ── Data shards (Ray Data autoshards across workers) ──
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    def _to_device(batch: dict) -> dict:
        # Each numpy column is shape (bs, ...). Squeeze pixel_values back to
        # (bs * num_patches_per_image, patch_dim) since the processor returned
        # (1, N, D) per row and we stacked bs of them.
        pv = np.stack(batch["pixel_values"])
        if pv.ndim == 4:
            pv = pv.reshape(-1, pv.shape[-1])
        return {
            "input_ids":      torch.tensor(np.stack(batch["input_ids"])).to(device),
            "labels":         torch.tensor(np.stack(batch["labels"])).to(device),
            "attention_mask": torch.tensor(np.stack(batch["attention_mask"])).to(device),
            "pixel_values":   torch.tensor(pv).to(device, dtype=torch.bfloat16),
            "image_grid_thw": torch.tensor(np.stack(batch["image_grid_thw"])).to(device),
        }

    # ── Helper: gather FSDP shards on rank 0 and report a Ray Train ckpt ──
    # Used both at mid-epoch step cadence and at end-of-epoch. Pulled out so
    # the two callsites stay identical (same gather + same report shape).
    def _save_checkpoint_and_report(metrics: dict, prefix: str):
        ckpt_dir = tempfile.mkdtemp(prefix=prefix)
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_cfg):
            full_state = fsdp_model.state_dict()
            if rank == 0:
                fsdp_model.module.save_pretrained(
                    ckpt_dir, state_dict=full_state, safe_serialization=True,
                )
        ckpt = train.Checkpoint.from_directory(ckpt_dir) if rank == 0 else None
        train.report(metrics=metrics, checkpoint=ckpt)

    # ── Train ──
    fsdp_model.train()
    global_step = 0
    save_every = int(config.get("save_every_n_steps", 0))
    # Carry-forward last val_loss into step-checkpoint metrics so retention
    # by score_attribute="val_loss" can rank step ckpts vs. epoch ckpts.
    # Epoch 0 step ckpts get inf — they'll lose to any post-validation ckpt,
    # which is the right preference (no val signal yet = lower confidence).
    last_val_loss = float("inf")
    for epoch in range(config["num_epochs"]):
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_shard.iter_batches(
            batch_size=config["per_device_bs"], batch_format="numpy",
        )):
            inputs = _to_device(batch)
            outputs = fsdp_model(**inputs)
            loss = outputs.loss / config["grad_accum"]
            loss.backward()

            if (step + 1) % config["grad_accum"] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % 10 == 0:
                    print(f"  step {global_step:>5}/{total_steps} "
                          f"loss={loss.item() * config['grad_accum']:.4f} "
                          f"lr={scheduler.get_last_lr()[0]:.2e}")

                # ── Mid-epoch step checkpoint ──
                # All ranks must enter the FSDP gather collective together;
                # `global_step` is identical on all ranks because Ray Data
                # iter_batches is in lockstep, so this branch fires uniformly.
                if save_every and global_step % save_every == 0:
                    step_train_loss = float(loss.item() * config["grad_accum"])
                    _save_checkpoint_and_report(
                        metrics={
                            "step": global_step,
                            "epoch": epoch,
                            "train_loss": step_train_loss,
                            "val_loss": last_val_loss,
                        },
                        prefix=f"adapter_step{global_step}_",
                    )
                    if rank == 0:
                        print(f"  [ckpt] step={global_step} "
                              f"train_loss={step_train_loss:.4f} "
                              f"(carrying val_loss={last_val_loss:.4f})")

        # ── Validation pass ──
        # Eval batch can be larger than train (no backward, no grad
        # checkpointing tax). val_loss accumulation is sample-weighted, not
        # batch-weighted, so the result is correct regardless of whether the
        # last batch is full — important once eval_bs > 1.
        fsdp_model.eval()
        val_loss_sum, val_count = 0.0, 0
        with torch.no_grad():
            for batch in val_shard.iter_batches(
                batch_size=int(config.get("eval_per_device_bs", config["per_device_bs"])),
                batch_format="numpy",
            ):
                inputs = _to_device(batch)
                outputs = fsdp_model(**inputs)
                bs = inputs["input_ids"].shape[0]
                val_loss_sum += outputs.loss.item() * bs   # un-mean per-batch
                val_count += bs                             # count samples

        # All-reduce so the reported val_loss is a true global mean across
        # ranks, not a per-rank shard mean. Without this, rank 0 and rank 1
        # report different val_loss values for the same epoch.
        val_stats = torch.tensor([val_loss_sum, float(val_count)], device=device)
        torch.distributed.all_reduce(val_stats, op=torch.distributed.ReduceOp.SUM)
        avg_val = (val_stats[0] / torch.clamp(val_stats[1], min=1.0)).item()
        last_val_loss = avg_val   # carry forward into the next epoch's step ckpts

        # Per-epoch checkpoint via the same helper used by the step path —
        # populates result.metrics + result.checkpoint and lets
        # CheckpointConfig(score_attribute="val_loss") keep the best epoch.
        _save_checkpoint_and_report(
            metrics={"epoch": epoch, "step": global_step, "val_loss": avg_val},
            prefix=f"adapter_epoch{epoch}_",
        )

        if rank == 0:
            print(f"[epoch {epoch}] val_loss = {avg_val:.4f}")
        fsdp_model.train()

    # ── Save LoRA adapter (rank 0 only) ──
    # FullStateDictConfig gathers the sharded params on rank 0 only — every
    # other rank gets an empty dict, so we only call save_pretrained there.
    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT, save_cfg):
        full_state = fsdp_model.state_dict()
        if rank == 0:
            peft_model = fsdp_model.module
            peft_model.save_pretrained(
                config["adapter_dir"],
                state_dict=full_state,        # PEFT auto-filters to LoRA weights
                safe_serialization=True,
            )
            print(f"[save] adapter written to {config['adapter_dir']}")


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
    print(f"[run] N_ROWS={N_ROWS} (teacher source has {TEACHER_N_ROWS}) "
          f"SEED={SEED} NUM_WORKERS={NUM_WORKERS} "
          f"effective_batch={PER_DEVICE_BATCH_SIZE * NUM_WORKERS * GRAD_ACCUM_STEPS}")
    print("Cluster resources:", json.dumps(ray.cluster_resources(), indent=2))

    # STAGE 1 — build / reuse the SFT cache (image bytes + split labels)
    cached = build_sft_cache()

    # Materialized counts so the schedule math in train_loop_per_worker is
    # right — we avoid calling .count() on a lazy view inside the trainer.
    train_size = cached.filter(lambda r: r["split"] == "train").count()
    val_size = cached.filter(lambda r: r["split"] == "val").count()
    test_size = cached.filter(lambda r: r["split"] == "test").count()
    print(f"[split] train={train_size} val={val_size} test={test_size}")

    # STAGE 2 — SFT example builder (one processor instance per CPU actor)
    common_kwargs = dict(
        fn_constructor_kwargs={
            "model_id": STUDENT_MODEL_ID,
            "max_length": MAX_SEQ_LEN,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        num_cpus=2,
        concurrency=BUILD_CONCURRENCY,
    )
    train_ds = cached.filter(lambda r: r["split"] == "train").map(BuildSFTExample, **common_kwargs)
    val_ds   = cached.filter(lambda r: r["split"] == "val").map(BuildSFTExample, **common_kwargs)

    # STAGE 3 — Ray Train + FSDP + LoRA
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "model_id": STUDENT_MODEL_ID,
            "lr": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "per_device_bs": PER_DEVICE_BATCH_SIZE,
            "grad_accum": GRAD_ACCUM_STEPS,
            "warmup_ratio": WARMUP_RATIO,
            "weight_decay": WEIGHT_DECAY,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "lora_target_modules": LORA_TARGET_MODULES,
            "train_size": train_size,
            "adapter_dir": ADAPTER_OUTPUT_DIR,
            "save_every_n_steps": SAVE_EVERY_N_STEPS,
            "eval_per_device_bs": EVAL_PER_DEVICE_BATCH_SIZE,
            "seed": SEED,
        },
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS,
            use_gpu=True,
            accelerator_type="L4",
            resources_per_worker={"GPU": 1, "CPU": 4},
        ),
        run_config=RunConfig(
            storage_path=TRAIN_RUN_DIR,
            # Keep top-3 by val_loss across both epoch and step checkpoints.
            # With NUM_EPOCHS=2 + SAVE_EVERY_N_STEPS=50 + ~156 steps/epoch
            # → ~6 step ckpts + 2 epoch ckpts = 8 total candidates; we retain
            # the 3 best. Drop num_to_keep if you're storage-tight.
            checkpoint_config=CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
        datasets={"train": train_ds, "val": val_ds},
    )
    result = trainer.fit()
    print(f"\n[done] last metrics: {result.metrics}")
    print(f"[done] best checkpoint: {result.checkpoint}")
    print(f"[done] LoRA adapter at: {ADAPTER_OUTPUT_DIR}")

    # ── PREVIEW — qualitative side-by-side on the held-out test split ──
    # The student outputs aren't generated here (that's an inference-time
    # concern); we just print what the teacher said for 3 test products so
    # you have a reference to compare against after loading the adapter into
    # run_enc_vlm_batch_emb_enrich_3b.py or the serve app.
    print("\n[preview] held-out test set — teacher targets:")
    for r in cached.filter(lambda r: r["split"] == "test").take(3):
        print(f"\n  title:    {r['title'][:90]}")
        print(f"  teacher:  {_strip_code_fence(r['raw_output'])[:240]}")

    print("\n[next] swap the adapter into the inference scripts:")
    print("  from peft import PeftModel")
    print("  base = Qwen2_5_VLForConditionalGeneration.from_pretrained(STUDENT_MODEL_ID)")
    print(f"  model = PeftModel.from_pretrained(base, '{ADAPTER_OUTPUT_DIR}')")


if __name__ == "__main__":
    main()


# ──────────────────────────────────────────────────────────
# CHEAT SHEET — Where each stage runs
# ──────────────────────────────────────────────────────────
#
#  Stage                 Runs on        Scales via                       Bottleneck
#  ────────────────────  ─────────────  ───────────────────────────────  ──────────
#  LOAD teacher parquet  CPU            Ray Data block parallelism       cheap
#  FILTER bad JSON       CPU            block parallelism                cheap
#  CPU FETCH images      CPU pool       FETCH_CONCURRENCY                net IO
#  WRITE sft cache       CPU            block parallelism                disk
#  BUILD SFT examples    CPU pool       BUILD_CONCURRENCY                processor (PIL + tok)
#  TRAIN LOOP            GPU pool       NUM_WORKERS (one Ray Train       L4 FLOPs (heavy)
#                                          worker per L4)
#  SAVE adapter          GPU rank 0     —                                cheap
#
#  GPU footprint at defaults (g6.12xlarge, 4× L4 24GB):
#    NUM_WORKERS=4 × num_gpus=1  → 4 L4s (one full node)
#
#  Memory math (per L4):
#    Qwen2.5-VL-3B params bf16, FULL_SHARD across 4 GPUs    ~1.5 GB
#    LoRA trainable params (~50M @ bf16)                    ~0.1 GB
#    AdamW state for LoRA (fp32)                            ~0.4 GB
#    Activations (bs=1, seq=2048, grad checkpoint)          ~6–8 GB
#    Vision encoder (frozen, full-replicated bf16)          ~1.2 GB
#    Total                                                  ~10 GB / 24 GB
#
#  Multi-node / multi-category scale-up:
#    - Bump NUM_WORKERS for more GPUs (one ScalingConfig knob).
#    - Per-category training: re-run with a different teacher parquet
#      (point TEACHER_PARQUET at vlm_enriched_32b_<Category>.parquet) and a
#      different ADAPTER_OUTPUT_DIR. One base model, one LoRA per category.
#
#  Output:
#    ADAPTER_OUTPUT_DIR/
#      adapter_config.json          ← LoraConfig (r, alpha, target_modules)
#      adapter_model.safetensors    ← LoRA weights only (~100 MB for r=16)
#
#  Loading the adapter at inference time:
#    from peft import PeftModel
#    base  = Qwen2_5_VLForConditionalGeneration.from_pretrained(STUDENT_MODEL_ID)
#    model = PeftModel.from_pretrained(base, ADAPTER_OUTPUT_DIR)
#
#  Or, with vLLM in the existing batch / serve pipelines, attach the LoRA
#  via engine_kwargs:
#    LLMConfig(
#      ...,
#      engine_kwargs={..., "enable_lora": True,
#                     "lora_modules": [{"name": "enrich",
#                                       "path": ADAPTER_OUTPUT_DIR}]},
#    )
