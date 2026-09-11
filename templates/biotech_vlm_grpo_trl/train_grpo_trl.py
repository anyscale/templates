"""
Vanilla TRL GRPO on NCT-CRC-HE with Qwen3-VL-2B-Instruct -- the "TRL arm" of the
PathAI spike, sitting next to the SkyRL arm in ../biotech_vlm_grpo.

What "vanilla" means here:
  * `GRPOTrainer` with `use_vllm=False`: rollouts come from HF `model.generate`
    inside the trainer, on the same weights that train. No weight sync, no engine.
  * PyTorch DDP via accelerate (whatever the launcher hands us), LoRA via peft.
  * The same data, prompt and reward as the SkyRL arm. The parquet written by
    ../biotech_vlm_grpo/nct_crc_dataset.py is read directly and reshaped in memory
    into TRL's conversational-VLM row format (see `to_trl_rows`).
  * One extra line: `instrument_grpo_trainer(trainer)` from grpo_step_timing.py, which
    breaks every optimizer step into rollout / reward / forward / backward / sync.

Launchers (the train function body is identical in all three):

    python train_grpo_trl.py                       # 1 GPU, plain process
    accelerate launch --num_processes N train_grpo_trl.py    # N-GPU DDP, PathAI's shape
    python train_grpo_trl.py --ray                 # same body inside ray.train TorchTrainer

On this workspace only `--ray` runs: the head node has no GPU, the 4x A10G worker is
Ray-managed. See run_trl.sh.

Everything is configured by environment variables (defaults below) so the run
script can override without argument plumbing.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time

# ----------------------------------------------------------------------------------
# configuration (env vars; every one has a default that runs the smoke test)
# ----------------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL", "Qwen/Qwen3-VL-2B-Instruct")
# Hub revision pinned at the time of writing (2026-09-11) so the two arms compare
# like with like. Override with MODEL_REVISION=main to float.
MODEL_REVISION = os.environ.get("MODEL_REVISION", "89644892e4d85e24eaac8bacfd4f463576704203")

DATA_DIR = os.environ.get("DATA_DIR", "/mnt/cluster_storage/data/nct_crc")
RUN_NAME = os.environ.get("RUN_NAME", "pathvlm_2b_trl")
RUN_ROOT = os.environ.get("RUN_ROOT", f"/mnt/cluster_storage/trl_grpo/{RUN_NAME}")
HF_HOME = os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")

NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "10"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "4"))
PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "4"))  # completions per GPU per step
MAX_COMPLETION = int(os.environ.get("MAX_COMPLETION", "384"))
LR = float(os.environ.get("LR", "1e-5"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
LORA_R = int(os.environ.get("LORA_R", "16"))
TRAIN_ROWS = int(os.environ.get("TRAIN_ROWS", "0"))  # 0 = all 1,998
PROFILE_EVERY = int(os.environ.get("PROFILE_EVERY", "5"))
SEED = int(os.environ.get("SEED", "42"))


def to_trl_rows(parquet_path: str, limit: int = 0) -> list[dict]:
    """
    SkyRL parquet row  ->  TRL conversational-VLM row.

    SkyRL stores the image *inside* the prompt as a base64 data URI content part.
    TRL wants the image out-of-band in an `images` column and a `{"type": "image"}`
    placeholder in the user turn (TRL fills the placeholder at rollout time). The
    prompt text is taken from the parquet verbatim so the two arms see identical
    prompts. `ground_truth` reaches the reward functions as a kwarg.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path, columns=["prompt", "reward_spec"])
    rows = []
    for prompt, spec in zip(table.column("prompt").to_pylist(), table.column("reward_spec").to_pylist()):
        messages, images = [], []
        for msg in prompt:
            content = msg["content"]
            if isinstance(content, str):  # arrow.json extension may come back as a string
                content = json.loads(content)
            parts = []
            for part in content:
                if isinstance(part, str):
                    part = json.loads(part)
                if part["type"] == "image_url":
                    b64 = part["image_url"]["url"].split(",", 1)[1]
                    images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
                    parts.append({"type": "image"})
                elif part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
            messages.append({"role": msg["role"], "content": parts})
        rows.append({"prompt": messages, "images": images, "ground_truth": spec["ground_truth"]})
        if limit and len(rows) >= limit:
            break
    return rows


def build_dataset(parquet_path: str, limit: int = 0):
    from datasets import Dataset, Features, Image, List, Value

    rows = to_trl_rows(parquet_path, limit)
    features = Features(
        {
            "prompt": List(
                {
                    "role": Value("string"),
                    # image parts have text=None; TRL keys on "type" so the extra key is harmless
                    "content": List({"type": Value("string"), "text": Value("string")}),
                }
            ),
            "images": List(Image()),
            "ground_truth": Value("string"),
        }
    )
    return Dataset.from_list(rows, features=features)


# ----------------------------------------------------------------------------------
# the train function: identical body for plain / accelerate / Ray Train
# ----------------------------------------------------------------------------------
def train_func(config: dict | None = None):
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from grpo_step_timing import instrument_grpo_trainer
    from rewards import format_reward, label_reward

    config = config or {}
    run_root = config.get("run_root", RUN_ROOT)
    os.environ.setdefault("HF_HOME", config.get("hf_home", HF_HOME))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # transformers 5 removed TrainingArguments.logging_dir; the TensorBoard callback reads this instead
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(run_root, "tensorboard"))

    t0 = time.perf_counter()
    train_ds = build_dataset(os.path.join(DATA_DIR, "train.parquet"), limit=TRAIN_ROWS)
    print(f"[data] {len(train_ds)} train rows in {time.perf_counter() - t0:.1f}s; columns={train_ds.column_names}")

    # LoRA on the language model's projections only. Qwen3-VL's vision tower uses
    # different module names (qkv / proj / linear_fc1 / linear_fc2), so this list
    # leaves it frozen. `target_modules="all-linear"` would adapt the ViT too.
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=2 * LORA_R,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else ["tensorboard"]
    args = GRPOConfig(
        output_dir=os.path.join(run_root, "out"),
        run_name=RUN_NAME,
        report_to=report_to,
        # --- the GRPO shape, matching the SkyRL arm where the knobs correspond ---
        use_vllm=False,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=1,
        max_completion_length=MAX_COMPLETION,
        temperature=TEMPERATURE,
        beta=0.0,  # no reference model, no KL (SkyRL arm: use_kl_loss=false)
        learning_rate=LR,
        max_steps=MAX_STEPS,
        # --- plumbing ---
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "revision": MODEL_REVISION},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=2,
        save_strategy="no",
        seed=SEED,
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=MODEL_ID,
        reward_funcs=[label_reward, format_reward],
        args=args,
        train_dataset=train_ds,
        peft_config=peft_config,
    )

    # ---- the one extra line ----
    wrapped = instrument_grpo_trainer(
        trainer, profile_every=PROFILE_EVERY, profile_dir=os.path.join(run_root, "traces")
    )
    print("instrumented:", wrapped)
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Persist the full per-step log so the run can be reviewed without W&B/TB.
    if trainer.accelerator.is_main_process:
        path = os.path.join(run_root, "log_history.jsonl")
        with open(path, "w") as f:
            for rec in trainer.state.log_history:
                f.write(json.dumps(rec) + "\n")
        print(f"[done] wrote {len(trainer.state.log_history)} log records to {path}")


# ----------------------------------------------------------------------------------
# Ray Train launcher (the "OSS Ray arm"): same body, TorchTrainer does DDP setup
# ----------------------------------------------------------------------------------
def main_ray():
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.v2.api.callback import UserCallback

    class StepMetricsToDriver(UserCallback):
        """
        Runs on the Ray Train controller. Receives what every rank passed to
        ray.train.report() each step, prints a one-line summary and appends the full
        dicts to a JSONL on shared storage. (Ray 2.56's Train dashboard shows run /
        worker state and system metrics, not user-reported metrics; this is how you
        see them.)
        """

        def __init__(self, path: str):
            self.path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        def after_report(self, run_context, metrics, checkpoint):
            with open(self.path, "a") as f:
                f.write(json.dumps({"ranks": metrics}) + "\n")
            m = metrics[0] or {}
            if "timing/step_s" in m:
                print(
                    f"[ray.train.report] step={int(m.get('step', -1))} "
                    f"step_s={m['timing/step_s']:.1f} gen={m.get('timing/generate_s', 0):.1f} "
                    f"sync_wait={m.get('timing/sync_wait_s', 0):.2f} reward={m.get('timing/reward_s', 0):.2f} "
                    f"fwd={m.get('timing/forward_s', 0):.1f} bwd={m.get('timing/backward_s', 0):.1f} "
                    f"| reward={m.get('reward', float('nan')):.3f} "
                    f"len={m.get('completions/mean_length', float('nan')):.0f} "
                    f"spread={m.get('rollout/generate_s_rank_spread', 0):.2f}s",
                    flush=True,
                )

    env_vars = {"HF_HOME": HF_HOME, "TOKENIZERS_PARALLELISM": "false"}
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    # With RAY_RUNTIME_ENV_HOOK=...uv_runtime_env_hook.hook set, ray.init() also adds
    # working_dir=cwd (this directory) and py_executable="uv run ..." so the workers
    # rebuild this exact uv environment. Same trick as the SkyRL arm.
    ray.init(runtime_env={"env_vars": env_vars, "excludes": [".venv", "out", "traces", "logs", "results"]})

    # Pre-fetch the model on the driver so 4 workers do not race the Hub for it.
    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_HOME
    t0 = time.perf_counter()
    snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    print(f"[hub] {MODEL_ID}@{MODEL_REVISION[:8]} cached in {HF_HOME} ({time.perf_counter() - t0:.0f}s)")

    os.makedirs(RUN_ROOT, exist_ok=True)
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"run_root": RUN_ROOT, "hf_home": HF_HOME},
        scaling_config=ScalingConfig(num_workers=NUM_GPUS, use_gpu=True),
        run_config=RunConfig(
            name=RUN_NAME,
            storage_path=os.path.join(os.path.dirname(RUN_ROOT.rstrip("/")), "ray_results"),
            callbacks=[StepMetricsToDriver(os.path.join(RUN_ROOT, "ray_reported_metrics.jsonl"))],
        ),
    )
    result = trainer.fit()
    print("[ray] result:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray", action="store_true", help="run inside ray.train.torch.TorchTrainer")
    cli = parser.parse_args()
    # This file is run by path; make its directory importable for rewards / timing.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if cli.ray:
        main_ray()
    else:
        train_func()
