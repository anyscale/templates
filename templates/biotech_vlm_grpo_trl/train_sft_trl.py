"""
Vanilla TRL SFT + Ray Train, LLM or VLM -- the end-to-end skeleton for PathAI's
Workload 1 (weeks 1-2: SFT on Ray Train), built on fake data so nothing waits on
their code or their JSON.

    MODE=vlm  Qwen3-VL-2B-Instruct on slide tiles + QA pairs   (default)
    MODE=llm  Qwen2.5-0.5B-Instruct on the text-only twin of the same QA

The training body is plain `SFTTrainer` + LoRA. Ray Train's only job is to start
N processes with torch.distributed set up and hand each one a GPU; accelerate inside
the Trainer sees the env vars and does DDP as usual. Every worker runs this same
function unchanged.

Launchers:

    python train_sft_trl.py                                 # 1 GPU
    accelerate launch --num_processes N train_sft_trl.py    # N-GPU DDP (PathAI today)
    python train_sft_trl.py --ray                           # same body under TorchTrainer

Data: JSONL produced by make_sft_data.py. The ONE place that knows the JSON shape is
`record_to_trl()`. Swap it when the real shape lands.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

MODE = os.environ.get("MODE", "vlm")
MODEL_ID = os.environ.get("MODEL", "Qwen/Qwen3-VL-2B-Instruct" if MODE == "vlm" else "Qwen/Qwen2.5-0.5B-Instruct")
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")
DATA_DIR = os.environ.get("SFT_DATA_DIR", "/mnt/cluster_storage/data/pathai_sft_fake")
RUN_NAME = os.environ.get("RUN_NAME", f"sft_{MODE}_fake")
RUN_ROOT = os.environ.get("RUN_ROOT", f"/mnt/cluster_storage/trl_sft/{RUN_NAME}")
HF_HOME = os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")

NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
PER_DEVICE_BS = int(os.environ.get("PER_DEVICE_BS", "4"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "1024"))
LR = float(os.environ.get("LR", "1e-4"))
LORA_R = int(os.environ.get("LORA_R", "16"))
EVAL_STEPS = int(os.environ.get("EVAL_STEPS", "10"))  # 0 = no eval
SEED = int(os.environ.get("SEED", "42"))

SYSTEM_PROMPT = "You are a pathology assistant. Answer questions about the tissue shown."


# ----------------------------------------------------------------------------------
# the JSON-shape seam
# ----------------------------------------------------------------------------------
def record_to_trl(rec: dict, data_dir: str, vlm: bool) -> dict:
    """
    One raw JSONL record  ->  one TRL prompt/completion row.

    TRL's SFTTrainer accepts a conversational prompt/completion pair natively and,
    with `images` present, uses its vision collator and masks the loss to the
    completion (`completion_only_loss` is inferred from the column names). The
    `{"type": "image"}` placeholder in the user turn is filled by TRL from `images`.

    Input shape assumed here is LLaVA-style (`conversations` with human/gpt turns
    and an `<image>` token in the human turn). Change THIS function for PathAI's shape.
    """
    turns = rec["conversations"]
    question = turns[0]["value"].replace("<image>", "").strip()
    answer = turns[1]["value"]
    if vlm:
        user_content = [{"type": "image"}, {"type": "text", "text": question}]
    else:
        user_content = [{"type": "text", "text": question}]
    row = {
        "prompt": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
    }
    if vlm:
        from PIL import Image

        row["images"] = [Image.open(os.path.join(data_dir, rec["image"])).convert("RGB")]
    return row


def build_dataset(jsonl_path: str, vlm: bool):
    from datasets import Dataset, Features, Image, List, Value

    data_dir = os.path.dirname(jsonl_path)
    with open(jsonl_path) as f:
        rows = [record_to_trl(json.loads(line), data_dir, vlm) for line in f if line.strip()]
    msg = List({"role": Value("string"), "content": List({"type": Value("string"), "text": Value("string")})})
    feats = {"prompt": msg, "completion": msg}
    if vlm:
        feats["images"] = List(Image())
    return Dataset.from_list(rows, features=Features(feats))


# ----------------------------------------------------------------------------------
# the train function: identical body for plain / accelerate / Ray Train
# ----------------------------------------------------------------------------------
def train_func(config: dict | None = None):
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    config = config or {}
    run_root = config.get("run_root", RUN_ROOT)
    os.environ.setdefault("HF_HOME", config.get("hf_home", HF_HOME))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    vlm = MODE == "vlm"
    suffix = "" if vlm else "_text"

    t0 = time.perf_counter()
    train_ds = build_dataset(os.path.join(DATA_DIR, f"train{suffix}.jsonl"), vlm)
    val_ds = build_dataset(os.path.join(DATA_DIR, f"val{suffix}.jsonl"), vlm) if EVAL_STEPS else None
    print(f"[data] mode={MODE} train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
          f"({time.perf_counter() - t0:.1f}s) columns={train_ds.column_names}")

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=2 * LORA_R,
        lora_dropout=0.05,
        # LM projections only; leaves Qwen3-VL's vision tower (qkv/proj/linear_fc*) frozen.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    report_to = ["wandb"] if os.environ.get("WANDB_API_KEY") else ["tensorboard"]
    args = SFTConfig(
        output_dir=os.path.join(run_root, "out"),
        run_name=RUN_NAME,
        logging_dir=os.path.join(run_root, "tensorboard"),
        report_to=report_to,
        per_device_train_batch_size=PER_DEVICE_BS,
        per_device_eval_batch_size=PER_DEVICE_BS,
        gradient_accumulation_steps=1,
        learning_rate=LR,
        max_steps=MAX_STEPS,
        max_length=MAX_LENGTH,
        packing=False,  # vision collator does not pack; keep the two modes identical
        bf16=True,
        model_init_kwargs={"dtype": "bfloat16", "revision": MODEL_REVISION},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        eval_strategy="steps" if EVAL_STEPS else "no",
        eval_steps=EVAL_STEPS or None,
        save_strategy="no",
        seed=SEED,
        dataloader_num_workers=0,
    )

    class RayReportCallback(TrainerCallback):
        """Forward every HF log dict to ray.train.report when inside a Ray Train worker.

        Ray Train V2 makes report() a barrier, so it must be called on every rank the
        same number of times; HF calls on_log on every rank, so this is symmetric."""

        def __init__(self):
            try:
                import ray.train

                ray.train.get_context().get_world_rank()
                self.ray_train = ray.train
            except Exception:
                self.ray_train = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.ray_train is None or not logs:
                return
            numeric = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            numeric["step"] = state.global_step
            self.ray_train.report(numeric)

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_config,
        callbacks=[RayReportCallback()],
    )
    trainer.model.print_trainable_parameters()
    print(f"[sft] completion_only_loss={trainer.completion_only_loss} vlm_collator={trainer._is_vlm}")

    trainer.train()

    if trainer.accelerator.is_main_process:
        path = os.path.join(run_root, "log_history.jsonl")
        with open(path, "w") as f:
            for rec in trainer.state.log_history:
                f.write(json.dumps(rec) + "\n")
        print(f"[done] wrote {len(trainer.state.log_history)} log records to {path}")
        # A qualitative check: greedy answer for one val example, before vs after is
        # visible by comparing with the dataset answer printed next to it.
        if val_ds is not None:
            _sample_generation(trainer, val_ds[0], vlm)


def _sample_generation(trainer, row: dict, vlm: bool) -> None:
    import torch
    from trl.data_utils import prepare_multimodal_messages

    proc = trainer.processing_class
    prompt = prepare_multimodal_messages(row["prompt"], images=row.get("images")) if vlm else row["prompt"]
    inputs = proc.apply_chat_template(
        conversation=[prompt], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(trainer.model.device)
    model = trainer.accelerator.unwrap_model(trainer.model)
    model.eval()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    text = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    print("[sample] question :", row["prompt"][-1]["content"][-1]["text"])
    print("[sample] reference:", row["completion"][0]["content"][0]["text"])
    print("[sample] model    :", text.strip())


# ----------------------------------------------------------------------------------
# Ray Train launcher
# ----------------------------------------------------------------------------------
def main_ray():
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.v2.api.callback import UserCallback

    class LossToDriver(UserCallback):
        def __init__(self, path: str):
            self.path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        def after_report(self, run_context, metrics, checkpoint):
            with open(self.path, "a") as f:
                f.write(json.dumps({"ranks": metrics}) + "\n")
            m = metrics[0] or {}
            keys = [k for k in ("loss", "eval_loss", "mean_token_accuracy", "eval_mean_token_accuracy", "learning_rate") if k in m]
            print(f"[ray.train.report] step={m.get('step')} " + " ".join(f"{k}={m[k]:.4g}" for k in keys), flush=True)

    env_vars = {"HF_HOME": HF_HOME, "TOKENIZERS_PARALLELISM": "false", "MODE": MODE}
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    ray.init(runtime_env={"env_vars": env_vars, "excludes": [".venv", "out", "traces", "logs", "results"]})

    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = HF_HOME
    snapshot_download(MODEL_ID, revision=MODEL_REVISION)

    os.makedirs(RUN_ROOT, exist_ok=True)
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"run_root": RUN_ROOT, "hf_home": HF_HOME},
        scaling_config=ScalingConfig(num_workers=NUM_GPUS, use_gpu=True),
        run_config=RunConfig(
            name=RUN_NAME,
            storage_path=os.path.join(os.path.dirname(RUN_ROOT.rstrip("/")), "ray_results"),
            callbacks=[LossToDriver(os.path.join(RUN_ROOT, "ray_reported_metrics.jsonl"))],
        ),
    )
    print("[ray] result:", trainer.fit())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray", action="store_true")
    cli = parser.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main_ray() if cli.ray else train_func()
