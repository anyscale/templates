"""
Vanilla TRL SFT + Ray Train, LLM or VLM -- the end-to-end skeleton for PathAI's
Workload 1 (weeks 1-2: SFT on Ray Train), built on fake data so nothing waits on
their code or their JSON.

    --config configs/sft_vlm_smoke.yaml    Qwen3-VL-2B-Instruct on slide tiles + QA pairs
    --config configs/sft_llm_smoke.yaml    Qwen2.5-0.5B-Instruct on the text-only twin

The training body is plain `SFTTrainer` + LoRA. Ray Train's only job is to start
N processes with torch.distributed set up and hand each one a GPU; accelerate inside
the Trainer sees the env vars and does DDP as usual. Every worker runs this same
function unchanged.

Configuration follows TRL's own scripts (trl/scripts/sft.py): `TrlParser` over
`ScriptArguments` (ours), `SFTConfig` and `ModelConfig`, filled from `--config <yaml>`
with CLI flags overriding.

    python train_sft_trl.py --config configs/sft_vlm_smoke.yaml               # 1 GPU
    accelerate launch --num_processes 4 train_sft_trl.py --config ...         # DDP (PathAI today)
    python train_sft_trl.py --config configs/sft_vlm_smoke.yaml --ray         # under TorchTrainer

Data: JSONL produced by make_sft_data.py. The ONE place that knows the JSON shape is
`record_to_trl()`. Swap it when the real shape lands.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

# Same system prompt as the GRPO arm (../biotech_vlm_grpo/nct_crc_dataset.py): SFT teaches the
# "reason from the image, then <answer>" contract that GRPO then rewards.
SYSTEM_PROMPT = "You are a pathology assistant. Examine the tissue patch and reason step by step before answering."


@dataclass
class ScriptArguments:
    data_dir: str = field(
        default="/mnt/cluster_storage/data/pathai_sft_fake",
        metadata={"help": "Directory with {train,val}[_text].jsonl and tiles/ from make_sft_data.py."},
    )
    mode: str = field(default="vlm", metadata={"help": "'vlm' (tiles + QA, image column) or 'llm' (text-only twin)."})
    sample_after_train: bool = field(
        default=True, metadata={"help": "Greedy-decode one val example on rank 0 after training and print it."}
    )
    ray: bool = field(default=False, metadata={"help": "Run the training function inside ray.train TorchTrainer."})
    num_gpus: int = field(default=4, metadata={"help": "Ray Train workers (one GPU each). Only with --ray."})


def _resolve_config_path(argv: list[str]) -> list[str]:
    """Make a relative --config path resolve against this file's directory.

    Ray Train workers chdir into the run's storage directory, so `configs/x.yaml` would
    otherwise be looked up in the wrong place. This directory is the uploaded working_dir
    on the worker, so the YAML is always next to this script."""
    argv = list(argv)
    if "--config" in argv:
        i = argv.index("--config") + 1
        if not os.path.isabs(argv[i]) and not os.path.exists(argv[i]):
            argv[i] = os.path.join(os.path.dirname(os.path.abspath(__file__)), argv[i])
    return argv


def parse(argv: list[str]):
    argv = _resolve_config_path(argv)
    from trl import ModelConfig, SFTConfig, TrlParser

    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    return parser.parse_args_and_config(args=argv)


def parse_script_args_only(argv: list[str]) -> ScriptArguments:
    """Driver-side parse. SFTConfig cannot be built on the CPU-only head (bf16 check)."""
    from trl import TrlParser

    parser = TrlParser((ScriptArguments,))
    (script_args, _remaining) = parser.parse_args_and_config(
        args=_resolve_config_path(argv), return_remaining_strings=True, fail_with_unknown_args=False
    )
    return script_args


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
        # Content-part lists: the processor's chat template renders the image placeholder.
        from PIL import Image

        return {
            "prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]},
            ],
            "completion": [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
            "images": [Image.open(os.path.join(data_dir, rec["image"])).convert("RGB")],
        }
    # Plain-string content: text-only chat templates (Qwen2.5) concatenate `content` as a str.
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def build_dataset(jsonl_path: str, vlm: bool):
    from datasets import Dataset, Features, Image, List, Value

    data_dir = os.path.dirname(jsonl_path)
    with open(jsonl_path) as f:
        rows = [record_to_trl(json.loads(line), data_dir, vlm) for line in f if line.strip()]
    if vlm:
        msg = List({"role": Value("string"), "content": List({"type": Value("string"), "text": Value("string")})})
        feats = {"prompt": msg, "completion": msg, "images": List(Image())}
    else:
        msg = List({"role": Value("string"), "content": Value("string")})
        feats = {"prompt": msg, "completion": msg}
    return Dataset.from_list(rows, features=Features(feats))


# ----------------------------------------------------------------------------------
# the train function: identical body for plain / accelerate / Ray Train
# ----------------------------------------------------------------------------------
def train_func(config: dict):
    # `config` must be a REQUIRED parameter: Ray Train passes train_loop_config only to a
    # function with one required positional; with a default it calls train_func() bare.
    argv = config["argv"]
    script_args, training_args, model_args = parse(argv)

    from transformers import TrainerCallback
    from trl import SFTTrainer, get_peft_config

    run_root = os.path.dirname(training_args.output_dir.rstrip("/"))
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(run_root, "tensorboard"))
    vlm = script_args.mode == "vlm"
    suffix = "" if vlm else "_text"
    do_eval = training_args.eval_strategy != "no"

    t0 = time.perf_counter()
    train_ds = build_dataset(os.path.join(script_args.data_dir, f"train{suffix}.jsonl"), vlm)
    val_ds = build_dataset(os.path.join(script_args.data_dir, f"val{suffix}.jsonl"), vlm) if do_eval else None
    print(f"[data] mode={script_args.mode} train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
          f"({time.perf_counter() - t0:.1f}s) columns={train_ds.column_names}")

    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "dtype": model_args.dtype,
        **({"attn_implementation": model_args.attn_implementation} if model_args.attn_implementation else {}),
    }

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
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=get_peft_config(model_args),
        callbacks=[RayReportCallback()],
    )
    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()
    print(f"[sft] completion_only_loss={trainer.completion_only_loss} vlm_collator={trainer._is_vlm}")

    trainer.train()

    if trainer.accelerator.is_main_process:
        path = os.path.join(run_root, "log_history.jsonl")
        with open(path, "w") as f:
            for rec in trainer.state.log_history:
                f.write(json.dumps(rec) + "\n")
        print(f"[done] wrote {len(trainer.state.log_history)} log records to {path}")
        # A qualitative check: greedy answer for one val example next to the reference.
        if script_args.sample_after_train and val_ds is not None:
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

    def _text(content):  # content-part list (VLM rows) or plain string (LLM rows)
        return content[-1]["text"] if isinstance(content, list) else content

    print("[sample] question :", _text(row["prompt"][-1]["content"]))
    print("[sample] reference:", _text(row["completion"][0]["content"]))
    print("[sample] model    :", text.strip())


# ----------------------------------------------------------------------------------
# Ray Train launcher
# ----------------------------------------------------------------------------------
def main_ray(argv: list[str], script_args: ScriptArguments):
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.v2.api.callback import UserCallback

    from train_grpo_trl import _lookup  # same helpers, same conventions

    run_root = os.path.dirname(_lookup(argv, "output_dir").rstrip("/"))
    run_name = _lookup(argv, "run_name", default=os.path.basename(run_root))

    class LossToDriver(UserCallback):
        def __init__(self, path: str):
            self.path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        def after_report(self, run_context, metrics, checkpoint):
            with open(self.path, "a") as f:
                f.write(json.dumps({"ranks": metrics}) + "\n")

    hf_home = os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")
    env_vars = {"HF_HOME": hf_home}
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    # No `excludes` here: under `ray job submit` the job already owns working_dir/excludes and Ray
    # refuses to merge the same field twice; Ray 2.56 skips .venv and __pycache__ by default anyway.
    ray.init(runtime_env={"env_vars": env_vars})

    from huggingface_hub import snapshot_download

    os.environ["HF_HOME"] = hf_home
    snapshot_download(_lookup(argv, "model_name_or_path"), revision=_lookup(argv, "model_revision", default="main"))

    os.makedirs(run_root, exist_ok=True)
    reported = os.path.join(run_root, "ray_reported_metrics.jsonl")
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"argv": argv},
        scaling_config=ScalingConfig(num_workers=script_args.num_gpus, use_gpu=True),
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.join(os.path.dirname(run_root), "ray_results"),
            callbacks=[LossToDriver(reported)],
        ),
    )
    print("[ray] result:", trainer.fit())
    print_loss_table(reported)


def print_loss_table(path: str) -> None:
    if not os.path.exists(path):
        return
    print(f"{'step':>4} {'loss':>7} {'tok_acc':>7} {'eval_loss':>9} {'eval_acc':>8}")
    for line in open(path):
        m = json.loads(line)["ranks"][0] or {}
        if "loss" in m:
            print(f"{int(m.get('step', -1)):>4} {m['loss']:>7.3f} {m.get('mean_token_accuracy', float('nan')):>7.3f}")
        elif "eval_loss" in m:
            print(f"{int(m.get('step', -1)):>4} {'':>7} {'':>7} {m['eval_loss']:>9.3f} "
                  f"{m.get('eval_mean_token_accuracy', float('nan')):>8.3f}")


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    argv = sys.argv[1:]
    script_args = parse_script_args_only(argv)
    if script_args.ray:
        main_ray(argv, script_args)
    else:
        train_func({"argv": argv})
