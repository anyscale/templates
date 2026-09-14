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

Configuration follows TRL's own scripts (trl/scripts/grpo.py): `TrlParser` over three
dataclasses -- `ScriptArguments` (ours: data, launcher), `GRPOConfig` (every training
knob) and `ModelConfig` (model id, revision, dtype, LoRA) -- filled from a YAML via
`--config` with CLI flags overriding. `configs/grpo_smoke.yaml` is the smoke test.

    python train_grpo_trl.py --config configs/grpo_smoke.yaml                 # 1 GPU
    accelerate launch --num_processes 4 train_grpo_trl.py --config ...        # DDP, PathAI's shape
    python train_grpo_trl.py --config configs/grpo_smoke.yaml --ray           # same body under Ray Train
    python train_grpo_trl.py --config ... --max_steps 30 --learning_rate 2e-5 # override anything

Under `--ray`, the driver forwards argv to every worker and each worker re-parses it,
exactly like `accelerate launch` starts N processes that each parse the same argv.
`GRPOConfig` must be built inside the worker (it initialises the distributed state).

On this workspace only `--ray` runs: the head node has no GPU, the 4x A10G worker is
Ray-managed. See run_trl.sh and job_grpo.yaml.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from dataclasses import dataclass, field


@dataclass
class ScriptArguments:
    """Ours. Everything about training itself lives in GRPOConfig / ModelConfig."""

    data_dir: str = field(
        default="/mnt/cluster_storage/data/nct_crc",
        metadata={"help": "Directory with the SkyRL arm's train.parquet (../biotech_vlm_grpo/nct_crc_dataset.py)."},
    )
    train_rows: int = field(default=0, metadata={"help": "Use only the first N train rows; 0 = all 1,998."})
    profile_every: int = field(
        default=0,
        metadata={"help": ">0: torch.profiler trace of one full step every N steps, rank 0 only. ~2 GB per trace."},
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
    from trl import GRPOConfig, ModelConfig, TrlParser

    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    return parser.parse_args_and_config(args=argv)


def parse_script_args_only(argv: list[str]) -> ScriptArguments:
    """Driver-side parse. GRPOConfig cannot be built on the CPU-only head (bf16 check)."""
    from trl import TrlParser

    parser = TrlParser((ScriptArguments,))
    (script_args, _remaining) = parser.parse_args_and_config(
        args=_resolve_config_path(argv), return_remaining_strings=True, fail_with_unknown_args=False
    )
    return script_args


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
def train_func(config: dict):
    # `config` must be a REQUIRED parameter: Ray Train passes train_loop_config only to a
    # function with one required positional; with a default it calls train_func() bare.
    argv = config["argv"]
    script_args, training_args, model_args = parse(argv)

    from trl import GRPOTrainer, get_peft_config

    from grpo_step_timing import instrument_grpo_trainer
    from rewards import format_reward, label_reward

    # Everything the run writes sits next to output_dir: <run_root>/{out,tensorboard,traces,...}
    run_root = os.path.dirname(training_args.output_dir.rstrip("/"))
    # transformers 5 removed TrainingArguments.logging_dir; the TensorBoard callback reads this instead.
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", os.path.join(run_root, "tensorboard"))

    t0 = time.perf_counter()
    train_ds = build_dataset(os.path.join(script_args.data_dir, "train.parquet"), limit=script_args.train_rows)
    print(f"[data] {len(train_ds)} train rows in {time.perf_counter() - t0:.1f}s; columns={train_ds.column_names}")

    # As in trl/scripts/grpo.py: the model is loaded by the trainer from ModelConfig.
    training_args.model_init_kwargs = {
        "revision": model_args.model_revision,
        "dtype": model_args.dtype,
        **({"attn_implementation": model_args.attn_implementation} if model_args.attn_implementation else {}),
    }

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[label_reward, format_reward],
        args=training_args,
        train_dataset=train_ds,
        peft_config=get_peft_config(model_args),
    )

    # ---- the one extra line ----
    wrapped = instrument_grpo_trainer(
        trainer, profile_every=script_args.profile_every, profile_dir=os.path.join(run_root, "traces")
    )
    print("instrumented:", wrapped)
    if hasattr(trainer.model, "print_trainable_parameters"):
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
def main_ray(argv: list[str], script_args: ScriptArguments):
    import ray
    from ray.train import RunConfig, ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train.v2.api.callback import UserCallback

    # The run root is needed driver-side (results path, reported-metrics file) without
    # building GRPOConfig here: read output_dir the cheap way, from the merged argv/YAML.
    run_root = os.path.dirname(_lookup(argv, "output_dir").rstrip("/"))
    run_name = _lookup(argv, "run_name", default=os.path.basename(run_root))

    class StepMetricsToDriver(UserCallback):
        """
        Runs on the Ray Train controller. Receives what every rank passed to
        ray.train.report() each step and appends the full dicts to a JSONL on shared
        storage. (Ray 2.56's Train dashboard shows run / worker state and system
        metrics, not user-reported metrics; this file and TensorBoard are how you see them.)
        """

        def __init__(self, path: str):
            self.path = path
            os.makedirs(os.path.dirname(path), exist_ok=True)

        def after_report(self, run_context, metrics, checkpoint):
            with open(self.path, "a") as f:
                f.write(json.dumps({"ranks": metrics}) + "\n")

    # HF_HOME must be in the process environment before transformers is imported, so it
    # is genuinely environmental: run_trl.sh / job_grpo.yaml set it, and we forward it.
    hf_home = os.environ.get("HF_HOME", "/mnt/cluster_storage/hf_cache")
    env_vars = {"HF_HOME": hf_home}
    if os.environ.get("HF_TOKEN"):
        env_vars["HF_TOKEN"] = os.environ["HF_TOKEN"]
    # With RAY_RUNTIME_ENV_HOOK=...uv_runtime_env_hook.hook set, ray.init() also adds
    # working_dir=cwd (this directory) and py_executable="uv run ..." so the workers
    # rebuild this exact uv environment. Same trick as the SkyRL arm.
    # No `excludes` here: under `ray job submit` the job already owns working_dir/excludes and Ray
    # refuses to merge the same field twice; Ray 2.56 skips .venv and __pycache__ by default anyway.
    ray.init(runtime_env={"env_vars": env_vars})

    # Pre-fetch the model on the driver so N workers do not race the Hub for it.
    from huggingface_hub import snapshot_download

    model_id = _lookup(argv, "model_name_or_path")
    revision = _lookup(argv, "model_revision", default="main")
    os.environ["HF_HOME"] = hf_home
    t0 = time.perf_counter()
    snapshot_download(model_id, revision=revision)
    print(f"[hub] {model_id}@{revision[:8]} cached in {hf_home} ({time.perf_counter() - t0:.0f}s)")

    os.makedirs(run_root, exist_ok=True)
    reported = os.path.join(run_root, "ray_reported_metrics.jsonl")
    trainer = TorchTrainer(
        train_func,
        train_loop_config={"argv": argv},
        scaling_config=ScalingConfig(num_workers=script_args.num_gpus, use_gpu=True),
        run_config=RunConfig(
            name=run_name,
            storage_path=os.path.join(os.path.dirname(run_root), "ray_results"),
            callbacks=[StepMetricsToDriver(reported)],
        ),
    )
    result = trainer.fit()
    print("[ray] result:", result)
    print_step_table(reported)


def _lookup(argv: list[str], key: str, default: str | None = None) -> str:
    """Value of `key` from the CLI (`--key v`) or, failing that, the `--config` YAML."""
    flag = f"--{key}"
    if flag in argv:
        return argv[argv.index(flag) + 1]
    if "--config" in argv:
        import yaml

        with open(_resolve_config_path(argv)[argv.index("--config") + 1]) as f:
            cfg = yaml.safe_load(f) or {}
        if key in cfg:
            return str(cfg[key])
    if default is None:
        raise SystemExit(f"{flag} is required (on the CLI or in the --config YAML)")
    return default


def print_step_table(path: str) -> None:
    """Per-step summary from what rank 0 passed to ray.train.report, in the driver log."""
    if not os.path.exists(path):
        return
    print(f"{'step':>4} {'step_s':>7} {'gen_s':>6} {'wait_s':>6} {'rwd_s':>6} {'fwd_s':>6} {'bwd_s':>6} "
          f"{'opt_s':>6} {'reward':>7} {'len':>5} {'pad%':>5} {'spread':>6}")
    for line in open(path):
        m = json.loads(line)["ranks"][0] or {}
        if "timing/step_s" not in m:
            continue
        g = lambda k, d=0.0: m.get(k, d)
        print(f"{int(g('step', -1)):>4} {g('timing/step_s'):>7.1f} {g('timing/generate_s'):>6.1f} "
              f"{g('timing/sync_wait_s'):>6.1f} {g('timing/reward_s'):>6.3f} {g('timing/forward_s'):>6.2f} "
              f"{g('timing/backward_s'):>6.2f} {g('timing/optimizer_s'):>6.2f} {g('reward', float('nan')):>7.3f} "
              f"{g('completions/mean_length', float('nan')):>5.0f} {100 * g('rollout/padding_waste_frac'):>5.1f} "
              f"{g('rollout/generate_s_rank_spread'):>6.1f}")


if __name__ == "__main__":
    # This file is run by path; make its directory importable for rewards / timing.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    argv = sys.argv[1:]
    script_args = parse_script_args_only(argv)
    if script_args.ray:
        main_ray(argv, script_args)
    else:
        train_func({"argv": argv})
