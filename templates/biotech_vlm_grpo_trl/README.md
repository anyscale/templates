# Biotech VLM GRPO — TRL arm (+ TRL SFT skeleton)

The PathAI-baseline counterpart to `../biotech_vlm_grpo`: the **same** NCT-CRC-HE
patches, prompt, and rule reward, but run through **vanilla Hugging Face TRL**
instead of SkyRL.

- `GRPOTrainer` with `use_vllm=False`: rollouts come from HF `model.generate`
  inside the trainer, on the weights being trained. No inference engine, no
  weight sync.
- DDP via accelerate, LoRA via peft, launched through `ray.train.torch.TorchTrainer`
  (the "OSS Ray arm"). The training function body is plain TRL; Ray only starts the
  processes.
- One extra line, `instrument_grpo_trainer(trainer)`, breaks every optimizer step
  into rollout / sync-wait / reward / forward / backward / optimizer and logs it
  next to TRL's own metrics. That stacked bar is the argument for a faster rollout
  engine later (vLLM inside TRL, or SkyRL).

Also in here, because the POV's Workload 1 starts with SFT: `train_sft_trl.py`, a
TRL `SFTTrainer` + Ray Train skeleton (LLM or VLM) on fake slide-tile QA data, so
there is runnable code before PathAI's data or JSON shape arrives.

## Run it

```bash
bash run_trl.sh                                   # GRPO, 4 GPUs, 10 steps (the smoke test)
NUM_GPUS=2 MAX_STEPS=30 MAX_COMPLETION=512 bash run_trl.sh
bash run_sft.sh                                   # SFT, VLM mode (Qwen3-VL-2B on tiles + QA)
MODE=llm bash run_sft.sh                          # SFT, text-only twin (Qwen2.5-0.5B-Instruct)
```

Both scripts source `~/.workspacerc` for `HF_TOKEN`, set Ray's uv runtime-env hook,
and launch from this directory. Outputs land on shared storage:

| what | where |
|---|---|
| per-step metrics (HF log stream, incl. `timing/*`) | `/mnt/cluster_storage/trl_grpo/<run>/log_history.jsonl` |
| what every rank passed to `ray.train.report` | `/mnt/cluster_storage/trl_grpo/<run>/ray_reported_metrics.jsonl` |
| TensorBoard | `tensorboard --logdir /mnt/cluster_storage/trl_grpo/<run>/tensorboard` |
| sampled completions per step (TRL `log_completions`) | `/mnt/cluster_storage/trl_grpo/<run>/out/completions/*.parquet` |
| torch.profiler traces (every `PROFILE_EVERY` steps) | `/mnt/cluster_storage/trl_grpo/<run>/traces/grpo_step<N>_rank<R>.json` (Perfetto) |
| Ray Train run state / worker logs | `/mnt/cluster_storage/trl_grpo/ray_results/<run>` and the Ray dashboard → Train |

Stacked bar of the phases per step:

```bash
uv run --frozen python plot_step_breakdown.py /mnt/cluster_storage/trl_grpo/pathvlm_2b_trl/log_history.jsonl step_breakdown.png
```

## Files

| File | What |
|---|---|
| `train_grpo_trl.py` | the GRPO script. Reads the SkyRL parquet, reshapes rows for TRL, `GRPOTrainer` + LoRA, `--ray` wraps the same body in `TorchTrainer` |
| `rewards.py` | `label_reward` (1.0) and `format_reward` (0.2), same scoring as `../biotech_vlm_grpo/env.py`; self-test with `python rewards.py` |
| `grpo_step_timing.py` | the instrumentation. Monkeypatches the trainer instance; returns a dict of what it managed to wrap |
| `plot_step_breakdown.py` | stacked bar from `log_history.jsonl` |
| `train_sft_trl.py` | `SFTTrainer` + LoRA, `MODE=vlm\|llm`, `--ray`; `record_to_trl()` is the one function that knows the JSON shape |
| `make_sft_data.py` | fake LLaVA-style JSONL: real 224px H&E tiles from the NCT-CRC val split + templated QA; text-only twin |
| `run_trl.sh`, `run_sft.sh` | launchers |
| `pyproject.toml`, `uv.lock` | the environment (ray 2.56.0, torch 2.11.0+cu128, trl 1.13.0, transformers 5.17.0, peft 0.20.0) |
| `results/` | distilled evidence from the smoke run (see below) |

## How the environment reaches the GPU worker

Same cluster facts as the SkyRL arm (`../biotech_vlm_grpo/ARCHITECTURE.md`): the head
node is CPU-only and its base conda has no torch; the 4x A10G worker autoscales on
demand. So this is a uv project, and `RAY_RUNTIME_ENV_HOOK=...uv_runtime_env_hook.hook`
makes `ray.init()` ship this directory as `working_dir` and `uv run --frozen` as the
workers' `py_executable`. Every Ray Train worker rebuilds the locked environment
(seconds when uv's cache is warm, a few minutes from cold). Everything the workers read
or write is on `/mnt/cluster_storage`; the model is pre-fetched into
`/mnt/cluster_storage/hf_cache` by the driver so four workers do not race the Hub.

## What the instrumentation wraps (TRL 1.13, `use_vllm=False`)

| metric | wrapped call | note |
|---|---|---|
| `timing/tokenize_s` | `_tokenize_prompts` | chat template + image preprocessing, CPU |
| `timing/generate_s` | `_generate_single_turn` | the pure HF `generate`. Not `_generate`: that one also runs a cross-rank `gather`, which would hide straggler wait inside generate time |
| `timing/generate/{prefill,decode}_s`, `ms_per_token` | a `LogitsProcessor` injected into `model.generate` | first decode step marks the end of prefill (incl. the ViT forward) |
| `timing/sync_wait_s` | `dist.barrier()` right after this rank's generate returns | how long fast ranks wait for the slowest rollout, measured *before* TRL's first gather |
| `timing/reward_s`, `timing/reward/<fn>_s` | `_calculate_rewards`, each reward func | includes TRL's gather of rewards across ranks |
| `timing/forward_s`, `timing/backward_s` | `compute_loss`, `accelerator.backward` | DDP all-reduce is inside backward |
| `timing/optimizer_s` | `optimizer.step` | wrapped lazily on the first step (the Trainer builds the optimizer inside `train()`) |
| `rollout/*` | completion ids seen by the reward stage | `padding_waste_frac` = share of decode work spent on padding shorter sequences up to the longest |
| `gpu/util_pct/<phase>`, `gpu/mem_peak_gb/<phase>` | NVML sampler thread + `torch.cuda.max_memory_allocated` | tagged by whichever phase is active |

Metrics are injected by wrapping `trainer.log`, so they reach TensorBoard/W&B and
`log_history` together with TRL's `reward`, `completions/mean_length`, etc. Inside a Ray
Train worker every rank also calls `ray.train.report(...)` with the same dict (Ray Train
V2 makes `report` a barrier, so it must be all ranks or none). A `UserCallback` on the
driver prints a one-line summary per step and appends every rank's dict to
`ray_reported_metrics.jsonl`.

**Ray Train dashboard caveat.** Ray 2.56's Train dashboard shows run and worker state,
logs, and system metrics (GPU util per worker). Its run schema has no field for
user-reported metrics, so `timing/*` does *not* appear as charts there; use
TensorBoard or the JSONL. `ray.train.report` is still the right hook for anything
driver-side (checkpoint bookkeeping, `Result.metrics`).

TRL has its own coarse timers (`profiling/Time taken: GRPOTrainer.<method>`), but they
log straight to W&B/MLflow/Trackio, not into the HF log stream, so with TensorBoard
you never see them. The keys above are a superset of what they cover.

## Known limits and choices

- **Not TRL's vLLM mode.** `use_vllm=True` (server or colocate) is the next arm; this
  one is deliberately the PathAI baseline shape.
- `beta=0`: no reference model, no KL, matching the SkyRL arm's `use_kl_loss=false`.
  `num_iterations=1`, so old log-probs are not recomputed; the step is generate →
  reward → one forward/backward.
- LoRA targets the language model's projections only; Qwen3-VL's vision tower uses
  different module names and stays frozen.
- `MAX_COMPLETION=384` vs. SkyRL's 1024: HF `generate` runs every sequence to the
  longest, so a long cap is paid on every step. The SkyRL run's mean completion was
  ~230 tokens; 384 truncates the tail and the `format_reward` teaches conciseness.
- Eval is not wired into the GRPO script (HF `generate` over 198 val rows × 4 GPUs is
  slow); the SFT script does eval every `EVAL_STEPS`.
- Not a registered Anyscale template (no `BUILD.yaml` entry, compute configs, depset
  lock, or test block). Use the `/template` skill if that is wanted.

## SFT skeleton notes

- Data shape is a *guess* (LLaVA-style `conversations` + `image` path + `metadata`),
  chosen because it is the most common VLM-SFT JSON in the wild. `record_to_trl()` is
  the only code that touches it.
- The TRL row format is conversational **prompt/completion** + `images`, so TRL infers
  `completion_only_loss=True` and uses its vision collator. (`assistant_only_loss` is
  not supported for vision datasets in TRL 1.13.)
- Tiles are real H&E from the NCT-CRC val split; the QA text is templated from the
  label. It proves the plumbing, not the model.
- `make_sft_data.py --source tcga` is a stub: real whole-slide images need
  `gdc-client` to fetch SVS files and `openslide-python` to tile them; neither is in
  this environment.
