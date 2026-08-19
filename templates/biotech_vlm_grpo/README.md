# Biotech VLM GRPO — PathAI POV spike

Single-turn GRPO on **NCT-CRC-HE** colorectal H&E patches with
`Qwen/Qwen3-VL-2B-Instruct`. The model sees one 224x224 tissue patch plus a
9-class list, reasons, and commits to `<answer>class_name</answer>`. Reward is a
rule: 1.0 for the right class, +0.2 for well-formed tags.

This is the PathAI shape — reasoning VLM, image inputs, rule reward now and a
custom reward *model* later — riffed off `examples/train/geometry3k/` in the
SkyRL repo.

- `ARCHITECTURE.md` — **what actually runs on the Ray cluster**, drawn from the live run
- `PLAN.md` — the build plan and locked decisions
- `RUNSHEET.md` — **findings, gaps and workarounds**; read this one
- `context/pathai-skyrl-spike.md` — the full spike spec (phases, gates)
- `context/grpo-step-anatomy.md` — one GRPO step with PathAI-shaped tensors

## Run it

```bash
bash run_pathvlm.sh
```

That is the whole thing. The script sources `~/.workspacerc` for `HF_TOKEN`,
sets the Ray/uv hook, copies the sources into the SkyRL repo (see below),
generates the dataset on first run, and launches training on 4 GPUs.

Useful overrides — env vars before the command, or SkyRL config keys after it:

```bash
NUM_GPUS=4 LOGGER=console bash run_pathvlm.sh
bash run_pathvlm.sh trainer.epochs=3 generator.n_samples_per_prompt=8
MODEL=Qwen/Qwen3-VL-8B-Instruct bash run_pathvlm.sh   # Phase 4
```

Watch it: `tensorboard --logdir tensorboard_log`.

## Files

| File | What |
|---|---|
| `nct_crc_dataset.py` | HF dataset -> SkyRL parquet, class-balanced 1,998 train / 198 val |
| `env.py` | `NctCrcEnv` — single-turn, `<answer>` parsing, rule reward |
| `pathvlm_entrypoint.py` | registers `env_class="nct_crc"`, runs `BasePPOExp` |
| `run_pathvlm.sh` | the 4-GPU launch config |

## Three things that are not obvious

1. **Nothing is copied into the SkyRL repo — but the launch directory is.**
   Ray's uv integration ships `os.getcwd()` as the runtime working_dir and
   rejects a `pyproject.toml` outside it, so you must launch from the SkyRL repo
   root. That means this directory is *not* on the worker's `sys.path`, and
   SkyRL's entrypoint task always runs on a worker. `pathvlm_entrypoint.py`
   solves it by registering the env class **by value** (cloudpickle) rather than
   by dotted import path, so the class definition travels inside the task. This
   directory stays the one and only home for the code.

2. **`max_model_len` must be capped explicitly.** Qwen3-VL advertises a 262k
   context; vLLM sizes its KV cache from that and won't start on a 24GB A10G.
   There is no first-class SkyRL field for it —
   `generator.inference_engine.engine_init_kwargs.max_model_len=4096` is the
   pass-through.

3. **Every path the trainer reads must be on `/mnt/cluster_storage`.**
   `/home/ray/default` is node-local, and the trainer does not run on the head
   node.

Full detail on all three, plus the dataset gotchas, in `RUNSHEET.md`.

## Not done yet

Phase 3 (custom reward model), Phase 4 (LoRA + 8B), Phase 5 (2-node, long
context). This is also **not** a registered Anyscale template — no `BUILD.yaml`
entry, compute configs, `job_config.yaml` or `python_depset.lock`. Use the
`/template` skill if the POV wants it shipped as one.
