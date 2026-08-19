# Biotech VLM GRPO — build plan

PathAI POV spike: prove SkyRL can run GRPO on a model of PathAI's shape (reasoning
VLM, image inputs, LoRA later, custom reward models later). This directory is the
spike-code home for that. Read `context/pathai-skyrl-spike.md` first (the full
spec — phases, decisions, gates) and `context/grpo-step-anatomy.md` second (one
GRPO step worked through with PathAI-shaped tensors). Riff base for all code is
`examples/train/geometry3k/` in the SkyRL repo — copy schema/config keys from
the actual files there, not from prose in this doc.

Scope of this pass: **spike code only** — Phase 1 (dataset) + Phase 2 (env +
entrypoint + run script, rule reward, no LoRA) from the spike doc. Explicitly
not doing yet: Phase 3 (reward model), Phase 4 (LoRA + 8B), Phase 5 (stretch),
or full template registration (BUILD.yaml entry, compute configs, depset, test
block — this repo's `AGENTS.md`/`/template` skill own that if it's wanted later).

## Locked decisions (this session, 2026-08-19)

- **Workspace:** `expwrk_gjlpp3xqhhe7xe1y397uzl3qn8` ("skyrl-incalculable-numerous-bird"),
  cloud `cld_kvedZWag2qA8i5BjxUevf5i7`, project `prj_cz951f43jjdybtzkx1s5sjgz99`.
- **Compute:** head `m5.2xlarge` (CPU only), one worker `g5.12xlarge` = 4x A10G
  24GB, `max_nodes=1` → **4 GPUs total**, not the geometry3k default of 8.
  Chose to iterate at 4 GPUs rather than bump the compute config.
- **Image:** `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8` — confirmed fine
  as-is, no rebuild needed.
- **SkyRL repo:** already cloned in the workspace at `~/default/SkyRL`, on
  `main` @ `9719b4f7` (same commit as the local laptop checkout). Root
  `pyproject.toml` pins `vllm==0.26.0`.
- **The "clone vLLM + local `[tool.uv.sources]` override" step is STALE — skip
  it.** `docs/content/docs/tutorials/vision_language_rl.mdx` and the header
  comment in `run_geometry3k_lora.sh` both say the repo is pinned to
  `vllm==0.19.0` and needs vLLM commit `80b18230e` (dated 2026-04-18) via a
  local source override. That's out of date: the repo bumped to `vllm==0.26.0`
  in PR #1854 (merged 2026-08-13), a month after those docs were last touched
  (2026-07-05), and 0.26.0 is well past the April commit. Confirmed by reading
  `pyproject.toml` directly in both the local and workspace checkouts — no
  local vLLM clone/build needed.
- **Model:** start with `Qwen/Qwen3-VL-2B-Instruct` (fits comfortably in 24GB
  A10Gs). `Qwen/Qwen3-VL-8B-Instruct` + LoRA is Phase 4, later, matching
  `run_geometry3k_lora.sh`'s `lora.rank=32 alpha=32 lr=3.0e-5`.
- **Logger:** `trainer.logger=tensorboard`. No MLflow tracking server exists;
  MLflow would default to a local `./mlruns` file store anyway, but tensorboard
  is simpler here — it's already a base dependency (top-level in
  `pyproject.toml`, no extra needed), needs zero env vars, and writes to
  `./tensorboard_log` by default (override with `TENSORBOARD_DIR`).
- **HF_TOKEN:** wanted on the workspace, but `anyscale workspace_v2 update
  --env` requires the workspace to be **TERMINATED** first, and I didn't want
  to stop a running workspace without checking first. **Didn't set it.**
  Simplest fix that avoids touching workspace config at all: just prefix the
  run command, e.g. `HF_TOKEN=<token> bash run_pathvlm.sh`, or `export
  HF_TOKEN=...` once in your workspace terminal session before running.
  Worth checking the Qwen3-VL-2B-Instruct HF page first — it's very likely
  ungated, in which case this may be unnecessary.

## What to build

### 1. `nct_crc_dataset.py` — model on `geometry_3k_dataset.py`

- Source: `1aurent/NCT-CRC-HE` on HF (or the `DykeF/NCTCRCHE100K` mirror) — 9
  tissue classes. **Not yet verified:** the actual column names for image/label
  on that HF dataset. Load it and inspect (`datasets.load_dataset(...).features`)
  before writing the `map_fn` — the spike doc names the 9 classes and the
  reasoning for this dataset choice but doesn't cite the literal schema.
- Subsample 2,000 train / 200 val, class-balanced across the 9 classes.
- Same record shape as geometry3k: `prompt` is a list of chat turns, image(s)
  go in as base64 JPEG data URIs inside `image_url` content parts (see
  `_pil_to_data_uri` in `geometry_3k_dataset.py` — reuse verbatim),
  `env_class="nct_crc"`, `reward_spec={"method": "rule", "ground_truth": <class_name>}`.
- Prompt template — copy verbatim from the spike doc's "Prompt template"
  section (system: pathology assistant persona; user: `<image>` + the 9-class
  list + the `<answer>class_name</answer>` instruction).
- Gate: load 5 rows back, decode a data URI to a PIL image, eyeball one.

### 2. `env.py` — model on geometry3k's `env.py`, minus the tool machinery

- **Single-turn**: `done=True` after one `step()` call, no turn loop.
- Reward: `1.0` if `<answer>...</answer>` contents match `ground_truth`
  (case/whitespace-insensitive), `+0.2` format bonus for well-formed tags even
  when the answer is wrong, `0` otherwise. Re-read the spike doc's exact wording
  before implementing — check whether the bonus is additive on top of the 1.0
  correct case or only applies when wrong.
- Drop entirely: `TOOL_CALL_RE`, `SUPPORTED_TOOL_NAMES`, `_extract_tool_call`,
  `_build_tool_feedback`, and all the tool-call branches in `step()`. Keep the
  `<answer>` extraction + scoring shape, simplified to one step.
- Keep the `get_metrics`/`aggregate_metrics` pattern (accuracy).

### 3. `pathvlm_entrypoint.py` — model on `geometry3k_entrypoint.py`

Registers `env_class="nct_crc"` pointing at this directory's `env.py`. See the
open question below before assuming the geometry3k dotted-path registration
style just works here unmodified.

### 4. `run_pathvlm.sh` — model on `run_geometry3k.sh` (not the LoRA variant)

Deltas from the geometry3k defaults:

- `NUM_GPUS=4` (not 8)
- `trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct"` (not 8B)
- `generator.max_turns=1` (not 3 — PathAI shape is single-turn)
- `trainer.logger=tensorboard` (script currently defaults `LOGGER=console`;
  either override the default or hardcode tensorboard for this template)
- `environment.env_class=nct_crc`
- data paths point at wherever `nct_crc_dataset.py` writes `train.parquet`/`val.parquet`

Keep as-is: `trainer.algorithm.advantage_estimator=grpo`,
`generator.n_samples_per_prompt=4-5`, `generator.vision_language_generator=true`,
`trainer.remove_microbatch_padding=false`, `generator.batched=false`,
`trainer.strategy=fsdp`, `trainer.placement.colocate_all=true`.

Watch for OOM: 4x A10G 24GB is much tighter than geometry3k's usual 8-GPU
setup. May need to shrink `micro_forward_batch_size_per_gpu` /
`micro_train_batch_size_per_gpu` below geometry3k's `4`, and/or lower
`gpu_memory_utilization` / `max_model_len` / `train_batch_size` — the VLM
tutorial explicitly calls out calibrating these for memory-heavy VLM runs.

## Open question: cross-repo module import (verify before writing real code)

geometry3k's `env.py` lives *inside* the SkyRL repo
(`examples/train/geometry3k/`), so `register(entry_point=
"examples.train.geometry3k.env:Geometry3kEnv")` resolves as a dotted import
path against the SkyRL repo root. This template's `env.py` lives in a
**separate repo** (`templates/templates/biotech_vlm_grpo/`), so that exact
string won't resolve the same way. Traced so far:

- `skyrl_gym.envs.registration.register()` accepts `entry_point` as either a
  string (`"module:ClassName"`, resolved via `importlib.import_module`) **or a
  live class object** — see `make()` in
  `skyrl-gym/skyrl_gym/envs/registration.py`:
  `elif callable(env_spec.entry_point): env_creator = env_spec.entry_point`.
- SkyRL's `initialize_ray()` calls `ray.init(runtime_env={"env_vars":
  env_vars}, ...)` with **no `working_dir` key**
  (`skyrl/train/utils/utils.py`) — Ray isn't packaging/replicating a working
  directory to workers. Whatever makes `examples.train.geometry3k.env`
  importable today comes from the venv/PYTHONPATH context `uv run` sets up
  when invoked from the repo root, not from Ray's runtime_env.

Two ways to unblock, in preference order:

1. In `pathvlm_entrypoint.py`: `sys.path.insert(0,
   os.path.dirname(os.path.abspath(__file__)))`, then `from env import
   NctCrcEnv`, and register with the **class object directly**
   (`register(id="nct_crc", entry_point=NctCrcEnv)`) instead of a string. Only
   valid if env instantiation (`skyrl_gym.make()`) happens in the *same
   process* that called `register()` — true for the
   `@ray.remote(num_cpus=1)` entrypoint task itself, unconfirmed for whatever
   actually creates env instances during rollout. Verify with a smoke run
   before trusting it for real.
2. Fallback: run `uv run --project ~/default/SkyRL --isolated --extra fsdp ...
   python pathvlm_entrypoint.py` **from inside** the `biotech_vlm_grpo`
   directory (not from the SkyRL repo root) — uv's `--project` flag points it
   at SkyRL's `pyproject.toml`/lockfile for dependency resolution *without*
   changing the invoking shell's CWD, so a plain `import env` resolves via
   `sys.path[0]` (the script's own directory), the same way geometry3k's
   dotted path resolves via CWD=repo root today.

Try (1) first — mirrors upstream most closely, no CLI-flag juggling. Fall back
to (2) if `register()` and `make()` turn out to run in different processes.
**Verify this before writing the real dataset/env** — it's a 5-step smoke test
(register a trivial dummy env, run 1 step), exactly like the spike doc's own
Phase 0 gate already asks for before touching anything real.

## Not done yet / explicit non-goals for this pass

- Phase 3 (reward-model seam), Phase 4 (LoRA + 8B), Phase 5 (2-node stretch,
  long-context probe) — see spike doc for specifics when ready.
- Full template registration (`BUILD.yaml` entry, AWS/GCP compute configs,
  `job_config.yaml`, `requirements.txt` + compiled `python_depset.lock`, test
  block) — this repo's `AGENTS.md` requires all of that for anything that
  ships as a real Anyscale template. Use the `/template` skill for that later
  if the POV wants it.
