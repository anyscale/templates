# SkyRL Spike — PathAI POV

**Owner:** Geoff · **Timebox:** 2 days, complete before Aug 31 kickoff (ideally pre-Summit)
**Single question:** can SkyRL run GRPO on a model of PathAI's shape (reasoning VLM, image inputs, LoRA, custom reward models) — or do we fall back?

---

## Why this spike exists

The POV's Workload 1 is now **sequential, not either/or** (decided in Slack, Aug 13):

- **Weeks 1–2:** SFT on **Ray Train** — minimal-refactor port of PathAI's existing HF/DDP code. Low value, low risk, proves the platform.
- **Weeks 2–5:** GRPO on **SkyRL** — the main event and the future-state buy-in (Pawarit: "make SkyRL a really important part of the eval").

The RL weeks are the 🔴 risk in the plan. SkyRL's VLM support is recent; if a gap surfaces in week 3 of a 6-week POV, the platform evaluation stalls for reasons unrelated to the platform. The spike moves that discovery to before day 1.

**Fallback ladder if the spike fights back:**
1. SkyRL clean → weeks 2–5 as planned
2. SkyRL gaps → **veRL** (proven Qwen-VL GRPO recipes; same "Ray-native RL" pitch, minor doc edit)
3. Floor → GRPO stays in TRL, wrapped in Ray Train + Ray actors

Gaps are still a useful result: written summary → SkyRL team (Sumanth & co), possible joint recipe, SkyRL becomes phase two.

---

## Part 1 — Orientation reading (mostly already paid down)

harbor-on-ray already covered SkyRL's config surface, trial/rollout consumption shape, and how it drives Ray actors. Skim to fill the training-side half:

| Read | Why |
|---|---|
| [Anyscale blog: VLM RL with SkyRL](https://www.anyscale.com/blog/vision-language-model-reinforcement-learning-skyrl) | The validated recipe shape Pawarit cited — this spike's template. Note which SkyRL commit/branch it uses |
| [SkyRL dataset prep docs](https://docs.skyrl.ai/docs/datasets/dataset-preparation) | The parquet schema every phase below depends on |
| [SkyRL new-env tutorial](https://docs.skyrl.ai/docs/tutorials/new_env) | The custom env/reward seam — Phase 2 and 3 are built on this |
| `skyrl-train` GRPO example configs (`examples/train/gsm8k/`) | Trainer/rollout topology, where FSDP + vLLM engine config live; the template runs exactly this |
| GRPO paper (DeepSeekMath §4) — skim | Enough to sanity-check group stats when validating output |
| veRL README + Qwen2.5-VL GRPO example | Know the fallback's shape before you need it |
| TRL `GRPOTrainer` docs | PathAI's current stack; you'll read their modules in week 1 |

Skip: multi-turn/agentic paths, Search-R1, harbor integration — irrelevant to single-turn diagnosis rollouts.

---

## Part 2 — The SFT vs RL split (the argument, for reference)

- **Ray Train + engine = the SFT tool.** Mature; SkyRL's SFT path is young (one production user, dataloading built to order). PathAI's SFT code exists and runs — port it, don't replace it.
- **SkyRL = the RL tool.** Ray Train has *nothing* for the RL loop: inference engines inside the training loop, NCCL weight sync after each policy update, GRPO/DAPO, async rollouts. Not a feature gap — a different problem.
- **The handoff is the hidden cost:** SFT on stack A + RL on stack B = checkpoint conversion **plus a second tokenization/chat-template implementation that must agree exactly**. Logprob-drift bug class at the stage boundary — the argument for SkyRL owning the RL stage outright.
- **Guardrail:** never compare SFT performance of Ray Train vs SkyRL. SFT throughput is measured on Ray Train only; SkyRL is measured on the RL stage, where Ray Train has no equivalent.

---

## Part 3 — Design & implementation plan

### Target demo shape

```
SkyRL GRPO · Qwen3-VL-2B → 8B · LoRA · pathology patch images in prompt
· rule reward first, then custom reward MODEL · vLLM rollouts colocated
  with FSDP trainer · 1× 8-GPU node · 2-node stretch
· riff base: examples/train/geometry3k (the official VLM RL example)
```

| Element | PathAI unknown it retires |
|---|---|
| Qwen3-VL through vLLM engines | VLM in the rollout path (image tensors in generation) |
| Multimodal batches → FSDP workers | Vision tower + LLM through the *training* side |
| LoRA (**confirmed shipped**: `run_geometry3k_lora.sh`) | adapter × weight-sync interaction, on OUR task shape |
| Custom reward **model** | Their 2 in-house reward models; the non-env scoring seam |
| Colocated rollout + train | The orchestration SkyRL exists for |
| Multi-image / long prompts (stretch) | WSI-derived visual context length |

**Verified facts (docs.skyrl.ai, VLM blog — Apr 2026):** VLM RL runs on the
**FSDP backend only** (no Megatron, no sample packing, no context parallelism —
all roadmap). Geometry3k example: Qwen3-VL-8B-Instruct, GRPO,
`n_samples_per_prompt=4`, multi-turn (3) with a `calc_score` tool. SkyRL routes
tokenization through **vLLM as source of truth** (render endpoint → token-in-
token-out) because HF-processor-side rendering caused measurable logprob drift
and reward collapse on VLMs — the blog's Figure 2 is the logprob-drift bug class,
observed. Good PathAI talking point.

### Decisions made for you

**Dataset: NCT-CRC-HE-100K** (`1aurent/NCT-CRC-HE` on HF, or the `DykeF/NCTCRCHE100K` mirror). 100K H&E-stained colorectal patches, 224×224, **9 tissue classes** — real pathology, verifiable label, small images. Subset: **2,000 train / 200 val**, class-balanced. Why not alternatives: PCam is binary (reward too easy, group advantages collapse to ±1 fast); generic VQA sets lose the pathology story when you demo this to Ryun. 9-class + CoT prompt gives a reward distribution with real variance inside a GRPO group, which is what you need to see the machinery work.

**Model: `Qwen/Qwen3-VL-2B-Instruct` first** — the family SkyRL actually validated (blog uses Qwen3-VL-2B for SFT, 8B-Instruct for geometry3k RL). Move to **8B in Phase 4** once the loop is proven. Qwen-VL-class is the most likely proxy for PathAI's base model anyway (confirm with Ryun — baseline-data ask already in the POV doc).

**Reward model for Phase 3: any small HF sequence-classification reward model** (e.g. `OpenAssistant/reward-model-deberta-v3-large-v2`, ~400M, CPU-servable). Quality is irrelevant — the seam being proven is "score completions with my own model's forward pass."

**Prompt template (train + val identical):**
```
system: You are a pathology assistant. Examine the tissue patch and
        reason step by step before answering.
user:   <image> What tissue type is shown? Choose one of:
        [adipose, background, debris, lymphocytes, mucus, smooth muscle,
        normal colon mucosa, cancer-associated stroma, colorectal
        adenocarcinoma epithelium]. Think step by step, then give your
        final answer as <answer>class_name</answer>.
```

### Phase 0 — Validate the template as-is (~1 hr, text GRPO)

You've started this. Finish it to prove cluster/image/uv/NCCL before touching anything:

1. Workspace on the template's image (`novaskyai/skyrl-train-ray-2.51.1-slim-py312-cu128-megatron-2.10-te`), 1 node × 8 GPUs (A100-80G or H100 class), `/mnt/cluster_storage` mounted.
2. `git clone https://github.com/NovaSky-AI/SkyRL.git && git checkout acbc21c` (template pin), run the GSM8K prep + train exactly per the template README, `trainer.logger=console`, `trainer.epochs=1`.
3. **Gate:** rewards logged per step, checkpoint written to `/mnt/cluster_storage/ckpts/`, no NCCL/placement-group errors. Kill it after ~20 steps — you're validating plumbing, not training.

⚠️ **Re-pin + vLLM override before Phase 1** (this replaces any image-tag worry —
your `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8` image is fine as the system
layer; uv ships the Python deps per-run):

1. `git checkout main` (or the latest release tag) instead of the template's `acbc21c` — VLM support is post-April-2026, the template pin predates it.
2. `grep vllm skyrl-train/pyproject.toml`. The VLM tutorial requires vLLM ≥ commit `80b18230e` (disaggregated multimodal render/generate). If the pin is still `vllm==0.19.0`:
   - `git clone https://github.com/vllm-project/vllm && git checkout 80b18230e` (or newer)
   - add under `[tool.uv.sources]` in the repo-root `pyproject.toml`: `vllm = { path = "/abs/path/to/vllm" }` and unpin vllm in the `fsdp` extra
   - try `VLLM_USE_PRECOMPILED=1` to dodge a from-source kernel build; budget time here — this is the one genuinely annoying env step
   - if the docs' "until the next vLLM release" has since happened and the pin is already new enough: skip all of this.
3. Rerun a 5-step GSM8K smoke on the new pin (proves image ↔ commit compat before you invest in dataset work).

The riff base is **`examples/train/geometry3k/`** — dataset script, env, run script, LoRA variant, custom entrypoint (`geometry3k_entrypoint.py`, not `main_base`). Copy schema and config keys **from it**, not from this doc.

### Phase 1 — Pathology dataset in SkyRL's schema (~2–3 hrs)

1. Write `nct_crc_dataset.py` modeled on `examples/train/geometry3k/geometry_3k_dataset.py`: pull the HF dataset, subsample 2,000/200 class-balanced, emit `train.parquet`/`val.parquet` to `/mnt/cluster_storage/data/nct_crc/`.
2. Schema (verified from the tutorial): images go **inside the prompt as base64 JPEG data URIs**, no separate image column:
   ```json
   {"prompt": [{"role": "user", "content": [
       {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<...>"}},
       {"type": "text", "text": "What tissue type is shown? ..."}]}],
    "env_class": "nct_crc",
    "reward_spec": {"method": "rule", "ground_truth": "lymphocytes"}}
   ```
3. **Gate:** load 5 rows back, decode a data URI to a PIL image, eyeball one. (Tokenization is vLLM's job at rollout time — the render-endpoint design means you don't pre-validate with an HF processor.)

### Phase 2 — VLM GRPO with rule reward (day 1 core)

The geometry3k example with three swaps: dataset → yours, env → single-turn label match, tools → none.

1. **Env:** copy `examples/train/geometry3k/env.py`, strip the `calc_score` tool protocol:
   - reward = `1.0` if `<answer>` contents match the label (case/whitespace-insensitive), `+0.2` format bonus for well-formed tags, `0` otherwise; single-turn (`done` after one step).
   - Register it and reference it from the dataset's `env_class` (`nct_crc`).
2. **Run script:** copy `run_geometry3k.sh` + `geometry3k_entrypoint.py` → `run_pathvlm.sh` / `pathvlm_entrypoint.py`. Key overrides (verified from the example):
   - `trainer.policy.model.path="Qwen/Qwen3-VL-2B-Instruct"`, `trainer.strategy=fsdp`
   - **`generator.vision_language_generator=true`** and **`trainer.remove_microbatch_padding=false`** — the two flags that switch on VLM mode (sample packing unsupported for VLMs); `generator.batched=false`
   - `generator.max_turns=1` (geometry3k uses 3 + tool calls; PathAI shape is single-turn)
   - `trainer.algorithm.advantage_estimator="grpo"`, `generator.n_samples_per_prompt=4–5`, temperature ≥ 0.7 (group variance needs it)
   - example defaults: max prompt 1024 / max generate 2048 — fine as-is; small micro-batches to start (vision towers eat memory; tutorial explicitly says calibrate `gpu_memory_utilization`, `max_model_len`, `train_batch_size`)
   - checkpoints → `/mnt/cluster_storage/ckpts/pathvlm_2b`
3. **Validation gates, in order:**
   - a. vLLM engines load the VLM and rollouts mention tissue morphology (image conditioning is real — spot-read 5 completions)
   - b. rewards **vary within a group** (advantage ≠ 0; if all-0 the task is too hard → check prompt/parsing; if all-1 too easy → drop the format bonus)
   - c. policy update completes; step 2 generations differ from step 1 (**weight sync worked** — the single most important gate in the whole spike)
   - d. 30–50 steps: mean reward trending up. 3B on 9-class should move within ~30 steps; val accuracy vs step 0 is your demo chart.
4. **Likely snags:** FSDP wrapping of the vision tower (the VLM example's trainer config will show the wrap policy), flash-attn vs Turing-style dtype issues (not on A100/H100), processor chat-template mismatches between vLLM and the trainer side — that last one is exactly the logprob-drift class, so if you hit it, document it carefully; it's a finding, not just a bug.

### Phase 3 — Custom reward model seam (day 2 morning)

Swap the rule for a model, matching PathAI's actual shape (reward = in-house model forward pass):

1. **Step 1 (in-process):** inside the env, lazily load the deberta reward model once per env worker (CPU is fine at spike scale), score `(prompt, completion)`, blend: `0.7 · label_match + 0.3 · normalized_rm_score`. Keeping the rule component preserves training signal while proving the seam.
2. **Step 2 (if time — closer to their prod shape):** move the scorer to a dedicated Ray actor with its own GPU fraction, called from envs — demonstrates reward models as separately scheduled cluster citizens, which is the story for their two in-house models at scale.
3. **Gate:** rewards flow from a model forward pass; note throughput impact (rollout→reward latency now matters); training still progresses.

### Phase 4 — LoRA + 8B (day 2 afternoon)

**LoRA-for-VLM is confirmed shipped** (`run_geometry3k_lora.sh` exists) — so this
phase is validation on *our* task shape, not an investigation:

1. Add to the Phase 2 script (straight from the LoRA variant):
   ```
   trainer.policy.model.lora.rank=32 \
   trainer.policy.model.lora.alpha=32 \
   trainer.policy.optimizer_config.lr=3.0e-5 \
   ```
   and bump model → `Qwen/Qwen3-VL-8B-Instruct`.
2. The critical check stays: **adapter × weight sync** — confirm step-2-generations-differ under LoRA, and note how sync handles adapters (there's a `lora_sync_path` param; see the LoRA example page for `target_modules`/`exclude_modules` if the vision tower needs excluding).
3. Any breakage here is a precise, reportable gap (PathAI trains LoRA today) — and note full-param 8B on 40 B300s is viable, so it may be a POV non-blocker even if rough.

### Phase 5 — Stretch (only if 0–4 are clean)

- **2 nodes:** same run, 2× nodes — placement groups span nodes, weight sync crosses NCCL boundaries. This is the multi-node claim in the POV doc.
- **Context length probe:** pack 4–8 patches per prompt (multi-image = WSI-context proxy), walk `max_prompt_length` up (4k → 16k → 32k) until something breaks; **record the number** — long-context CP is on SkyRL's roadmap, not shipped, and PathAI's WSI-derived inputs will care.
- Async rollout config if trivially available.

### Runsheet + deliverable

Keep a running log (pin/override/workaround + timestamp) — it becomes both the gap summary and the week-2 runbook. Final deliverable: ½-page — pass/fail per unknown in the table above, gaps with effort-to-close, go/no-go for weeks 2–5, veRL recommendation if no-go. Feed gaps to the SkyRL team regardless.

**Known limits — say them up front:**
- **Blackwell isn't covered.** Cloud A100/H100 ≠ their 5× B300 fleet; B300 CUDA/NCCL gets proven in week 1 on their operator install. Separate risks, retired separately.
- **Model/context are proxies** until Ryun confirms base VLM architecture + typical context length (already in the POV doc's baseline-data asks). Exotic answer → rerun the shape-sensitive parts only.
