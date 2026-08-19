# Runsheet — biotech VLM GRPO spike

Running log of pins, overrides, workarounds and findings. Feeds both the gap
summary for the SkyRL team and the week-2 runbook. All times 2026-08-19 UTC,
workspace `expwrk_gjlpp3xqhhe7xe1y397uzl3qn8` (head `m5.2xlarge` CPU-only +
one `g5.12xlarge` worker = 4x A10G 24GB).

## Environment facts confirmed

| Thing | Value | How confirmed |
|---|---|---|
| SkyRL checkout | `~/default/SkyRL` @ `9719b4f7` (main) | `git log` |
| Image | `novaskyai/skyrl-train-ray-2.56.0-py3.12-cu12.8` | `$ANYSCALE_RAY_IMAGE_URI` |
| torch | 2.11.0+cu128 | built the `--extra fsdp` env and imported |
| vLLM | **0.26.0** | same |
| datasets | 5.0.1 | same |
| Base conda python | has **no** torch/vllm/skyrl | `pip list`; everything comes from `uv run` |
| Cluster GPUs | 4x A10G (`anyscale/accelerator_shape:4xA10G`) | `ray status` |

**The vLLM local-source override in the docs is confirmed stale — skipped.**
`docs/.../vision_language_rl.mdx` and `run_geometry3k.sh`'s header still say the
repo pins `vllm==0.19.0` and needs a local clone at commit `80b18230e`. The repo
is on `vllm==0.26.0` and the env built and imported clean with zero overrides.
No vLLM clone, no `[tool.uv.sources]` edit, no `VLLM_USE_PRECOMPILED`.

## Findings (in the order they bit)

### 1. HF_TOKEN — resolved, and not actually needed for this dataset
`anyscale workspace_v2 update --env` requires terminating the workspace, so the
token now lives in `~/.workspacerc`; `run_pathvlm.sh` sources it if present.
Independently, `1aurent/NCT-CRC-HE` is **ungated** (`"gated": false` from the HF
API), so the dataset step needs no token at all.

### 2. Cross-node module import — the PLAN's open question, settled
The plan offered two options. **Both are wrong as written**; the answer is a
third one.

Traced: `skyrl_gym.make()` is called in `SkyRLVLMGymGenerator.agent_loop`
(`skyrl/train/generators/skyrl_vlm_generator.py:95`), and the generator is
constructed by `BasePPOExp` in the same process as the `@ray.remote` entrypoint
task. So `register()` and `make()` *are* in one process — but that process is
**not necessarily on the head node**. Observed: `(geometry3k_entrypoint
pid=5851, ip=10.0.104.69)` — the entrypoint task ran on the GPU **worker**.

- Plan option (1), registering the class object after a `sys.path` insert, only
  works if the file exists on whichever node the task lands on. It does not.
- Plan option (2), `uv run --project ~/default/SkyRL` from the template dir,
  **cannot work**: Ray's uv hook sets `runtime_env["working_dir"] = os.getcwd()`
  and then `_check_working_dir_files` *rejects* a `pyproject.toml` outside that
  working_dir (`ray/_private/runtime_env/uv_runtime_env_hook.py:404`).
- **What actually works, and keeps the template self-contained:** stop relying on
  the worker importing anything. `ray.cloudpickle.register_pickle_by_value(env_module)`
  in the entrypoint makes cloudpickle serialise the env class *by value* into the
  task closure, so the definition travels with the task and is registered as a
  class object rather than a dotted string. Launch CWD is still the SkyRL repo
  root (the uv hook demands it), but scripts are invoked by absolute path and
  **nothing is copied into the SkyRL checkout**.

  Verified on the worker with the SkyRL tree pristine:
  ```
  import env                                 FAILED (ModuleNotFoundError)
  import examples.train.biotech_vlm_grpo.env FAILED (ModuleNotFoundError)
  correct+formatted                          1.2   ✓
  wrong+formatted                            0.2   ✓
  ```

  **Correction:** an earlier pass of this document recommended copying the
  sources into `$SKYRL_HOME/examples/train/biotech_vlm_grpo/` and using
  geometry3k's dotted entry point. That works — it is what the first training
  run used — but it is *not required*, and it leaves untracked files in someone
  else's repo. The by-value registration above supersedes it. Results are
  unaffected: this is import plumbing, not behaviour.

Prerequisite: `export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook`
(SkyRL's own install doc), otherwise Ray workers get the bare conda python,
which has no torch.

### 3. Anything the training job reads must be on `/mnt/cluster_storage`
First control run died with `FileNotFoundError: /home/ray/data/geometry_3k/train.parquet`
even though the dataset script had just written it — the script ran on the head
node, the trainer ran on the worker, and `/home/ray/default` is node-local
overlay, not a shared mount. `/mnt/cluster_storage` is the NFS mount. Data,
ckpts, exports and `trainer.log_path` all point there now.

### 4. Qwen3-VL-2B's 262k context does not fit an A10G 🔴
The first real VLM engine init failed:

```
ValueError: To serve at least one request with the model's max seq len (262144),
(28.0 GiB KV cache is needed, which is larger than the available KV cache memory
(11.66 GiB). ... estimated maximum model length is 109200.
```

Qwen3-VL-2B-Instruct advertises a 262,144-token context and vLLM sizes the KV
cache from that. SkyRL has **no first-class `max_model_len` field** —
`trainer.max_prompt_length` and `generator.max_input_length` do not reach the
engine. The pass-through is `engine_init_kwargs`, applied last onto the vLLM arg
namespace (`skyrl/backends/skyrl_train/inference_servers/utils.py:157`):

```
generator.inference_engine.engine_init_kwargs.max_model_len=4096
```

This is a genuine sharp edge worth reporting upstream: any modern long-context
model on <40GB GPUs hits it, the error arrives ~5 minutes into startup, and the
fix is not discoverable from the config schema.

### 5. Mixed message content types silently corrupt the parquet 🔴
Adding a **system** turn with plain-string content alongside a **user** turn with
a content-part list makes Arrow unable to unify the `content` column, so HF
`datasets` coerces the whole field to an `extension<arrow.json>` **string**. The
prompt then reaches the generator as literal JSON text rather than messages.

Wrong (what a straight read of the spike doc's prompt template produces):
```
prompt: list<struct<role: string, content: extension<arrow.json>>>
```
Right (matches geometry3k):
```
prompt: list<struct<role: string, content: list<extension<arrow.json>>>>
```
Fix: give the system turn a content-part list too —
`{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}`.
geometry3k never hits this because it has a single user turn. Anyone adding a
system prompt to a SkyRL VLM dataset will.

### 6. `load_dataset(repo, split=...)` downloads the whole config
`1aurent/NCT-CRC-HE` has 3 splits totalling ~31GB. Asking for one split still
pulls every file. `data_files={"pool": "data/CRC_VAL_HE_7K-*.parquet"}` fetches
just the 3 shards (~1.1GB), but then trips split verification
(`ExpectedMoreSplitsError`), so `verification_mode=NO_CHECKS` is required too.

### 7. NCT-CRC-HE-100K is stored sorted by class
Shard `00000` is 100% DEB, shard `00010` is 105 ADI + 3121 LYM, shard `00030` is
100% MUS. A class-balanced subsample of the 100K split therefore means
downloading most of its 15GB. **CRC-VAL-HE-7K** (3 shards, 1.1GB, 7,180 patches,
all 9 classes) is the sampling pool instead, split disjointly into 1,998 train /
198 val. Both subsamples come from the same cohort, so val accuracy is a
training-signal check, not a generalization claim — call that out in the demo.

## Gates

- **Phase 1 (dataset) — PASS.** 1,998 train / 198 val, exactly 222 and 22 per
  class across all 9 classes. Prompt schema byte-for-byte matches geometry3k's.
  5 rows decoded from base64 back to 224x224 RGB JPEG; `results/sample_patch.png`
  is eosinophilic fibrillar stroma with elongated spindle nuclei, matching its
  `cancer-associated stroma` label.
- **Env reward logic — PASS.** 8/8 unit cases: correct+formatted 1.2,
  wrong+formatted 0.2, unformatted 0.0, last-tag-wins, empty tags rejected,
  abbreviations (`LYM`, `TUM`) and punctuation/case variants accepted.
- **Phase 2 gates a–d** — see below, in progress.

## Phase 2 results — the run got up (17:00–17:07)

Timeline of a cold start on 4x A10G, from `bash run_pathvlm.sh` to first policy
update: **~7 minutes**.

```
17:01:01  ray init, registries synced
17:01:22  train parquet read on the WORKER (1998 rows)
17:01:44  prompt-length filter: 1998 -> 1998   (nothing dropped)
17:02:35  vLLM resolves Qwen3VLForConditionalGeneration, max_seq_len=4096
17:03:xx  CUDA graphs captured, 51/51, on all 4 engines
17:04:46  FSDP RayActorGroup process group init
17:05:32  eval (before train) starts
17:06:22  eval done, 49.74s
17:07:02  step 1 rollouts scored
17:07:17  old-logprob recompute done (14.99s), policy update starts
```

### Gate a — image conditioning is real ✅
Rollouts describe morphology specific to each patch, not boilerplate. On a
`normal colon mucosa` patch: *"glandular structures ... arranged in a regular,
organized pattern ... epithelial cells appear to be cuboidal or columnar ...
presence of goblet cells (which produce mucus)"*. On a
`cancer-associated stroma` patch: *"a dense, fibrous matrix with scattered dark
nuclei ... lacks the typical features of normal colon mucosa, such as epithelial
lining or glandular structures"*. Different patches, different descriptions,
correct vocabulary. The VLM is looking at the image.

### Step-0 baseline (the left endpoint of the demo chart)
| metric | value |
|---|---|
| `eval/all/environment/avg_acc` | **0.5354** (9-class; chance = 0.111) |
| `eval/all/environment/avg_format_ok` | **0.9596** |
| `eval/all/avg_score` (mean reward) | 0.7273 |
| mean completion length | 250 tokens (min 132, max 1024) |

**The reward function is provably being applied end-to-end.** Re-deriving the
numbers from the raw completion text independently of SkyRL: 190/198 well-formed,
106/198 correct, so `0.5354*1.0 + 0.9596*0.2 = 0.72727` — matching SkyRL's
reported `avg_score` of `0.7272727` exactly.

The 8 zero-reward samples (`avg_tokens_zero_rewards: 1024.0`) are all
`stop_reason: length` — they ran the full 1024-token budget writing a numbered
list and never emitted the closing tag. 4% truncation is acceptable and gives
GRPO a real "be concise" gradient; raise `generator.sampling_params.max_generate_length`
if it bothers you.

53.5% base accuracy is close to ideal for a spike demo: far above chance, far
below ceiling, so there is headroom to show movement and guaranteed reward
variance inside a GRPO group.

### Gate b — reward variance within a group ✅
Step 1: `reward/avg_pass_at_4: 1.0` (every group of 4 has at least one correct)
with `reward/avg_raw_reward: 0.6375` — strictly between the 0.0 and 1.2 extremes,
so group std > 0 and advantages are non-degenerate. Neither collapse mode (all-0
"too hard" or all-1 "too easy") is present.

### Notes
- **`HF_TOKEN` does not reach the Ray workers.** Both the entrypoint task and the
  vLLM engines log *"You are sending unauthenticated requests to the HF Hub"*.
  `prepare_runtime_environment()` builds its own env-var dict and does not
  forward it. Harmless for ungated Qwen3-VL-2B; it will bite on a gated model or
  under rate limiting. Workaround: add it to the runtime env explicitly.
- `SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1` is worth setting while iterating. Without
  it, infra logs go to `trainer.log_path` **on the worker node**, which is
  invisible from the head node unless that path is on `/mnt/cluster_storage`.
- `SKYRL_RAY_PG_TIMEOUT_IN_S` default is 180s. A cold `g5.12xlarge` autoscale
  takes longer, so the placement group times out on the first run of the day.
  `run_pathvlm.sh` sets 1800.

### Gate c — weight sync ✅ (the one that matters)

```
step 1   sync_weights  5.72s -> generate 4.39s -> avg_raw_reward 0.6375
         policy_train 96.65s,  policy_loss 9.964e-05
step 2   sync_weights 10.85s -> generate 5.30s -> avg_raw_reward 0.7625
```

`sync_weights` completes as an explicit logged NCCL operation between the
optimizer step and the next rollout, and `policy_loss` is non-zero, so gradients
flowed and the updated weights reached the vLLM engines. **This is the single
most important gate in the spike and it passes on the first try.**

Do **not** read the 0.6375 -> 0.7625 move as learning. Those are two different
16-prompt batches and that swing is comfortably inside batch noise at n=16. The
evidence for weight sync is the completed sync op plus a non-zero loss, nothing
more. A real trend needs ~30 steps — that is gate (d), still accumulating.

### Step economics on 4x A10G (Qwen3-VL-2B, 16 prompts x 4 samples = 64 seqs)

| phase | time |
|---|---|
| `generate` (rollouts) | ~5s |
| `fwd_logprobs_values_reward` | ~15s |
| `policy_train` | ~97s |
| `sync_weights` | 6–11s |
| **total per step** | **~2 min** |

124 steps per epoch => ~4.5h for a full epoch; a 30-step demo is ~1 hour.
`policy_train` dominates at 75% of step time — it is running
`micro_train_batch_size_per_gpu=1` with gradient checkpointing, which is the
conservative setting chosen for 24GB cards. On bigger cards raise the micro
batch first.

### 8. Ray workers get almost no env vars — `TENSORBOARD_DIR` fails silently 🔴

`prepare_runtime_environment()` (`skyrl/train/utils/utils.py`) forwards a
**hardcoded allowlist** into `ray.init`'s runtime_env: `NCCL_NET_PLUGIN`,
`RAY_CGRAPH_get_timeout`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`, `FLA_TILELANG`,
plus a few it sets itself. There is **no generic passthrough**. Since the
entrypoint task runs on a *worker*, exporting anything else in the launching
shell has no effect on the training process.

Two things this breaks:

- **`TENSORBOARD_DIR` — silent data loss.** `Tracking` does
  `os.environ.get("TENSORBOARD_DIR", "tensorboard_log")`. With the var missing on
  the worker it falls back to that *relative* path, which resolves inside Ray's
  ephemeral `runtime_resources/working_dir_files/_ray_pkg_*/` directory on the
  worker and is deleted when the job ends. Observed: `Saving tensorboard log to
  tensorboard_log.` and no `events.out.tfevents*` anywhere on either node.
  Nothing errors; the metrics just never arrive.
- **`HF_TOKEN`** — same mechanism, hence the unauthenticated-HF-Hub warnings.

Fix applied in `pathvlm_entrypoint.py`: capture `TENSORBOARD_DIR`, `HF_TOKEN`
and `HF_HOME` from the driver's environment at submit time, pass them as a task
argument, and `os.environ.update()` them inside the task before `BasePPOExp`
constructs the tracker. `run_pathvlm.sh` also now defaults `TENSORBOARD_DIR` to
`/mnt/cluster_storage/tensorboard/pathvlm_2b` so it survives on shared storage.

Note the run captured above predates this fix, so **its** tensorboard output was
lost. Its eval metrics were not: `trainer.export_path` was already on
`/mnt/cluster_storage`, so `dumped_evals/global_step_*/aggregated_results.jsonl`
persisted correctly and is the source of truth for the accuracy chart.

This is the third distinct instance of one root cause worth stating plainly to
the SkyRL team: **the training process does not run where you launched it, and
almost nothing from the launching shell follows it there.** Data paths, log
paths and env vars all have to be made explicitly cluster-visible. Each failure
mode is different — a `FileNotFoundError`, a hub rate-limit warning, and silent
loss of all metrics.

### Gate d — reward trending up: NOT YET MET at step 10 ⚠️

Eval runs on the same 198 held-out patches each time, so step 0 vs step 10 is a
*paired* comparison and McNemar's test applies:

| | step 0 | step 10 |
|---|---|---|
| accuracy | 106/198 = 0.5354 | 112/198 = 0.5657 |
| `avg_format_ok` | 190/198 = 0.9596 | 192/198 = 0.9697 |
| `avg_score` (mean reward) | 0.7273 | 0.7596 |
| truncated at 1024 tokens | 8 | 6 |

Discordant pairs: 12 regressed, 18 improved. **McNemar exact two-sided
p = 0.362 — not significant.** All three metrics move the right way and none
move the wrong way, which is encouraging, but +6/198 at n=30 discordant is
comfortably inside noise. **Do not present step 10 as evidence of learning.**

This is consistent with the spike doc's own expectation ("3B on 9-class should
move within ~30 steps"). `lr=1.0e-6` is inherited from geometry3k, where it is
tuned for an *8B* model; it is conservative for 2B. If a later run needs to show
movement faster, raising the LR is the first knob, not more steps.

Training-batch rewards over the first 7 steps — same story, noise-dominated at
16 prompts/step: 0.638, 0.763, 0.747, 0.747, 0.809, 0.653, 0.841.

---

## FINAL RESULT — gate d passed (100 steps, ~3.5h on 4 × A10G)

Eval runs greedily (`eval_sampling_params.temperature: 0.0`) on the same 198
held-out patches every time, so step-to-step differences are real policy changes,
not resampling. That makes the comparison paired, and McNemar's exact test applies.

| step | val accuracy | format_ok | improved | regressed | McNemar p |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.5354 | 0.9596 | — | — | — |
| 10 | 0.5657 | 0.9697 | 18 | 12 | 0.362 |
| 20 | 0.5758 | 0.9747 | 27 | 19 | 0.302 |
| 30 | 0.5808 | 0.9848 | 27 | 18 | 0.233 |
| 40 | 0.5707 | 0.9545 | 26 | 19 | 0.371 |
| 50 | 0.6212 | 0.9798 | 34 | 17 | **0.024** |
| 60 | 0.6313 | 0.9848 | 32 | 13 | 0.0066 |
| 70 | 0.6667 | 0.9798 | 38 | 12 | 0.0003 |
| 80 | 0.6970 | 0.9848 | 41 | 9 | <0.0001 |
| 90 | 0.6970 | 0.9697 | 44 | 12 | <0.0001 |
| 100 | **0.7071** | 0.9798 | 46 | 12 | **<0.0001** |

**53.5% → 70.7% on 9-class colorectal tissue classification** (chance = 11.1%),
significant from step 50 onward and overwhelming by step 80. Training-batch
reward rose 0.638 → 1.138 against a 1.2 ceiling. 107 steps, zero tracebacks.

Note how the early reads would have misled: at step 10 the accuracy delta was
+3 points with p = 0.36, and step 40 actually *dipped*. Anyone calling the
result at step 10–40 would have called it wrong in both directions. The spike
doc's "should move within ~30 steps" was about right for the trend and
optimistic for significance, which arrived at 50.

Per-step eval metrics are archived in `results/eval_step_*.jsonl` (plus per-step timings in
`results/train_step_metrics.txt`) so the chart
survives the cluster.

## Deferred / known gaps

- **The run is invisible in the Anyscale Workloads tab.** It was launched as a
  bare driver (`uv run … python …`) attached to the workspace cluster, so it is
  a Ray driver, not a submitted workload — nothing to click on, no lifecycle, and
  it dies with the terminal. For weeks 2–5, submit with `anyscale job submit`
  (or `ray job submit`) so runs get a workload entry, retries, durable logs and
  independence from the shell. Worth doing before the first long run.
- Phase 3 (reward model), Phase 4 (LoRA + 8B), Phase 5 (2-node, long context).
- Not a registered Anyscale template — no BUILD.yaml entry, compute configs,
  job_config.yaml or python_depset.lock. Use the `/template` skill for that.
