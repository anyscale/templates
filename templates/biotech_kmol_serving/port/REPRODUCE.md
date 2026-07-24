# Reproducing the kMoL port + throughput results

Every number in `README.md` / `TAKEDA_BRIEF.md` comes from a script in `port/scripts/`
run on the Anyscale workspace below. This file is the map + the exact commands.

## Where everything is

**In this repo** (`templates/biotech_kmol_serving/`, branch `geoff/biotech_kmol_serving`):

| Path | What |
|---|---|
| `plan.md` | The plan, post-mortem, and P0→P3 progress log |
| `port/README.md` | Full technical writeup + every result |
| `port/TAKEDA_BRIEF.md` | Shareable external brief |
| `port/kmolport/` | The ported package (torch / torch_geometric / rdkit only) |
| `port/scripts/` | All drivers (below) |
| `port/*_results.json` | Raw measurements: `gpu_`, `serve_bulk_`, `serve_singlereq_`, `pipeline_`, `scale_` |
| `port/serve_app.py`, `Dockerfile`, `service*.yaml` | Deployment |

**On the workspace** `expwrk_9e9qajmqr7w6astmetm8v9tv9s`
(project `prj_cz951f43jjdybtzkx1s5sjgz99`, cloud `cld_kvedZWag2qA8i5BjxUevf5i7`) —
everything is already staged and persists under `/home/ray/default`:

| Path | What |
|---|---|
| `/home/ray/default/kmol_port/` | The driver scripts + result JSONs (run from here) |
| `/home/ray/default/kmol_ship/` | Self-contained dir shipped to GPU workers as Ray `working_dir` (kmolport + config.json + checkpoints + ref_logits.json + serve_app.py) |
| `/home/ray/default/kmol_serving/` | Original kMoL source, `configs/`, synthetic `checkpoints/`, `stubs/` |
| `/home/ray/anaconda3/envs/kmol` | The py3.9 / torch 1.13 kMoL env (parity ground truth) |

All commands below run **on the workspace**, via:
`anyscale workspace_v2 run_command --id expwrk_9e9qajmqr7w6astmetm8v9tv9s '<cmd>'`
(or just open a terminal in the workspace).

---

## Fast path — re-run on this workspace (everything is staged)

### 1. Reference logits from the real py3.9 kMoL (parity ground truth)
```bash
cd /home/ray/default/kmol_serving
PYTHONPATH=kmol/src:stubs /home/ray/anaconda3/envs/kmol/bin/python \
  /home/ray/default/kmol_port/ref_logits.py \
  configs/ensemble_serve.cpu.json /home/ray/default/kmol_port/ref_logits.json
# -> writes ref_logits.json (10 SMILES x 12 logits)
```

### 2. CPU parity — the port matches, bit-for-bit
```bash
# base env is py3.11; install the modern stack once (CPU build for this check)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric==2.6.1 rdkit==2024.3.5 "numpy<2"

cd /home/ray/default/kmol_port
PYTHONPATH=/home/ray/default/kmol_port python port_check.py \
  --config /home/ray/default/kmol_serving/configs/ensemble_serve.example.json \
  --ckpt-dir /home/ray/default/kmol_serving/checkpoints \
  --ref ref_logits.json --device cpu
# -> PARITY PASS  max abs diff = 0.000e+00
```

### 3–6. GPU runs (autoscale + benchmark)
Each driver connects to the managed cluster, ships `kmol_ship` as the Ray `working_dir`,
installs CUDA torch on the GPU worker(s) via `runtime_env` pip, runs, and writes a JSON.
**Launch detached and poll** (see gotchas — `run_command` kills its process tree on exit):

```bash
cd /home/ray/default/kmol_port

# 3) GPU parity + forward-only throughput curve  -> gpu_results.json (60,101 mol/s peak, parity 1.14e-05)
setsid nohup python gpu_run.py        < /dev/null > gpu_run.log      2>&1 & disown

# 4) served, Ray Serve, bulk requests           -> serve_bulk_results.json (1,697 mol/s)
setsid nohup python serve_bulk.py     < /dev/null > serve_bulk.log   2>&1 & disown

# 5) two-stage pipeline (12 featurizers -> 1 GPU)-> pipeline_results.json (3,904 mol/s)
setsid nohup python scaled_pipeline.py< /dev/null > pipeline.log     2>&1 & disown

# 6) multi-GPU scaling, 1->4 L4                  -> scale_results.json (8,792 mol/s, 98% eff)
setsid nohup python scale_gpus.py     < /dev/null > scale.log        2>&1 & disown
```

Poll any of them (first GPU run per node takes ~4–5 min for autoscale + torch install):
```bash
pgrep -f "python scale_gpus.py" >/dev/null && echo RUNNING || echo DONE
cat /home/ray/default/kmol_port/scale_results.json    # appears when finished
tail -20 /home/ray/default/kmol_port/scale.log        # live progress
```
Run these **one at a time** (each wants the GPU node). Serve/pipeline/scaling clean up
after themselves (`serve.shutdown()` / `remove_placement_group`), so GPUs autoscale back
to zero between runs.

---

## From scratch (fresh workspace / just the repo)

1. Start an Anyscale workspace on image `anyscale/ray:2.56.0-py311-cu121` with a GPU
   worker group of `g6.2xlarge` (L4), min 0 / max 4.
2. Get this branch into the workspace and stage the ship dir:
   ```bash
   D=/home/ray/default/kmol_ship; mkdir -p $D
   cp -r <repo>/port/kmolport $D/kmolport
   cp <repo>/port/serve_app.py $D/serve_app.py
   # config + checkpoints:
   cp <config.json> $D/config.json          # e.g. configs/ensemble_serve.example.json
   cp -r <checkpoints_dir> $D/checkpoints    # 5x model_*.pt (real, or synthetic — see below)
   ```
   Copy the drivers from `port/scripts/` into `/home/ray/default/kmol_port/` (they use
   these absolute paths; edit the `SHIP_DIR`/`CONFIG`/`CKPT_DIR` constants if you relocate).
3. For the **parity** check you also need the py3.9 kMoL env + a `ref_logits.json`. Rebuild
   the env from `kmol_serving/scripts/setup_env.sh`, then run step 1 above. Parity is
   optional for throughput-only reproduction.
4. Run steps 3–6 as above.

**Checkpoints:** the staged ones are *synthetic* (random weights, correct architecture/
format) — fine for throughput, not for real predictions. To confirm real outputs, drop
your trained `model_*.pt` into `checkpoints/` (and the ship dir), regenerate
`ref_logits.json` with the py3.9 env, and re-run `port_check.py`. Synthetic ones are made
with `scripts/make_synthetic_checkpoints.py <config.json>` (needs the kMoL env).

---

## Deploy as a Service (gated on approval — persistent GPU spend)
```bash
# Option A: pip runtime_env (no image build)
anyscale service deploy -f port/service.yaml
# Option B: baked image
docker build -f port/Dockerfile -t <registry>/kmol-serve:latest port/
docker push <registry>/kmol-serve:latest
# set image_uri in port/service.image.yaml, then:
anyscale service deploy -f port/service.image.yaml
```

---

## Gotchas (learned the hard way)

- **`run_command` kills its whole process tree when it returns.** Launch long jobs with
  `setsid nohup <cmd> < /dev/null > log 2>&1 & disown`, then poll with separate commands.
- **Never `pkill -f 'python x.py'` in the same command that launches `x.py`** — the launch
  command's own line contains that string, so it kills itself. Kill in a separate step, or
  use a bracket pattern like `[p]ython`.
- **`/home/ray/default` is head-local — NOT mounted on autoscaled GPU workers.** Drivers
  ship everything the task needs via `ray.init(runtime_env={"working_dir": SHIP_DIR})`;
  task-side paths (`config.json`, `checkpoints`, `ref_logits.json`) are **relative** to it.
  A directory `working_dir` is only allowed at the `ray.init` (job) level, not on
  `@ray.remote`.
- **GPU workers get deps from `runtime_env` pip** (`torch==2.5.1` → CUDA 12.4 wheel, runs
  on L4). First actor on each new node installs (~3–5 min); later actors reuse it.
- **Never run `ray stop` on the shared workspace** — it kills the managed cluster.
- The **py3.9 kMoL conda env is node-local** and may be wiped if the head node is replaced;
  rebuild via `kmol_serving/scripts/setup_env.sh`.
