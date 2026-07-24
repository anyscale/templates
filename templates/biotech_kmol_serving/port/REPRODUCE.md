# Reproducing the kMoL port + throughput results

`port/` is a self-contained bundle. Every number in `README.md` / `TAKEDA_BRIEF.md`
comes from a script in `scripts/`, and all paths default to this bundle (overridable
by env). Run everything from the bundle root: `cd port`.

## What's in the bundle

| Path | What |
|---|---|
| `kmolport/` | The ported package — torch / torch_geometric / rdkit only |
| `serve_app.py` | Production Ray Serve deployment |
| `configs/` | `ensemble_serve.example.json` (GPU) + `.cpu.json` (CPU parity) |
| `requirements.txt` | The serving stack |
| `Dockerfile`, `service.yaml`, `service.image.yaml` | Deployment |
| `scripts/` | `make_synthetic_checkpoints.py`, `ref_logits.py`, `port_check.py`, `bench.py`, `gpu_run.py`, `serve_bulk.py`, `serve_run.py`, `scaled_pipeline.py`, `scale_gpus.py`, `locustfile.py` |
| `*_results.json` | The raw measurements behind every quoted number |

`checkpoints/` is **not** in the bundle (weights are gitignored). Generate synthetic
ones (below) or drop in real `model_*.pt`.

## 0. Setup

```bash
cd port
pip install -r requirements.txt
# CPU-only box: install torch from the CPU index (see requirements.txt header) first.

# Weights: SYNTHETIC (correct architecture/format, random values) — fine for throughput,
# not for real predictions. For real outputs, put your trained model_0..4.pt in checkpoints/.
PYTHONPATH=. python scripts/make_synthetic_checkpoints.py configs/ensemble_serve.example.json checkpoints
```

## 1. Local — parity + microbench (no cluster)

```bash
# Forward + end-to-end throughput on this box (CPU or CUDA):
PYTHONPATH=. python scripts/bench.py \
    --config configs/ensemble_serve.example.json --ckpt-dir checkpoints --device cuda

# Numerical parity vs a reference (see §3 to produce ref_logits.json from your kMoL):
PYTHONPATH=. python scripts/port_check.py \
    --config configs/ensemble_serve.example.json \
    --ckpt-dir checkpoints --ref ref_logits.json --device cpu
```

## 2. Ray / Anyscale — GPU throughput, serving, scaling

Run these on an Anyscale workspace (or any Ray cluster) whose GPU worker group is
`g6.2xlarge` (L4), min 0 / max 4. Each driver connects to the cluster, uploads this
bundle as the Ray `working_dir`, installs CUDA torch on the GPU worker(s) via
`runtime_env` pip, runs, and writes a JSON. **Launch detached and poll** (see gotchas):

```bash
cd port
setsid nohup python scripts/gpu_run.py        < /dev/null > gpu.log   2>&1 & disown  # forward curve -> gpu_results.json
setsid nohup python scripts/serve_bulk.py     < /dev/null > serve.log 2>&1 & disown  # served       -> serve_bulk_results.json
setsid nohup python scripts/scaled_pipeline.py< /dev/null > pipe.log  2>&1 & disown  # 2-stage      -> pipeline_results.json
setsid nohup python scripts/scale_gpus.py     < /dev/null > scale.log 2>&1 & disown  # 1->4 GPU     -> scale_results.json

# poll (first GPU run per node ~4-5 min for autoscale + torch install):
pgrep -f scripts/scale_gpus.py >/dev/null && echo RUNNING || echo DONE ; cat scale_results.json
```

Run them **one at a time** (each wants the GPU). Serve/pipeline/scaling clean up after
themselves so GPUs autoscale back to zero between runs.

Overridable via env (defaults shown): `KMOL_SHIP_DIR` (bundle root), `KMOL_CONFIG`
(`configs/ensemble_serve.example.json`), `KMOL_CKPT_DIR` (`checkpoints`), `KMOL_OUT`,
`KMOL_REF`. Replica/GPU knobs: `KMOL_NUM_GPUS`, `KMOL_REPLICAS`, `KMOL_MAX_BATCH`.

## 3. Parity ground truth (against your own kMoL)

Parity is checked against logits from the **real kMoL**. Produce them in your kMoL
environment (Python 3.9 / torch 1.13), then compare with `port_check.py` (§1):

```bash
# in your kMoL checkout / env, with kmol importable (+ its stubs on PYTHONPATH):
python <path-to>/scripts/ref_logits.py <your-config>.json ref_logits.json
```

Copy `ref_logits.json` into the bundle. This is the only step that needs kMoL; the
port itself never imports it. Expected: max abs diff 0 on CPU, ~1e-5 on GPU.

## 4. Deploy as a Service (persistent GPU spend — gate on approval)

```bash
# A) pip runtime_env (no image build). Stage real checkpoints into ./checkpoints first.
anyscale service deploy -f service.yaml

# B) baked image
docker build -f Dockerfile -t <registry>/kmol-serve:latest .
docker push <registry>/kmol-serve:latest
# set image_uri in service.image.yaml, then:
anyscale service deploy -f service.image.yaml
```

Both enable `query_auth_token_enabled` — requests need the bearer token Anyscale issues.

## Anyscale workspace gotchas

- **`anyscale workspace_v2 run_command` kills its process tree on exit.** Launch long
  jobs with `setsid nohup <cmd> < /dev/null > log 2>&1 & disown`, then poll separately.
- **Don't `pkill -f 'python x.py'` in the same command that launches `x.py`** — the
  launch line matches the pattern and kills itself. Kill in a separate step.
- **A workspace's `/home/ray/default` is head-local — NOT on autoscaled GPU workers.**
  The drivers handle this by uploading the bundle as the Ray `working_dir`; task-side
  paths resolve relative to it. (A directory `working_dir` is only allowed at
  `ray.init`, not on `@ray.remote`.)
- **GPU workers get deps from `runtime_env` pip** (`torch==2.5.1` → CUDA 12.x wheel,
  runs on L4). First actor on each new node installs (~3–5 min); later actors reuse it.
- **Never `ray stop`** on a shared workspace — it kills the managed cluster.
