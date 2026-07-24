# kmolport — kMoL molecule ensemble ported to modern PyTorch

`kmolport` is a minimal, self-contained port of kMoL's **molecule GNN inference
path** to Python 3.11 + current PyTorch + PyTorch-Geometric. It exists so the
5-model ensemble runs as ordinary Ray Serve actors on Anyscale's managed cluster
with native GPU autoscaling — no frozen py3.9 stack, no isolated Ray, no NFS conda
staging, no container gymnastics.

## Run it yourself — start here

In the Anyscale workspace **`takeda-kmol`**, open a terminal. There is **nothing to
install**: each script declares the pinned stack (`torch`, `torch_geometric`, `rdkit`) in
its Ray `runtime_env`, so the workers install it themselves, and `checkpoints/` already
holds demo weights.

```bash
cd ~/default/port

# 1. Correctness check — deploys the app on one L4 and asserts a real prediction comes back
python scripts/serve_pipeline_bulk.py --smoke

# 2. Per-molecule latency on minoxidil / sildenafil / atorvastatin, both serving designs
python scripts/bench_three_molecules.py --reps 100    # -> three_molecule_results.json

# 3. Throughput with the CPU featurizer tier at 24 replicas on one L4
python scripts/serve_pipeline_bulk.py --pool3 --replicas 24
```

The first run on a fresh GPU node takes ~4–5 minutes (node autoscale plus the
`runtime_env` install); later runs reuse it. Idle nodes scale back to zero, so the
workspace costs nothing sitting there.

**Predictions are placeholders right now.** `checkpoints/` holds *synthetic* weights —
correct architecture and checkpoint format, random values — so throughput and latency are
real but the numbers coming out carry no information. To get real predictions, drop your
trained `model_0.pt` … `model_4.pt` into `checkpoints/` and re-run; nothing else changes.
See "Model weights" in [`TAKEDA_BRIEF.md`](TAKEDA_BRIEF.md).

**Sizing note for this workspace.** Its CPU worker group maxes at 4 nodes (32 vCPU), so
`--replicas 24` is about the largest that schedules. The 48-replica configuration behind
the headline number in the brief needs that raised to ≥ 8 nodes — otherwise Serve waits
forever for replicas it can never place.

Results and interpretation live in [`TAKEDA_BRIEF.md`](TAKEDA_BRIEF.md); deeper
reproduction notes, every other script, and the workspace gotchas are in
[`REPRODUCE.md`](REPRODUCE.md).

## What it is

A faithful extraction of only the molecule path — no openbabel / prody / openfold /
torch_scatter. Deps: `torch`, `torch_geometric`, `rdkit`.

| File | Ported from | Notes |
|---|---|---|
| `abstract_network.py` | `kmol/model/architectures/abstract_network.py` | checkpoint loader (`state_dict`, `strict=False`) |
| `ensemble_network.py` | `.../ensemble_network.py` | 5 sub-models in one `ModuleList`; `forward` = `mean` (+ `var`) |
| `graph_convolutional_network.py` | `.../graph_convolutional_network.py` | verbatim; LEConv ×7, hidden 96 |
| `layers.py` | `kmol/model/layers/layers.py` | `GraphConvolutionWrapper` + `BatchNorm`, torch_scatter import dropped (unused here) |
| `read_out.py` | `kmol/model/read_out.py` | max/sum/mean only (config default `("max","sum")`) |
| `featurizer.py` | `kmol/data/featurizers.py` | `GraphFeaturizer` + 17 RDKit descriptors, minus the protein imports |
| `dgllife_featurizers.py` | `kmol/vendor/dgllife/utils/featurizers.py` | **verbatim** (rdkit/torch/numpy only) — the 45-dim atom features |
| `helpers.py` | `kmol/core/helpers.py` | tiny `SuperFactory` (reflect + create) |

Reusing kMoL's exact class/attribute structure is deliberate: it makes the checkpoint
keys line up so kMoL `state_dict`s load unchanged.

## Which molecule set a number used — read this first

Throughput here depends heavily on molecule size (~0.285 ms per heavy atom), so every
number below is tagged with its molecule set:

| tag | set | mean featurize | single-core mol/s |
|---|---|---:|---:|
| **real library** | 15,751 unique tox21 + ZINC + ChEMBL-MW800+, shuffled | 7.65 ms | **131** |
| **10-pool** | the old 10 hardcoded drug-like SMILES | 2.46 ms | **407** |

The 10-pool is **3.1× optimistic** — its molecules average **12.1 heavy atoms**, versus a
median of 24 and a mean of 41 in the real library (p95 = 107, max = 343). Sections P0, P1, P1b and P1c below were
measured on the 10-pool and **have not been re-run**; they are kept for provenance only.
**P2 is the only section measured on the real library.** Never compare a 10-pool row
against a real-library row.

### Featurization (preprocessing) cost — the specs behind those rates

Featurization is RDKit work on one CPU core and it is the whole bottleneck; the GPU
forward is ~20× cheaper. Cost is close to linear in molecule size:

> **featurize_ms ≈ 0.285 × (heavy atoms) − 3.89**  (r² = 0.86, fit over ~10–120 heavy
> atoms — the negative intercept is a fit artifact, don't extrapolate it to tiny molecules)

Per size bucket, single-core, measured over all 15,751 molecules:

| heavy atoms | n | mean featurize | single-core mol/s |
|---|---:|---:|---:|
| < 15 | 3,448 | 1.96 ms | 510 |
| 15–25 | 4,502 | 3.12 ms | 321 |
| 25–40 | 2,559 | 4.29 ms | 233 |
| 40–60 | 925 | 8.39 ms | 119 |
| 60+ | 4,317 | 18.75 ms | 53 |
| **full mix** | **15,751** | **7.65 ms** (median 3.71) | **131** (median 270) |

Aggregate throughput is governed by the **mean** (total time is the sum), so 131 mol/s/core
is the right number for a screen; the median matters for what a single request feels like.
The 60+ bucket is 27% of the library (ChEMBL MW>800) and dominates the mean — a
drug-like-only deck would land nearer 300 mol/s/core.

Raw stats and histograms: [`molecule_library_stats.json`](molecule_library_stats.json).

## Proven results (2026-07-23, one NVIDIA L4)

**Parity — the port is numerically identical to real kMoL.**
- CPU (head, py3.11 / torch 2.5.1): max abs diff vs py3.9 kMoL (torch 1.13 / PyG 2.3)
  reference = **0.000e+00** (bit-for-bit).
- GPU (NVIDIA L4, torch 2.5.1+cu124): max abs diff = **1.14e-05** (expected fp drift). PASS.
- This also settles the open risk: **torch runs on the L4** (kMoL's torch 1.13/cu117
  could not) — the port was required, not just cleaner.

**Throughput — one L4, `port/scripts/gpu_run.py`** (full data in `gpu_results.json`):

| batch | forward-only mol/s | ms/batch |
|---:|---:|---:|
| 1 | 69 | 14.6 |
| 64 | 4,270 | 15.0 |
| 512 | 33,950 | 15.1 |
| 1024 | **60,101** | 17.0 |

- **Peak forward-only: 60,101 mol/s ≈ 353× the ~170/s baseline.** ms/batch is flat
  (~15 ms) from batch 1→512 → the old "~5 ms/molecule" was pure batch-size-1 launch
  overhead; the GNN forward is trivially cheap. The email's premise, confirmed.
- **End-to-end (single-thread RDKit featurize + forward): ~561 mol/s ≈ 3.3× baseline,
  and it is featurization-bound, not GPU-bound.** RDKit is ~1.75 ms/mol/core; the GPU
  has ~100× headroom. The lever for end-to-end throughput is parallelizing
  featurization across cores/replicas — which is exactly what Ray Serve does (P1).

## P1 — served on Ray Serve, one L4, native autoscale *(10-pool; monolithic deployment)*

`port/scripts/serve_bulk.py` deploys the ported ensemble as **6 fractional-GPU
replicas packed on a single L4** (`num_gpus=0.16`, `num_cpus=1`) on the managed
cluster — the autoscaler brought the node up, `runtime_env` installed CUDA torch on
the replicas, no container. Each replica featurizes its own chunk (parallelism across
replicas) and shares the cheap GPU forward.

**Screening workload (chunked requests, `serve_bulk_results.json`):**

| chunk | concurrency | mol/s | req p50 | req p99 |
|---:|---:|---:|---:|---:|
| 128 | 24 | 1,672 | 1.8 s | 3.1 s |
| 256 | 24 | **1,697** | 3.5 s | 4.9 s |

**Peak served: 1,697 mol/s on one L4 = 10.0× the ~170/s baseline.** It plateaus at
~6× the per-core RDKit rate (6 replicas), i.e. it is **CPU-featurization-bound, not
GPU-bound** — consistent with P0's forward-only 60k/s. Naive one-SMILES-per-request
load (`serve_run.py`, `serve_singlereq_results.json`) tops out at ~244 mol/s: that's
the *client/RPC* limit (REC 4 — "you're benchmarking the client"), not the service.

## P1b — two-stage pipeline as raw Ray actors *(10-pool; mislabelled in earlier drafts)*

`port/scripts/scaled_pipeline.py` runs the split as pure Ray actors: **12 CPU featurizer
actors → 1 GPU forward actor**, processing 60k molecules.

| metric | mol/s |
|---|---:|
| pipeline (12 featurizer vCPU → 1 GPU actor) | **3,904** |
| featurize-only aggregate (same 12 vCPU) | 4,050 |

The useful conclusion holds: the pipeline rate is within ~4% of the featurize-only rate,
so **the GPU adds almost nothing — it is ~7% utilized.** Two labels on this run were
wrong in earlier drafts and are corrected here:

- **It was never a "one L4" number.** The featurizers are `num_cpus=1` with no GPU
  request, and this workspace's head node has **0 schedulable CPU** (verified). Those 12
  vCPU therefore came from GPU worker nodes — the autoscaler brought up roughly two
  `g6.2xlarge` to satisfy them. Per single GPU node (8 vCPU) the comparable figure is
  much lower.
- **"No Serve/HTTP overhead" was a false contrast.** The P1 Serve benchmark
  (`serve_bulk.py:78`) calls `handle.infer_bulk.remote(...)` — a `DeploymentHandle`, the
  same Ray RPC path these actors use. Neither number involved HTTP, so the gap between
  them was never Serve overhead; it was core count (12 vs 6) plus the monolith
  serializing featurize-then-forward in one thread.

## P1c — multi-GPU: near-linear scaling, 1→4 L4 *(10-pool; raw actors, compute capacity)*

`port/scripts/scale_gpus.py` runs one self-contained mini-pipeline per L4 node
(placement-group bundle = 1 GPU actor + 6 co-located featurizers, `STRICT_SPREAD` so
each bundle is a distinct node), then measures aggregate throughput at G = 1..4 GPUs.

| GPUs | mol/s | speedup vs 1 | efficiency | ×baseline |
|---:|---:|---:|---:|---:|
| 1 | 2,235 | 1.00× | 100% | 13× |
| 2 | 4,209 | 1.88× | 94% | 25× |
| 3 | 6,500 | 2.91× | 97% | 38× |
| 4 | **8,792** | **3.93×** | **98%** | **52×** |

**3.93× on 4 GPUs = 98% scaling efficiency** (`port/scale_results.json`). Each g6.2xlarge
node adds both an L4 and 8 vCPU, and since throughput is featurization-bound the two scale
together — so adding nodes adds throughput almost perfectly linearly.

## P2 — the current number: two-stage **composed Ray Serve** app, real library

`serve_pipeline_app.py` expresses the two-stage split as a single Serve application with
three independently-scaled deployments, and `scripts/serve_pipeline_bulk.py` measures it:

```
Ingress (thin, no torch)  →  Featurizer (num_cpus=1, NO GPU)  →  GpuForward (num_gpus=1)
```

The `Featurizer` deployment requesting **no GPU** is the whole point: its replicas
schedule onto plain CPU nodes and autoscale independently, instead of being pinned to a
GPU node by a `num_gpus=0.16` slice the way the P1 monolith was.

**One L4, growing only the CPU tier** (`m5.2xlarge`, real library, chunk 256, load from
4 in-cluster client actors) — [`serve_pipeline_results.json`](serve_pipeline_results.json):

| featurizer replicas (vCPU) | mol/s | per replica | p50 | p99 |
|---:|---:|---:|---:|---:|
| 6 | 631 | 105 | 4.6 s | 5.7 s |
| 12 | 956 | 80 | 4.1 s | 7.9 s |
| 24 | 1,775 | 74 | 3.9 s | 7.2 s |
| 48 | **2,809** | 59 | 4.9 s | 7.5 s |

**The per-replica decline is hyperthreading, not contention.** `m5.2xlarge` = 8 vCPU on
**4 physical cores** (`lscpu`: `Thread(s) per core: 2`), and Ray schedules per vCPU — so
48 replicas is 24 physical cores. At the one point where node count was verified (48
replicas / 6 nodes): **468 mol/s per node ≈ 117 mol/s per physical core**, against **131
mol/s** measured single-threaded → **~89% of ideal in real cores.**

**What the bottleneck is not** — both tested by doubling and watching nothing happen:

| change | mol/s | delta | file |
|---|---:|---:|---|
| baseline (48 replicas, 1 L4) | 2,809 | — | `serve_pipeline_results.json` |
| 2 × L4 | 2,892 | **+3.0%** | `serve_pipeline_2gpu.json` |
| 3× ingress replicas + 2× clients | 1,611 | **−9%** | `serve_pipeline_diag_ingress12.json` |

So it is neither GPU-bound nor front-door-bound: it is CPU-featurization-bound, as
designed, and the lever is CPU nodes.

**Two implementation findings worth keeping:**
- **Don't broker the `Batch` through the ingress.** Passing the featurizer's
  `DeploymentResponse` into the GPU handle from the ingress makes Serve *materialize* it
  there (`serialization.py → pickle.loads → torch_geometric`), which would put torch on
  the front door plus a deserialize/reserialize of every batch on the one component all
  traffic crosses. Having `Featurizer` call `GpuForward` itself gives the Batch exactly
  one hop, CPU node → GPU node.
- **Shuffle the library before chunking.** `molecules.csv` is grouped by source (tox21,
  then ZINC, then ChEMBL-MW800+), so an in-order walk makes every chunk size-homogeneous
  and makes each sweep point sample a different part of the size distribution. Unshuffled,
  the sweep came out non-monotonic (12 replicas ≈ 24 replicas; per-replica rate 195 → 58).
  The driver now shuffles with a fixed seed.

**Conclusion.** The old "~5 ms/molecule" was batch-size-1 launch overhead on the *offline*
path, not a GPU limit: the GNN forward runs at ~60k mol/s on one L4, ~20× more than the
CPU tier can feed it. Served end-to-end on a genuinely hard, size-diverse library, the
composed Serve app sustains **2,809 mol/s (0.36 ms/molecule) on one L4 plus six
`m5.2xlarge`**, at ~89% of ideal per physical core, with the GPU nearly idle. The
architecture conclusion is the durable one: **buy CPU, not GPUs.**

**Still open:** synthetic weights (no accuracy claim), no external open-loop HTTP load
test, and no verified per-node scaling curve on the real library — see
[`TAKEDA_BRIEF.md`](TAKEDA_BRIEF.md) next steps.

## Deploy as a container (P3)

`port/Dockerfile` bakes the ported stack onto a stock modern Anyscale Ray image (no conda,
no GPU-at-build, no CUDA-kernel compile — the whole point of the port). Pair with
`port/service.image.yaml` (`anyscale service deploy`). `port/service.yaml` is the
pip-`runtime_env` variant (no image build). **Deploying is persistent GPU spend — gated
on approval.**

## How to reproduce

`port/` is self-contained. From the bundle root (`cd port`):

```bash
pip install -r requirements.txt      # torch / PyG / rdkit (CPU box: see file header)

# demo weights (SYNTHETIC). For real predictions, drop your model_*.pt into checkpoints/.
PYTHONPATH=. python scripts/make_synthetic_checkpoints.py configs/ensemble_serve.example.json checkpoints

# local forward + end-to-end microbench
PYTHONPATH=. python scripts/bench.py --config configs/ensemble_serve.example.json --ckpt-dir checkpoints --device cuda

# on Ray / Anyscale (each writes a *_results.json):
python scripts/gpu_run.py               # GPU parity + forward throughput curve
python scripts/serve_pipeline_bulk.py --smoke   # composed Serve app: correctness check
python scripts/serve_pipeline_bulk.py   # composed Serve app: CPU-tier sweep  <-- P2, current
python scripts/serve_bulk.py            # monolithic deployment (P1, 10-pool; provenance)
python scripts/scale_gpus.py            # multi-GPU scaling (P1c, 10-pool; provenance)
```

The composed-Serve driver needs a CPU worker group on the cluster (we used `m5.2xlarge`,
min 0 / max ≥ 8) in addition to the GPU group — that separation is the thing being
measured. It reads the molecule library from `KMOL_LIBRARY`.

Full step-by-step — including the py3.9 parity ground truth (run against **your own**
kMoL install) and the detached-launch/poll pattern for Anyscale workspaces — is in
[`REPRODUCE.md`](REPRODUCE.md).
