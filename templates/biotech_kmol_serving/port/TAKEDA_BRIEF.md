# Serving the kMoL GNN ensemble on Anyscale — early results

**Prepared by Anyscale · for the Takeda / kMoL team · 2026-07-24**

> **How to read the numbers below:** these come from a proof-of-concept on an Anyscale
> workspace using a **small test set (10 drug-like molecules) and synthetic (random)
> weights**. Treat them as **trends and ratios, not absolute throughput you should expect
> in production** — a real, size-diverse library will be slower per molecule (featurization
> cost grows with molecule size), and real trained weights are a drop-in swap. The point of
> this brief is the *shape* of the result: **near-linear horizontal scaling and warm,
> load-once replicas** — the two things you told us weren't working.

---

## The problem you described

- **Horizontal scaling isn't linear.** "I can do 150–200 molecules/sec on one machine. If I
  had a 2nd, this should be 400 — and it's like 120, or sometimes it slows down."
- **Replicas won't stay warm.** "It always keeps saying cold start, cold start… why are you
  reloading all this stuff?"
- **The GPU wasn't clearly worth it.** 8 CPUs ≈ **12 ms/molecule**, 1 GPU ≈ **5 ms/molecule**
  — only ~2.4×, and "we needed 4× faster to be worth having it."

We reproduced the kMoL 5-model molecule ensemble on Anyscale and went after exactly those.

---

## Result 1 — near-linear horizontal scaling (your #1 ask)

Adding GPU nodes multiplies throughput almost perfectly — the "literally NX" scaling you were
missing:

| GPU nodes (L4) | throughput | speedup vs 1 | efficiency |
|---:|---:|---:|---:|
| 1 | 2,235 mol/s | 1.00× | 100% |
| 2 | 4,209 mol/s | 1.88× | 94% |
| 3 | 6,500 mol/s | 2.91× | 97% |
| 4 | 8,792 mol/s | **3.93×** | **98%** |

*Source: [`scripts/scale_gpus.py`](scripts/scale_gpus.py) → [`scale_results.json`](scale_results.json).*
The reason it's linear where yours was asymptotic: each node is a **self-contained, balanced
unit** (its own GPU *and* its own CPU featurizers), so nodes don't contend — see "Why yours
was sublinear" below.

## Result 2 — warm replicas, loaded once (your #2 ask)

Your "cold start, cold start" is the offline kMoL `predict` path **reloading all 5 checkpoints
on every request**. We load the ensemble **once** in a long-lived actor, run a warm-up forward
before the replica is marked healthy, and keep `min_replicas ≥ 1` — so a request never pays
the reload cost. That single change is most of the gap between "looks perpetually cold" and
"hot and fast."

---

## In your units: ms / molecule, and the 4× bar

| Setup | topology / what it measures | throughput | **ms/molecule** |
|---|---|---:|---:|
| **Your 8-CPU baseline** | kMoL `predict`, reload-per-call, batch≈1 | ~83 mol/s | **12 ms** |
| **Your 1-GPU baseline** | kMoL `predict`, reload-per-call, on GPU | ~200 mol/s | **5 ms** |
| _your "worth it" bar (4× faster)_ | _target_ | _~700–800 mol/s_ | _~1.3–3 ms_ |
| **Served, batched — 1 L4** | Ray Serve HTTP endpoint; 6 replicas share one L4; load-once + dynamic batching; featurize inline | 1,697 mol/s | **0.59 ms** |
| **2-stage pipeline — 1 L4** | decoupled: 12 CPU featurizer actors → 1 GPU forward actor (no HTTP overhead) | 3,904 mol/s | **0.26 ms** |
| **4 L4, near-linear** | the 2-stage unit replicated per node (1 GPU + 6 featurizers each) | 8,792 mol/s | **0.11 ms** |
| _GPU forward ceiling_ | _pure GPU forward on a pre-featurized batch — raw headroom, not a real workload_ | _60,101 mol/s_ | _0.017 ms_ |

Even discounted for the small-molecule caveat, this clears your 4× bar with room to spare — and
on **the box you already run** (64 CPU + 1 GPU), the math is favorable: featurization is
embarrassingly parallel, so 64 cores amortize to well under a millisecond per molecule while
the single GPU sits mostly idle. Your 12→5 ms was only 2.4× because you were comparing 8 CPUs
to 1 GPU on a *reload-per-request* path; the real lever is **load-once + batch + parallel
featurization**, which finally puts that 64-CPU box to work.

### What the topology rows mean

- **Served, batched** — the realistic "deploy it and POST to it over HTTP" path. Ray Serve packs
  several replicas onto one L4 (fractional GPU) and coalesces concurrent requests into one batch.
  ([`serve_app.py`](serve_app.py), [`service.yaml`](service.yaml), [`scripts/serve_bulk.py`](scripts/serve_bulk.py))
- **2-stage pipeline** — the same work with **featurization split off the GPU**: a pool of CPU
  featurizer actors turns SMILES into graphs and feeds one GPU forward stage over a queue. This
  isolates the real bottleneck (the L4 ran only ~7% utilized here) and is the design that scales.
  ([`scripts/scaled_pipeline.py`](scripts/scaled_pipeline.py) → [`pipeline_results.json`](pipeline_results.json))
- **GPU forward ceiling** — featurization removed entirely, replaying one pre-built batch. It's a
  *headroom* number (how fast the GNN itself is), not a throughput you'd serve. ([`scripts/bench.py`](scripts/bench.py) → [`gpu_results.json`](gpu_results.json))

### Your own test molecules

We ran the three drugs you sanity-check with — minoxidil, Viagra (sildenafil), Lipitor
(atorvastatin) — end-to-end through the ported ensemble. Featurization time tracks molecule
size just as expected, and two of the three are already in our 15,751-molecule test library
(they come from tox21 — the model's own target domain):

| Drug | heavy atoms | featurize time | in our test library? |
|---|---:|---:|---|
| minoxidil | 15 | 2.7 ms | no |
| sildenafil (Viagra) | 33 | 5.2 ms | yes (as the citrate salt) |
| atorvastatin (Lipitor) | 41 | 6.2 ms | yes (tox21) |

Each produces a full 12-target tox21 prediction plus per-target variance. The sizes and
timings are real; the prediction *values* aren't meaningful here (synthetic weights). Note the
spread — Lipitor is ~1.7× the library's median size — which is exactly why per-molecule cost,
and therefore throughput, depends on your library's size mix.

---

## Correctness — port fidelity (not an accuracy claim yet)

To run on modern GPUs we ported the molecule path from kMoL's frozen `Python 3.9 / torch 1.13 /
CUDA 11.7` stack to current PyTorch — the legacy stack **cannot run on an L4 at all**. The port
reuses kMoL's exact architecture and checkpoint format, and produces **numerically identical
outputs** for the same weights:

| Comparison | max abs difference |
|---|---:|
| Port vs original kMoL (CPU) | **0.000e+00** (bit-for-bit) |
| Port vs original kMoL (L4 GPU) | **1.14e-05** (ordinary float precision) |

This proves the reimplementation is faithful — it is **not** an accuracy result, because these
runs use synthetic weights. Confirming accuracy on your real trained checkpoints is step 1 below.

---

## Why your scaling was sublinear, and the fix

The workload is **CPU-featurization-bound, not GPU-bound**. Scaling *GPU* replicas of a path that
reloads weights and featurizes inline just multiplied the overhead, not the useful throughput —
which is the asymptotic curve you saw. The production topology is **disaggregated**: a fleet of
cheap CPU featurizer nodes feeding a small pool of fractional GPUs. You size the CPU tier to your
target throughput; the GPU is shared and mostly idle. "Featurization-bound" becomes the selling
point — **you scale on inexpensive CPU, not on GPUs.** (Full design in the internal plan.)

---

## Honest caveats & next steps (the 3-week prototype)

**Caveats:** small (10-molecule) test set, synthetic weights, and the served numbers are
closed-loop (a driver script, not an independent open-loop load generator). So read trends, not
absolutes.

**To turn this into the validated prototype you described:**
1. **Drop in your real trained checkpoints** → re-run the one-command parity check + a throughput sample.
2. **Open-loop load test** on a **real, size-diverse library** with your actual request pattern
   (single-molecule interactive vs bulk screening) → the defensible end-to-end numbers + p50/p99.
3. **Cost model** — $ per million molecules given the CPU:GPU ratio your target implies.

---

## Reproduce / artifacts

Everything is in the self-contained [`port/`](.) bundle; step-by-step commands are in
[`REPRODUCE.md`](REPRODUCE.md). Each number above links to the script and the raw
`*_results.json` that produced it. Deploy configs: [`service.yaml`](service.yaml) (pip
`runtime_env`, no image build) and [`service.image.yaml`](service.image.yaml) (baked image) —
both require the bearer token (`query_auth_token_enabled`).

*Hardware: NVIDIA L4 (g6.2xlarge). Model: kMoL ensemble, 5 × GraphConvolutionalNetwork,
264,300 params each. Baseline reference: your reported ~150–200 mol/s single-machine.*
