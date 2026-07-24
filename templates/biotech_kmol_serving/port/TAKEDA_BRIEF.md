# Serving the kMoL GNN ensemble on Anyscale — early results

**Prepared by Anyscale · for the Takeda / kMoL team · 2026-07-24**

> **How to read the numbers below.** These come from a proof-of-concept on an Anyscale
> workspace. Two things to hold onto:
>
> 1. **The molecule set is real and deliberately hard.** 15,751 unique, RDKit-sanitized
>    structures — 7,831 from tox21 (the model's own target domain), 3,000 sampled from
>    ZINC, and 4,920 from ChEMBL with MW > 800. That last third is a heavy large-molecule
>    tail (median MW 351, mean 583, max 4,863). Featurization cost scales with molecule
>    size, so **this library is probably harsher than your screening deck.**
> 2. **The weights are synthetic** (correct architecture and format, random values).
>    Throughput is identical with real weights; prediction *values* are not meaningful
>    yet. Swapping in your trained checkpoints is step 1 below.
>
> If you saw an earlier version of this brief, its throughput figures were measured on a
> 10-molecule test set and were **~3.1× optimistic** — those small molecules featurize in
> 2.46 ms versus 7.65 ms for this library. Every number here is on the real library.

---

## The problem you described

- **Horizontal scaling isn't linear.** "I can do 150–200 molecules/sec on one machine. If
  I had a 2nd, this should be 400 — and it's like 120, or sometimes it slows down."
- **Replicas won't stay warm.** "It always keeps saying cold start, cold start… why are
  you reloading all this stuff?"
- **The GPU wasn't clearly worth it.** 8 CPUs ≈ **12 ms/molecule**, 1 GPU ≈ **5
  ms/molecule** — only ~2.4×, and "we needed 4× faster to be worth having it."

We reproduced the kMoL 5-model molecule ensemble on Anyscale and went after exactly those.

---

## Result 1 — the bottleneck is CPU featurization, and that is good news

The single most important measurement: **the GPU is not your constraint, and neither is
the serving layer.** We proved this by doubling each in turn and watching throughput not
move.

| Change | throughput | delta |
|---|---:|---:|
| Baseline: 1 L4 + 48 CPU featurizer slots | 2,809 mol/s | — |
| **Double the GPUs** (2 × L4) | 2,892 mol/s | **+3.0%** |
| **Triple the serving front door** (3× ingress replicas, 2× clients) | 1,611 mol/s | **−9%** (worse) |

*Sources: [`serve_pipeline_results.json`](serve_pipeline_results.json),
[`serve_pipeline_2gpu.json`](serve_pipeline_2gpu.json),
[`serve_pipeline_diag_ingress12.json`](serve_pipeline_diag_ingress12.json).*

Adding a second L4 bought 3%. The GNN forward itself runs at ~60,000 molecules/sec on one
L4 (see the headroom row later) — roughly 20× more than the CPU tier can feed it. So the
lever is CPU cores for RDKit featurization, and **you scale on inexpensive CPU nodes, not
on GPUs.** That is the cost story, and it is now measured rather than asserted.

## Result 2 — throughput tracks the CPU tier

One L4, growing only the CPU featurizer tier (each replica = 1 vCPU on `m5.2xlarge`):

| CPU featurizer replicas | throughput | per replica | req p50 | req p99 |
|---:|---:|---:|---:|---:|
| 6 | 631 mol/s | 105 | 4.6 s | 5.7 s |
| 12 | 956 mol/s | 80 | 4.1 s | 7.9 s |
| 24 | 1,775 mol/s | 74 | 3.9 s | 7.2 s |
| 48 | **2,809 mol/s** | 59 | 4.9 s | 7.5 s |

**Read the "per replica" column carefully — the decline is not contention, it's
hyperthreading.** An `m5.2xlarge` advertises 8 vCPU but has **4 physical cores** (`lscpu`:
`Thread(s) per core: 2`). Ray schedules one replica per vCPU, so 48 replicas is 24
physical cores of compute, not 48. Normalizing at the point where we verified the node
count (48 replicas across 6 nodes):

- **468 mol/s per `m5.2xlarge` node**
- **≈117 mol/s per physical core**, against **131 mol/s** measured single-threaded on an
  idle core — i.e. the CPU tier runs at **~89% of ideal in real cores.**

So the tier is scaling nearly perfectly in the unit that actually costs money; the
apparent falloff is an artifact of counting hyperthreads as cores. (Caveat: we verified
node count only at the 48-replica point. A clean per-node scaling curve — 1, 2, 4, 6
nodes with placement pinned — is a short follow-up run, listed in next steps.)

## Result 3 — warm replicas, loaded once (your #2 ask)

Your "cold start, cold start" is the offline kMoL `predict` path **reloading all 5
checkpoints on every request**. Both tiers now load once in long-lived replicas and run a
real warm-up pass before Ray Serve marks them healthy, with `min_replicas ≥ 1` so a
request never pays the reload cost. That single change is most of the gap between "looks
perpetually cold" and "hot and fast."

---

## In your units: ms / molecule, and the 4× bar

| Setup | topology | molecule set | throughput | **ms/molecule** |
|---|---|---|---:|---:|
| **Your box, using 8 CPUs** | kMoL `predict`, reload-per-call, batch≈1 | yours | ~83 mol/s | **12 ms** |
| **Your box, using the 1 GPU** | kMoL `predict`, reload-per-call, on GPU | yours | ~200 mol/s | **5 ms** |
| _your "worth it" bar (4× faster)_ | _target_ | — | — | _~1.3 ms_ |
| **Served — two-stage Ray Serve, 1 L4 + 6 CPU nodes** | Ingress → 48 CPU featurizer replicas → 1 L4 forward tier | **real 15,751 library** | **2,809 mol/s** | **0.36 ms** |
| Served — same, 2 L4 | evidence the GPU isn't the limit | real library | 2,892 mol/s | 0.35 ms |
| _GPU forward ceiling_ | _pre-featurized batch replayed on the GPU — headroom, not a workload_ | _10-molecule set_ | _60,101 mol/s_ | _0.017 ms_ |

**The single biggest opportunity is the box you already own.** You described it as ~64
vCPU with 1 GPU, and the two baselines above used **8 CPUs** or **the 1 GPU**. Either way
**~56 of those 64 vCPU are sitting idle** — and featurization, the actual bottleneck, is
embarrassingly parallel across exactly those cores.

64 vCPU is ~32 physical cores. At the **117 mol/s per physical core** we measured, that
box projects to **≈3,700 mol/s (~0.27 ms/molecule)** without buying anything — against
the ~200 mol/s you see today. Two honest flags on that figure: it is a **projection** from
our measured per-core rate, not a run on your hardware, and it assumes featurization can
keep all 32 cores busy (which is what the two-stage split exists to do). It is the first
thing worth verifying with your real checkpoints.

That also reframes the "is the GPU worth it?" question. The GPU was never the problem —
one L4 has roughly 20× more forward capacity than a 64-vCPU box can feed it. You weren't
choosing between 8 CPUs and 1 GPU; you were using ~12% of the machine either way.

**Cost.** 6 × `m5.2xlarge` + 1 × `g6.2xlarge` at us-west-2 on-demand list is ≈ $3.28/hr
for 2,809 mol/s ≈ 10.1 M molecules/hr → **≈ $0.32 per million molecules** on this library.
Spot and a larger CPU:GPU ratio both push that down.

### Predicting the rate for *your* library

Featurization time is close to linear in molecule size across our library:

> **featurize_ms ≈ 0.285 × (heavy atoms) − 3.89**  (r² = 0.86, valid over ~10–120 heavy
> atoms; don't extrapolate the negative intercept to very small molecules)

Per size bucket, single-core:

| heavy atoms | n | mean featurize | single-core mol/s |
|---|---:|---:|---:|
| < 15 | 3,448 | 1.96 ms | 510 |
| 15–25 | 4,502 | 3.12 ms | 321 |
| 25–40 | 2,559 | 4.29 ms | 233 |
| 40–60 | 925 | 8.39 ms | 119 |
| 60+ | 4,317 | 18.75 ms | 53 |

*Source: [`molecule_library_stats.json`](molecule_library_stats.json).* Tell us your
library's size distribution and we can predict your throughput and cost directly, rather
than quoting one context-free number.

### Your own test molecules

We ran the three drugs you sanity-check with — minoxidil, Viagra (sildenafil), Lipitor
(atorvastatin) — end-to-end through the ported ensemble. Two of the three are already in
our library (they come from tox21):

| Drug | heavy atoms | featurize time | in our library? |
|---|---:|---:|---|
| minoxidil | 15 | 2.7 ms | no |
| sildenafil (Viagra) | 33 | 5.2 ms | yes (as the citrate salt) |
| atorvastatin (Lipitor) | 41 | 6.2 ms | yes (tox21) |

Each produces a full 12-target tox21 prediction plus per-target variance. Sizes and
timings are real; prediction *values* aren't meaningful yet (synthetic weights).

---

## How it's built — one Ray Serve application, three tiers

```
HTTP  →  Ingress            →  Featurizer (CPU tier)      →  GpuForward (GPU tier)
         thin; no torch        SMILES → graph; no GPU,       batched forward on L4
         1 vCPU, N replicas    1 vCPU each, autoscales       whole or fractional GPU
                               6 → 64 on cheap CPU nodes     1 → 4
```

The point of the split: **the `Featurizer` deployment requests no GPU**, so Ray Serve
schedules it onto plain CPU nodes and autoscales it independently of the GPU tier. Each
tier has its own `autoscaling_config`. That is what a single-process model server cannot
do, and it's what turns "featurization-bound" from a problem into the cost lever.

It also explains why our own earlier one-deployment version plateaued: there, every
replica asked for a GPU slice (`num_gpus=0.16`), which pins it to a GPU node — so the
featurizer count was silently capped by that node's vCPU, for a stage that needs no GPU
at all.

Code: [`serve_pipeline_app.py`](serve_pipeline_app.py) ·
benchmark driver [`scripts/serve_pipeline_bulk.py`](scripts/serve_pipeline_bulk.py) ·
deploy configs [`service.yaml`](service.yaml) / [`service.image.yaml`](service.image.yaml)
(both require the bearer token Anyscale issues — `query_auth_token_enabled`).

---

## Correctness — port fidelity (not an accuracy claim yet)

To run on modern GPUs we ported the molecule path from kMoL's frozen `Python 3.9 / torch
1.13 / CUDA 11.7` stack to current PyTorch — the legacy stack **cannot run on an L4 at
all**. The port reuses kMoL's exact architecture and checkpoint format, and produces
**numerically identical outputs** for the same weights:

| Comparison | max abs difference |
|---|---:|
| Port vs original kMoL (CPU) | **0.000e+00** (bit-for-bit) |
| Port vs original kMoL (L4 GPU) | **1.14e-05** (ordinary float precision) |

This proves the reimplementation is faithful. It is **not** an accuracy result — these
runs use synthetic weights. Confirming accuracy on your real checkpoints is step 1 below.

---

## Honest caveats & next steps

**Caveats.**
- **Synthetic weights.** The single biggest gap; everything above is throughput and
  fidelity, not predictive accuracy.
- **Load is generated inside the cluster**, by Ray actors calling the service — much
  better than the single-process client we started with, but still not an external
  open-loop HTTP load test. Treat p50/p99 as indicative.
- **No per-node scaling curve on the real library yet.** We verified node count at one
  point (48 replicas / 6 nodes). The 1→4 GPU near-linear result we reported earlier was
  measured with raw Ray actors on the 10-molecule set and is not restated here.
- **Latencies are bulk-request latencies** (256 molecules per request, ~4–5 s). Single-
  molecule interactive latency is a different measurement we haven't tuned for.

**To turn this into the validated prototype you described:**
1. **Drop in your real trained checkpoints** → one-command parity check plus a throughput
   sample. Nothing else changes.
2. **Per-node scaling curve + external open-loop load test** on your actual request mix
   (interactive single-molecule vs bulk screening) → defensible p50/p99 and a clean
   "N nodes → N× throughput" claim.
3. **Cost model against your library's size mix** — the bucket table above plus your
   distribution gives $ per million molecules directly.

---

## Reproduce / artifacts

Everything is in the self-contained [`port/`](.) bundle; step-by-step commands are in
[`REPRODUCE.md`](REPRODUCE.md). Each number above links to the script and the raw
`*_results.json` that produced it.

*Hardware: NVIDIA L4 (`g6.2xlarge`) for the GPU tier, `m5.2xlarge` (4 physical cores /
8 vCPU) for the CPU tier. Model: kMoL ensemble, 5 × GraphConvolutionalNetwork, 264,300
params each.*
