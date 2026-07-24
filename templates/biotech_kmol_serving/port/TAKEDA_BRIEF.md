# Serving the kMoL GNN ensemble on Anyscale — findings

**Prepared by Anyscale · for the Takeda / kMoL team · 2026-07-24**

> **Scope of these numbers.**
>
> - **Molecule set.** 15,751 unique, RDKit-sanitized structures: 7,831 from tox21, 3,000
>   sampled from ZINC, 4,920 from ChEMBL with MW > 800. Heavy-atom count median 24, mean
>   41, p95 107, max 343. The ChEMBL third is a deliberate large-molecule tail, so this set
>   is likely larger-molecule than a typical screening deck.
> - **Weights are synthetic** — correct architecture and format, random values. Throughput
>   is unaffected; prediction *values* are not meaningful.
> - **Earlier figures are superseded.** A previous version of this brief measured
>   throughput on a 10-molecule set whose members average 12.1 heavy atoms. Those
>   molecules featurize in 2.46 ms versus 7.65 ms for this set, making those figures 3.1×
>   higher. Numbers below are on the 15,751-molecule set unless labelled otherwise.

---

## Context from the call

Three issues were described:

- Horizontal scaling was sublinear: ~150–200 molecules/sec on one machine, and a second
  machine produced roughly 120 rather than ~400.
- Replicas reported repeated cold starts under sustained traffic.
- The GPU's benefit was unclear: 8 CPUs measured ~12 ms/molecule, the 1 GPU ~5 ms/molecule
  — about 2.4×, against a stated bar of 4× to justify the GPU.

The hardware described was a single instance with ~64 vCPU and 1 GPU. Both the 12 ms and
the 5 ms figures were configurations on that same instance.

---

## Finding 1 — the workload is CPU-featurization-bound

Two components were doubled independently. Neither changed throughput materially.

| Configuration | throughput | change |
|---|---:|---:|
| 1 × L4, 48 CPU featurizer replicas | 2,809 mol/s | — |
| 2 × L4, same CPU tier | 2,892 mol/s | +3.0% |
| 1 × L4, 3× ingress replicas and 2× load clients | 1,611 mol/s | −9% |

*Sources: [`serve_pipeline_results.json`](serve_pipeline_results.json),
[`serve_pipeline_2gpu.json`](serve_pipeline_2gpu.json),
[`serve_pipeline_diag_ingress12.json`](serve_pipeline_diag_ingress12.json).*

Doubling GPU capacity added 3%, so the GPU is not the constraint. Tripling the serving
front door reduced throughput, so that is not the constraint either. The GNN forward
measures ~60,000 molecules/sec on one L4 in isolation — roughly 20× more than the CPU tier
supplies. The constraint is RDKit featurization on CPU cores.

## Finding 2 — throughput tracks the size of the CPU tier

One L4, varying only the CPU featurizer tier. Each replica occupies 1 vCPU on an
`m5.2xlarge`; requests carry 256 molecules.

| CPU featurizer replicas | throughput | per replica | req p50 | req p99 |
|---:|---:|---:|---:|---:|
| 6 | 631 mol/s | 105 | 4.6 s | 5.7 s |
| 12 | 956 mol/s | 80 | 4.1 s | 7.9 s |
| 24 | 1,775 mol/s | 74 | 3.9 s | 7.2 s |
| 48 | 2,809 mol/s | 59 | 4.9 s | 7.5 s |

The per-replica figure declines because `m5.2xlarge` presents 8 vCPU on **4 physical
cores** (`lscpu`: `Thread(s) per core: 2`), and Ray schedules one replica per vCPU. 48
replicas is therefore 24 physical cores of compute, not 48. At the one point where node
count was verified (48 replicas across 6 nodes):

- **468 mol/s per `m5.2xlarge`**
- **≈117 mol/s per physical core**, against **131 mol/s** measured single-threaded on an
  idle core — **~89% of the single-threaded rate.**

Measured in physical cores the tier scales close to proportionally. Node count was
verified only at the 48-replica point; a per-node scaling curve with placement pinned has
not been run.

## Finding 3 — checkpoint loading

The repeated cold starts correspond to the offline kMoL `predict` path loading all 5
checkpoints per call. In this implementation both tiers load once in long-lived replicas
and run a warm-up pass before Ray Serve marks them healthy, with `min_replicas ≥ 1`. A
request does not pay checkpoint-load cost.

---

## Throughput in ms/molecule

| Setup | topology | molecule set | throughput | ms/molecule |
|---|---|---|---:|---:|
| Your box, 8 CPUs | kMoL `predict`, load-per-call, batch≈1 | yours | ~83 mol/s | 12 ms |
| Your box, 1 GPU | kMoL `predict`, load-per-call, on GPU | yours | ~200 mol/s | 5 ms |
| _stated 4× bar_ | _reference_ | — | — | _~1.3 ms_ |
| Two-stage Ray Serve, 1 L4 + 6 CPU nodes | Ingress → 48 CPU featurizer replicas → 1 L4 | 15,751 set | 2,809 mol/s | 0.36 ms |
| Same, 2 L4 | GPU-sensitivity check | 15,751 set | 2,892 mol/s | 0.35 ms |
| _GPU forward ceiling_ | _pre-featurized batch replayed; headroom, not a workload_ | _10-molecule set_ | _60,101 mol/s_ | _0.017 ms_ |

**Per-core comparison.** The 5 ms/molecule figure corresponds to ~200 mol/s. Measured here
is ~117 mol/s per physical core on a larger-molecule set. A 64-vCPU instance is ~32
physical cores, which at that rate **projects to ≈3,700 mol/s (~0.27 ms/molecule)**. Two
qualifications: this is a projection from a per-core rate, not a measurement on your
hardware, and it assumes featurization can occupy all 32 cores, which is what the
two-stage split is for. The configurations measured on the call used 8 CPUs or the 1 GPU,
so ~56 of the 64 vCPU were not doing featurization work.

**Cost.** 6 × `m5.2xlarge` plus 1 × `g6.2xlarge` at us-east/us-west on-demand list is
≈$3.28/hr for 2,809 mol/s ≈ 10.1 M molecules/hr, i.e. **≈$0.32 per million molecules** on
this molecule set. Spot pricing and a higher CPU:GPU ratio both reduce it.

### Featurization cost versus molecule size

Featurization time is approximately linear in molecule size:

> **featurize_ms ≈ 0.285 × (heavy atoms) − 3.89**  (r² = 0.86, fit over ~10–120 heavy
> atoms; the negative intercept is a fit artifact and does not extrapolate to small
> molecules)

Single-core rates by size bucket, over all 15,751 molecules:

| heavy atoms | n | mean featurize | single-core mol/s |
|---|---:|---:|---:|
| < 15 | 3,448 | 1.96 ms | 510 |
| 15–25 | 4,502 | 3.12 ms | 321 |
| 25–40 | 2,559 | 4.29 ms | 233 |
| 40–60 | 925 | 8.39 ms | 119 |
| 60+ | 4,317 | 18.75 ms | 53 |
| **full mix** | **15,751** | **7.65 ms** (median 3.71) | **131** (median 270) |

Aggregate throughput follows the mean, since total time is the sum of per-molecule times;
the median is closer to what one request experiences. The 60+ bucket is 27% of this set
and dominates the mean. A drug-like-only deck would sit nearer 300 mol/s/core. Given your
library's size distribution, these rates predict its throughput directly.
*Source: [`molecule_library_stats.json`](molecule_library_stats.json).*

### The three test molecules

The three compounds mentioned as routine test inputs, run end-to-end through the ported
ensemble. Two are present in the 15,751-molecule set, via tox21.

| Compound | heavy atoms | featurize time | in the set? |
|---|---:|---:|---|
| minoxidil | 15 | 2.7 ms | no |
| sildenafil (Viagra) | 33 | 5.2 ms | yes (as the citrate salt) |
| atorvastatin (Lipitor) | 41 | 6.2 ms | yes (tox21) |

Each returns a 12-target tox21 prediction plus per-target variance. Sizes and timings are
measured; prediction values are not meaningful under synthetic weights. Note these three
average 29.7 heavy atoms against the set's mean of 41, implying ~213 mol/s/core — between
the 10-molecule set (407) and the full mix (131).

---

## Implementation

One Ray Serve application, three deployments, each with its own scaling configuration:

```
HTTP  →  Ingress            →  Featurizer (CPU tier)      →  GpuForward (GPU tier)
         no torch              SMILES → graph; no GPU        batched forward on L4
         1 vCPU, N replicas    1 vCPU each, 6 → 64           whole or fractional GPU, 1 → 4
```

The `Featurizer` deployment requests no GPU, so Ray Serve places it on CPU-only nodes and
scales it independently of the GPU tier. An earlier single-deployment version gave every
replica a `num_gpus=0.16` slice; because a GPU slice pins a replica to a GPU node, the
featurizer count was bounded by that node's vCPU even though the stage needs no GPU.

Code: [`serve_pipeline_app.py`](serve_pipeline_app.py) · driver
[`scripts/serve_pipeline_bulk.py`](scripts/serve_pipeline_bulk.py) · deploy configs
[`service.yaml`](service.yaml) and [`service.image.yaml`](service.image.yaml), both with
`query_auth_token_enabled` (requests carry the bearer token Anyscale issues).

---

## Port fidelity

The molecule path was ported from kMoL's `Python 3.9 / torch 1.13 / CUDA 11.7` stack to
current PyTorch, because the original stack does not run on an L4. The port reuses kMoL's
architecture and checkpoint format and produces numerically identical outputs for the same
weights:

| Comparison | max abs difference |
|---|---:|
| Port vs kMoL (CPU) | 0.000e+00 (bit-for-bit) |
| Port vs kMoL (L4 GPU) | 1.14e-05 (float precision) |

This establishes code equivalence. It is not an accuracy result — these runs use synthetic
weights.

---

## Model weights

Every number in this brief was produced with **synthetic weights** — correct architecture
and checkpoint format, random values. Throughput and latency are unaffected, since the
tensor shapes and the compute are identical either way, but the prediction *values* carry
no information.

Synthetic weights were used because kMoL ships none: every config under `data/configs/`
has `checkpoint_path: null`, and the repository contains no trained checkpoints for this
architecture (the only `.pt`/`.pkl` files are SchNet featurizer test fixtures). The five
weight files are produced by kMoL's own `cross_validation_folds: 5` training run, per
dataset. There is also no usable public substitute — the loader requires kMoL's exact
`state_dict` keys for 5 × `GraphConvolutionalNetwork`, which no HuggingFace molecular
model provides.

Two ways to get real predictions, and they are independent:

**A. Your trained checkpoints — no code changes.** Put `model_0.pt` … `model_4.pt` in
`port/checkpoints/` and restart. The port preserves kMoL's checkpoint format
(`torch.save({"model": state_dict})`) and its exact class and attribute names, which is
why kMoL `state_dict`s load unchanged. Production load uses `strict=True`, so a checkpoint
that does not match this architecture fails at startup rather than silently serving a
partly-random model. `scripts/port_check.py` then re-runs the parity check on your weights.

**B. We train on public tox21.** kMoL ships both the recipe and the data: the `model`
block of `data/configs/model/tox21.json` is byte-identical to a member of our ensemble
config, and `data/datasets.zip` contains `tox21.csv` — 7,831 molecules, 12 sparse assay
targets. `scripts/train_tox21.py` reproduces that recipe verbatim (AdamW lr 0.01 / weight
decay 5.6e-4, OneCycleLR, masked BCE, batch 128, 200 epochs) using kMoL's MolWt-stratified
80/20 split, training the five members concurrently as fractional-GPU tasks. On one L4
that is minutes, and it yields a per-target ROC-AUC we can report. It produces a real
model on public data — **not** a model of your endpoints, so it demonstrates realistic
outputs rather than your accuracy.

Option A is the one that answers "does this reproduce our numbers". Option B only makes
the demo's outputs plausible while waiting.

## Limitations

- **Synthetic weights.** No accuracy claim is made or implied. See "Model weights" above
  for the two paths to real predictions.
- **Load generated inside the cluster** by Ray actors calling the service, not by an
  external open-loop HTTP load generator. p50/p99 are indicative.
- **No verified per-node scaling curve** on this molecule set. Node count was confirmed at
  one configuration (48 replicas / 6 nodes). The earlier 1→4 GPU near-linear result was
  measured with raw Ray actors on the 10-molecule set and is not restated here.
- **Latencies are bulk-request latencies** — 256 molecules per request, ~4–5 s. Single-
  molecule interactive latency is a separate measurement.
- **The 15,751-molecule set is not bundled** with this code; the code reads it from a path
  given by `KMOL_LIBRARY`. Its summary statistics are in
  `molecule_library_stats.json`.

## Possible next steps

1. **Your `model_0..4.pt` in `port/checkpoints/`** → we re-run the parity check and a
   throughput sample on real weights. No code changes; see "Model weights" option A.
2. **Your library's molecule-size distribution** (a heavy-atom-count histogram is enough)
   → the per-bucket rates above convert directly into expected throughput, and therefore
   cost per million molecules, for your actual deck rather than our molecule mix.

Then, on our side:

3. Per-node scaling curve plus an external open-loop load test on your request mix
   (single-molecule interactive versus bulk screening), for defensible p50/p99 and a
   measured N-nodes-to-N×-throughput claim — the linear-scaling question from the call.
4. Optionally, train on public tox21 (option B above) so the demo returns plausible
   predictions while (1) is in flight.

---

## Reproducing

The [`port/`](.) bundle is self-contained; commands are in [`REPRODUCE.md`](REPRODUCE.md).
Each number above links to the script and the raw `*_results.json` behind it.

*Hardware: NVIDIA L4 (`g6.2xlarge`) GPU tier; `m5.2xlarge` (4 physical cores / 8 vCPU) CPU
tier. Model: kMoL ensemble, 5 × GraphConvolutionalNetwork, 264,300 parameters each.*
