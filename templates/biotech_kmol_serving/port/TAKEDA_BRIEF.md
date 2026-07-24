# Serving the kMoL GNN ensemble at scale — throughput brief

**Prepared by Anyscale · for the Takeda / kMoL team · 2026-07-24**

## Summary

We looked at the throughput of serving your kMoL 5-model molecule GNN ensemble and
found that the previously observed **~170 molecules/sec** was not a GPU limit — it was
per-call overhead on the offline `predict` path at batch size 1. After porting the
molecule inference path to a current PyTorch and running it with batching on Anyscale,
a **single NVIDIA L4 GPU** processes the ensemble far faster, and throughput scales
almost perfectly as GPUs are added:

| What we measured (one L4 unless noted) | Throughput | vs ~170/s |
|---|---:|---:|
| GPU forward pass, batched (batch 1024) | 60,101 mol/s | **353×** |
| End-to-end served (Ray Serve, 6 replicas) | 1,697 mol/s | **10×** |
| End-to-end pipeline (12 CPU featurizers → 1 GPU) | 3,904 mol/s | **23×** |
| **4× L4, near-linear scaling** | **8,792 mol/s** | **52×** |

The ported model produces **numerically identical** outputs to the original kMoL
(bit-for-bit on CPU; 1.1e-05 max difference on GPU), so this is the same model — just
running on a modern, supported stack.

## Background — the throughput question

kMoL's offline `predict` rebuilds a `Predictor` (reloading all 5 checkpoints) and a
`GeneralStreamer` (reloading/splitting the dataset) **on every call**. At batch size ~1
that overhead dwarfs the actual GNN forward of a small model (~264K parameters per
sub-model). The hypothesis was that loading the ensemble once and batching the forward
passes would blow past the ~170/s figure. It does — by a wide margin.

## What we changed

The blocker was environment, not the model. kMoL pins **Python 3.9 / PyTorch 1.13 /
CUDA 11.7**, which cannot share a cluster with Anyscale's modern Ray, and (as we
confirmed) does not run on current L4 GPUs at all. So we **ported the molecule inference
path to Python 3.11 + current PyTorch + PyTorch-Geometric**.

Key points about the port:

- It is a **faithful, minimal extraction** of only the molecule path — the graph
  featurizer (RDKit atom/bond features + the 17 RDKit descriptors), the
  `GraphConvolutionalNetwork` (LEConv ×7), and the `EnsembleNetwork` (mean + variance).
  It reuses kMoL's **exact class structure**, so your trained `state_dict` checkpoints
  load unchanged.
- It depends only on `torch`, `torch_geometric`, and `rdkit` — **no openbabel, prody,
  openfold, or torch_scatter**, and no CUDA-kernel compilation. (The only custom CUDA
  kernel in kMoL is OpenFold's, on the protein/MSA path, which molecule serving never
  touches.)
- It runs as **ordinary Ray Serve actors** on Anyscale with native GPU autoscaling — no
  separate Ray cluster, no NFS staging, no container gymnastics.

## Correctness — it is the same model

We ran the same molecules through the original py3.9 kMoL and the port and compared the
raw logits:

| Comparison | Max abs. difference |
|---|---:|
| Port on CPU vs original kMoL (CPU) | **0.000e+00** (bit-for-bit) |
| Port on L4 GPU vs original kMoL | **1.14e-05** (normal float precision) |

The featurization, the architecture, and the checkpoint loading are all identical; the
tiny GPU difference is ordinary floating-point non-determinism.

## Throughput results in detail

**The GPU forward is nearly free.** Timing the batched forward on one L4, the time per
batch stays essentially flat (~15 ms) from batch 1 to batch 512, while throughput climbs
from 69 to 33,950 mol/s — and reaches 60,101 mol/s at batch 1024. In other words, the
per-call cost is fixed launch overhead; the actual GNN compute is negligible. This
confirms the "~5 ms/molecule" figure was a batch-size-1 artifact.

**End-to-end is bounded by featurization, not the GPU.** RDKit featurization (SMILES →
molecular graph) costs ~1.75 ms per molecule per CPU core. In a two-stage pipeline of 12
CPU featurizer workers feeding one GPU, throughput was 3,904 mol/s — within 4% of the
featurization-only rate — meaning **the L4 was only ~7% utilized**. The lever for more
throughput is CPU featurization capacity, not more GPU.

**Scaling is near-linear.** Because each GPU node also brings CPU cores, adding nodes
adds both featurization and GPU capacity together:

| GPUs (L4) | mol/s | speedup vs 1 GPU | efficiency |
|---:|---:|---:|---:|
| 1 | 2,235 | 1.00× | 100% |
| 2 | 4,209 | 1.88× | 94% |
| 3 | 6,500 | 2.91× | 97% |
| 4 | 8,792 | 3.93× | **98%** |

## Important caveat — checkpoints used

These measurements used **synthetically-initialized checkpoints** (the same architecture
and file format as your real ones; random weights). This does **not** affect the
throughput conclusions — throughput is determined by the architecture (layer count,
sizes, parameter count), which is identical, so the numbers hold for your trained models.
It does mean the **prediction values in these runs are not meaningful**. The natural
confirmation step is to drop in your real trained checkpoints (a direct file swap) and
re-run the parity check and a short throughput sample; we expect both to match what is
shown here.

## Recommendations / next steps

1. **Swap in the real trained checkpoints** and re-run the included parity + throughput
   checks — a one-command confirmation that the port reproduces your production outputs.
2. **Serve it as-is on Anyscale.** Batching + load-once already clears the target by 10×
   on one GPU; multi-GPU scales linearly.
3. **If you need more than one GPU's worth**, scale featurization first — add a CPU-only
   featurizer tier feeding a thin GPU forward stage. One L4 can absorb ~60k mol/s of
   forward, so throughput will keep climbing with featurizer cores.
4. **Deployment** is a standard container on a stock modern Ray image (a Dockerfile and
   Anyscale Service config are prepared) — no more conda/CUDA-11.7 build fragility.

## Reproducing / artifacts

All code and raw results are in the `port/` directory:

- `port/kmolport/` — the ported package (torch/PyG/rdkit only).
- `port/scripts/` — reference-logits, parity, forward microbench, Ray Serve, two-stage
  pipeline, and multi-GPU scaling drivers.
- `port/gpu_results.json`, `serve_bulk_results.json`, `pipeline_results.json`,
  `scale_results.json` — the raw measurements behind every number above.
- `port/Dockerfile`, `port/service.image.yaml`, `port/service.yaml` — deployment.

*Baseline reference: the previously reported ~170 mol/s. Hardware: NVIDIA L4
(g6.2xlarge). Model: kMoL ensemble, 5 × GraphConvolutionalNetwork, 264,300 params each.*
