# kmolport — kMoL molecule ensemble ported to modern PyTorch

`kmolport` is a minimal, self-contained port of kMoL's **molecule GNN inference
path** to Python 3.11 + current PyTorch + PyTorch-Geometric. It exists so the
5-model ensemble runs as ordinary Ray Serve actors on Anyscale's managed cluster
with native GPU autoscaling — no frozen py3.9 stack, no isolated Ray, no NFS conda
staging, no container gymnastics (see `../plan.md` for why the old approach dead-ended).

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

## Proven results (2026-07-23, workspace `expwrk_9e9qajmqr7w6astmetm8v9tv9s`)

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

## How to reproduce

```bash
# 1) reference logits from real kMoL (py3.9 conda env, cwd = kmol_serving)
PYTHONPATH=kmol/src:stubs python port/scripts/ref_logits.py \
    configs/ensemble_serve.cpu.json ref_logits.json

# 2) parity on py3.11 (torch + torch_geometric + rdkit installed)
PYTHONPATH=port python port/scripts/port_check.py \
    --config configs/ensemble_serve.example.json \
    --ckpt-dir checkpoints --ref ref_logits.json --device cpu

# 3) GPU parity + throughput on an autoscaled L4 (driver on the workspace head)
python port/scripts/gpu_run.py    # submits a num_gpus=1 Ray task; writes gpu_results.json
```
