# Serving Takeda's kMoL GNN ensemble on Anyscale

This template serves kMoL's 5-model molecule GNN ensemble as a batched, autoscaling
Ray Serve endpoint — by **porting the molecule inference path to modern PyTorch**
(`port/`), so it runs as ordinary Ray Serve actors with native GPU autoscaling.

**Everything lives in [`port/`](port/) — that's the self-contained, shareable bundle.**

- **[`port/README.md`](port/README.md)** — the technical writeup + all results
  (parity, single-GPU throughput, near-linear multi-GPU scaling).
- **[`port/TAKEDA_BRIEF.md`](port/TAKEDA_BRIEF.md)** — shareable brief.
- **[`port/REPRODUCE.md`](port/REPRODUCE.md)** — how to reproduce, end to end.
- `port/kmolport/` — the ported package (torch / torch_geometric / rdkit only).

## Why a port (the short version)

kMoL pins Python 3.9 / torch 1.13 / CUDA 11.7, which can't share a cluster with
Anyscale's modern Ray and doesn't run on current L4 GPUs. The molecule path uses only
standard PyTorch + PyTorch-Geometric ops (no custom CUDA kernels; the OpenFold kernel is
protein-only), and checkpoints are plain `state_dict`s — so a minimal port to Python 3.11
runs natively. It reproduces kMoL's outputs bit-for-bit and serves far above the prior
throughput. See [`plan.md`](plan.md) for the full plan and post-mortem.

> The pre-port approach (running kMoL's frozen conda stack directly) has been removed;
> it lives in git history if ever needed.
