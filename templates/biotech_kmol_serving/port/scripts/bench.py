"""Throughput microbench for the ported kMoL ensemble (pure torch, no Ray).

Reports batched forward throughput vs batch size. Also reports end-to-end
throughput (RDKit featurization + collate + forward) so the CPU featurization
cost is visible separately from GPU forward capability.
"""
import argparse
import time

import torch

import kmolport
from kmolport.featurizer import collate

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512]
ITERS = 30
WARMUP = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    print(f"device={dev}  torch={torch.__version__}  "
          f"gpu={torch.cuda.get_device_name(0) if dev == 'cuda' else 'n/a'}")

    cfg = kmolport.load_config(args.config)
    model = kmolport.build_ensemble(cfg, checkpoint_dir=args.ckpt_dir, device=dev)
    feat = kmolport.GraphFeaturizer()
    print(f"ensemble models={len(model.models)}  "
          f"params/sub={sum(p.numel() for p in model.models[0].parameters()):,}")

    def make_batch(n):
        data = [feat.featurize(SMILES_POOL[i % len(SMILES_POOL)]) for i in range(n)]
        return collate(data).to(dev)

    print("\n--- forward-only (GPU-bound; batch pre-featurized) ---")
    print(f"{'batch':>6} {'mol/s':>12} {'ms/mol':>10} {'ms/batch':>10}")
    fwd_curve = {}
    with torch.no_grad():
        for bs in BATCH_SIZES:
            batch = make_batch(bs)
            for _ in range(WARMUP):
                model({"graph": batch})
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                model({"graph": batch})
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / ITERS
            fwd_curve[bs] = bs / dt
            print(f"{bs:>6} {bs/dt:>12,.0f} {dt/bs*1e3:>10.3f} {dt*1e3:>10.2f}")

    print("\n--- end-to-end (featurize + collate + forward) ---")
    print(f"{'batch':>6} {'mol/s':>12} {'ms/mol':>10}")
    with torch.no_grad():
        for bs in [32, 64, 128, 256]:
            # warm featurizer caches
            for _ in range(2):
                b = make_batch(bs)
                model({"graph": b})
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                b = make_batch(bs)
                model({"graph": b})
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / ITERS
            print(f"{bs:>6} {bs/dt:>12,.0f} {dt/bs*1e3:>10.3f}")

    best = max(fwd_curve.values())
    print(f"\nPEAK forward-only throughput: {best:,.0f} mol/s  (vs baseline ~170/s "
          f"=> {best/170:.1f}x)")


if __name__ == "__main__":
    main()
