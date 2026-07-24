"""GPU parity + throughput on an autoscaled L4, native on the managed cluster.

Driver runs on the head; submits a num_gpus=1 task whose runtime_env pip-installs
CUDA torch + PyG + rdkit. The `port/` bundle is uploaded as the Ray working_dir so
`import kmolport` and the config/checkpoints resolve on the worker. The autoscaler
brings up a g6.2xlarge (L4). No isolated Ray, no NFS conda, no container.
"""
import json
import os
import time

import ray

# The bundle (this file's grandparent) is uploaded to the GPU worker as the Ray
# working_dir; CONFIG/CKPT_DIR/REF are resolved relative to it on the worker. All
# overridable by env. OUT is written by the driver on the head (cwd by default).
_BUNDLE = os.environ.get("KMOL_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIP_DIR = os.environ.get("KMOL_SHIP_DIR", _BUNDLE)
CONFIG = os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
CKPT_DIR = os.environ.get("KMOL_CKPT_DIR", "checkpoints")
REF = os.environ.get("KMOL_REF", "ref_logits.json")   # optional; parity skipped if absent
OUT = os.environ.get("KMOL_OUT", "gpu_results.json")

# working_dir must be set at the job level (ray.init); tasks inherit it. Only the
# pip layer goes on the task's runtime_env.
# Default PyPI linux torch wheel is the CUDA 12.1 build — runs on L4 (sm_89).
TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2"]}

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
ITERS = 50
WARMUP = 10


@ray.remote(num_gpus=1, runtime_env=TASK_PIP, max_retries=0)
def gpu_task(config_path, ckpt_dir, ref_path):
    import os

    import numpy as np
    import torch

    import kmolport
    from kmolport.featurizer import collate

    res = {}
    res["gpu"] = torch.cuda.get_device_name(0)
    res["torch"] = torch.__version__
    res["cuda_avail"] = torch.cuda.is_available()

    dev = "cuda"
    cfg = kmolport.load_config(config_path)
    model = kmolport.build_ensemble(cfg, checkpoint_dir=ckpt_dir, device=dev)
    feat = kmolport.GraphFeaturizer()
    res["n_models"] = len(model.models)
    res["params_per_sub"] = sum(p.numel() for p in model.models[0].parameters())

    # ---- parity on GPU vs the py3.9 kMoL reference (optional) ----
    # Produce ref_logits.json first with scripts/ref_logits.py in your kMoL env; if
    # it's absent we skip parity and just measure throughput.
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))
        logits, _ = kmolport.predict(model, feat, ref["smiles"], device=dev)
        logits = logits.detach().cpu().numpy()
        diff = np.abs(logits - np.array(ref["logits"]))
        res["parity_max_abs_diff"] = float(diff.max())
        res["parity_mean_abs_diff"] = float(diff.mean())
        res["parity_pass"] = bool(diff.max() < 1e-2)  # looser on GPU (fp nondeterminism)
    else:
        res["parity_max_abs_diff"] = None
        res["parity_pass"] = "skipped (no ref_logits.json)"

    def make_batch(n):
        data = [feat.featurize(SMILES_POOL[i % len(SMILES_POOL)]) for i in range(n)]
        return collate(data).to(dev)

    # ---- forward-only throughput (GPU-bound) ----
    fwd = {}
    with torch.no_grad():
        for bs in BATCH_SIZES:
            batch = make_batch(bs)
            for _ in range(WARMUP):
                model({"graph": batch})
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                model({"graph": batch})
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / ITERS
            fwd[bs] = {"mol_s": bs / dt, "ms_per_batch": dt * 1e3, "ms_per_mol": dt / bs * 1e3}
    res["forward_only"] = fwd

    # ---- end-to-end (featurize + collate + forward) ----
    e2e = {}
    with torch.no_grad():
        for bs in [64, 128, 256, 512]:
            for _ in range(3):
                model({"graph": make_batch(bs)})
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(ITERS):
                model({"graph": make_batch(bs)})
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / ITERS
            e2e[bs] = {"mol_s": bs / dt, "ms_per_mol": dt / bs * 1e3}
    res["end_to_end"] = e2e

    res["peak_forward_mol_s"] = max(v["mol_s"] for v in fwd.values())
    return res


def main():
    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    print("submitting GPU task (autoscaler will bring up an L4)...", flush=True)
    t0 = time.perf_counter()
    res = ray.get(gpu_task.remote(CONFIG, CKPT_DIR, REF))
    res["wall_seconds_incl_autoscale_and_install"] = time.perf_counter() - t0
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)

    print("\n===== GPU RESULTS =====")
    print(f"gpu={res['gpu']}  torch={res['torch']}  models={res['n_models']}  "
          f"params/sub={res['params_per_sub']:,}")
    if res["parity_max_abs_diff"] is None:
        print(f"PARITY on GPU: {res['parity_pass']}")
    else:
        print(f"PARITY on GPU: max_abs_diff={res['parity_max_abs_diff']:.3e}  "
              f"{'PASS' if res['parity_pass'] else 'FAIL'}")
    print(f"\n{'batch':>6} {'mol/s (fwd)':>14} {'ms/batch':>10}")
    for bs, v in res["forward_only"].items():
        print(f"{bs:>6} {v['mol_s']:>14,.0f} {v['ms_per_batch']:>10.2f}")
    print(f"\nend-to-end (feat+forward):")
    for bs, v in res["end_to_end"].items():
        print(f"{bs:>6} {v['mol_s']:>14,.0f} mol/s")
    peak = res["peak_forward_mol_s"]
    print(f"\nPEAK forward-only: {peak:,.0f} mol/s  => {peak/170:.1f}x the 170/s baseline")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
