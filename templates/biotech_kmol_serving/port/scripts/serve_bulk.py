"""P1 (bulk): served screening throughput on ONE L4.

Same 6-replica fractional-GPU deployment as serve_run.py, but each request carries
a CHUNK of molecules (how a screen actually calls it). This amortizes per-RPC
overhead — the bottleneck in the single-SMILES test (REC 4: don't benchmark the
client) — so the number reflects the replicas' parallel featurization + GPU forward.
"""
import asyncio
import json
import os
import time

import ray
from ray import serve

SHIP_DIR = "/home/ray/default/kmol_ship"
CONFIG = "config.json"
CKPT_DIR = "checkpoints"
OUT = "/home/ray/default/kmol_port/serve_bulk_results.json"

NUM_GPUS = float(os.environ.get("KMOL_NUM_GPUS", "0.16"))
NUM_REPLICAS = int(os.environ.get("KMOL_REPLICAS", "6"))
TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2"]}

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]


@serve.deployment(
    ray_actor_options={"num_gpus": NUM_GPUS, "num_cpus": 1, "runtime_env": TASK_PIP},
    autoscaling_config={"min_replicas": NUM_REPLICAS, "max_replicas": NUM_REPLICAS},
    max_ongoing_requests=64,
)
class KmolEnsemble:
    def __init__(self, config_path, ckpt_dir):
        import torch

        import kmolport
        from kmolport.featurizer import collate

        torch.set_num_threads(1)
        self._torch = torch
        self._collate = collate
        cfg = kmolport.load_config(config_path)
        self.model = kmolport.build_ensemble(cfg, checkpoint_dir=ckpt_dir, device="cuda")
        self.feat = kmolport.GraphFeaturizer()
        self._infer_sync(["CCO"])  # warm-up

    def _infer_sync(self, smiles_list):
        torch = self._torch
        data = [self.feat.featurize(s) for s in smiles_list]
        batch = self._collate(data).to("cuda")
        with torch.no_grad():
            out = self.model({"graph": batch})
        logits = out["logits"].detach().cpu().numpy()
        return {"n": len(smiles_list), "logits_shape": list(logits.shape)}

    async def infer_bulk(self, smiles_list):
        return await asyncio.to_thread(self._infer_sync, smiles_list)


async def bench_bulk(handle, total, chunk, concurrency):
    reqs = [[SMILES_POOL[(j) % len(SMILES_POOL)] for j in range(chunk)]
            for _ in range(total // chunk)]
    sem = asyncio.Semaphore(concurrency)
    lat = []

    async def one(lst):
        async with sem:
            t = time.perf_counter()
            await handle.infer_bulk.remote(lst)
            lat.append(time.perf_counter() - t)

    t0 = time.perf_counter()
    await asyncio.gather(*[one(r) for r in reqs])
    dt = time.perf_counter() - t0
    lat.sort()
    return {
        "total_mols": total, "chunk": chunk, "concurrency": concurrency,
        "wall_s": dt, "mol_s": total / dt,
        "req_p50_ms": lat[len(lat) // 2] * 1e3,
        "req_p99_ms": lat[min(int(len(lat) * 0.99), len(lat) - 1)] * 1e3,
    }


def main():
    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    app = KmolEnsemble.bind(CONFIG, CKPT_DIR)
    print(f"deploying {NUM_REPLICAS} replicas @ num_gpus={NUM_GPUS} ...", flush=True)
    handle = serve.run(app, name="kmol")
    for _ in range(6):
        handle.infer_bulk.remote(["CCO"] * 32).result()

    results = []
    for chunk, conc in [(64, 12), (128, 12), (128, 24), (256, 24), (256, 48)]:
        r = asyncio.new_event_loop().run_until_complete(
            bench_bulk(handle, 24000, chunk, conc))
        results.append(r)
        print(f"chunk={chunk:>4} conc={conc:>3}  {r['mol_s']:>10,.0f} mol/s  "
              f"req_p50={r['req_p50_ms']:.0f}ms req_p99={r['req_p99_ms']:.0f}ms", flush=True)

    peak = max(r["mol_s"] for r in results)
    out = {"num_replicas": NUM_REPLICAS, "num_gpus_per_replica": NUM_GPUS,
           "results": results, "peak_mol_s": peak, "x_baseline": peak / 170}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPEAK served (bulk): {peak:,.0f} mol/s = {peak/170:.1f}x the 170/s baseline")
    print(f"wrote {OUT}")
    serve.shutdown()


if __name__ == "__main__":
    main()
