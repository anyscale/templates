"""P1: ported kMoL ensemble on Ray Serve, native on the managed cluster.

Packs N fractional-GPU replicas on one L4 so RDKit featurization (the end-to-end
bottleneck found in P0) runs in parallel across replicas while all share the cheap
GPU forward. Deploys, benchmarks served throughput + latency via the deployment
handle at increasing concurrency, then tears down.
"""
import asyncio
import json
import os
import time

import ray
from ray import serve

SHIP_DIR = "/home/ray/default/kmol_ship"
CONFIG = "config.json"        # relative to working_dir on the replica
CKPT_DIR = "checkpoints"
OUT = "/home/ray/default/kmol_port/serve_results.json"

# 6 replicas on ONE L4: 6*0.16=0.96 GPU, 6 CPUs (g6.2xlarge has 8 vCPU) leaves
# headroom for the Serve proxy/raylet so a 2nd node isn't triggered — keeps this a
# genuine single-GPU number.
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
    max_ongoing_requests=256,
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
        self._infer.set_max_batch_size(int(os.environ.get("KMOL_MAX_BATCH", "128")))
        self._infer.set_batch_wait_timeout_s(float(os.environ.get("KMOL_WAIT_S", "0.01")))
        self._infer_sync(["CCO"])  # warm-up: featurize + collate + 5-model forward

    def _infer_sync(self, smiles_list):
        torch = self._torch
        data = [self.feat.featurize(s) for s in smiles_list]
        batch = self._collate(data).to("cuda")
        with torch.no_grad():
            out = self.model({"graph": batch})
        logits = out["logits"].detach().cpu().numpy()
        var = out["logits_var"].detach().cpu().numpy()
        return [{"logits": logits[i].tolist(), "variance": var[i].tolist()}
                for i in range(len(smiles_list))]

    @serve.batch(max_batch_size=128, batch_wait_timeout_s=0.01)
    async def _infer(self, smiles_list):
        return await asyncio.to_thread(self._infer_sync, smiles_list)

    async def infer(self, smiles):
        return await self._infer(smiles)

    async def __call__(self, request):
        body = await request.json()
        return await self._infer(body["smiles"])


async def benchmark(handle, n, concurrency):
    sem = asyncio.Semaphore(concurrency)
    lat = []

    async def one(s):
        async with sem:
            t = time.perf_counter()
            await handle.infer.remote(s)
            lat.append(time.perf_counter() - t)

    t0 = time.perf_counter()
    await asyncio.gather(*[one(SMILES_POOL[i % len(SMILES_POOL)]) for i in range(n)])
    dt = time.perf_counter() - t0
    lat.sort()
    return {
        "n": n, "concurrency": concurrency, "wall_s": dt, "mol_s": n / dt,
        "p50_ms": lat[len(lat) // 2] * 1e3,
        "p99_ms": lat[int(len(lat) * 0.99)] * 1e3,
    }


def main():
    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    app = KmolEnsemble.bind(CONFIG, CKPT_DIR)
    print(f"deploying {NUM_REPLICAS} replicas @ num_gpus={NUM_GPUS} (autoscaler brings up L4)...",
          flush=True)
    handle = serve.run(app, name="kmol")
    # warm the pipeline / let replicas come up
    for _ in range(20):
        handle.infer.remote("CCO").result()

    results = []
    for conc in [8, 32, 64, 128, 256]:
        r = asyncio.get_event_loop().run_until_complete(benchmark(handle, 4000, conc))
        results.append(r)
        print(f"conc={conc:>4}  {r['mol_s']:>10,.0f} mol/s  "
              f"p50={r['p50_ms']:.1f}ms p99={r['p99_ms']:.1f}ms", flush=True)

    peak = max(r["mol_s"] for r in results)
    out = {"num_replicas": NUM_REPLICAS, "num_gpus_per_replica": NUM_GPUS,
           "results": results, "peak_mol_s": peak, "x_baseline": peak / 170}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPEAK served: {peak:,.0f} mol/s = {peak/170:.1f}x the 170/s baseline")
    print(f"wrote {OUT}")
    serve.shutdown()


if __name__ == "__main__":
    main()
