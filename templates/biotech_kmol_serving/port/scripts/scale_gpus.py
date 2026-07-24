"""Multi-GPU linear-scaling test for the ported kMoL ensemble.

One placement-group bundle per L4 node ({CPU:7, GPU:1}, STRICT_SPREAD => distinct
nodes). Each bundle runs a self-contained mini-pipeline: 1 GPU forward actor + 6 CPU
featurizer actors, all pinned to that node — so a node's featurization feeds its own
GPU with no cross-node transfer. We then measure aggregate throughput using G=1..4
bundles and report speedup vs 1 GPU (ideal = G).
"""
import json
import os
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

_BUNDLE = os.environ.get("KMOL_BUNDLE", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIP_DIR = os.environ.get("KMOL_SHIP_DIR", _BUNDLE)
CONFIG = os.environ.get("KMOL_CONFIG", "configs/ensemble_serve.example.json")
CKPT_DIR = os.environ.get("KMOL_CKPT_DIR", "checkpoints")
OUT = os.environ.get("KMOL_OUT", "scale_results.json")
TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2"]}

MAX_G = 4
FEAT_PER_NODE = 6
CHUNK = 256
CHUNKS_PER_GPU = 40

SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]


@ray.remote(num_cpus=1, runtime_env=TASK_PIP)
class Featurizer:
    def __init__(self):
        import torch
        import kmolport
        from kmolport.featurizer import collate
        torch.set_num_threads(1)
        self.feat = kmolport.GraphFeaturizer()
        self.collate = collate

    def featurize(self, smiles_list):
        return self.collate([self.feat.featurize(s) for s in smiles_list])


@ray.remote(num_gpus=1, num_cpus=1, runtime_env=TASK_PIP)
class GpuForward:
    def __init__(self, config_path, ckpt_dir):
        import torch
        import kmolport
        torch.set_num_threads(1)
        self._torch = torch
        cfg = kmolport.load_config(config_path)
        self.model = kmolport.build_ensemble(cfg, checkpoint_dir=ckpt_dir, device="cuda")

    def forward(self, batch):
        batch = batch.to("cuda")
        with self._torch.no_grad():
            out = self.model({"graph": batch})
        return int(out["logits"].shape[0])

    def ping(self):
        return self._torch.cuda.get_device_name(0)


def pgss(pg, idx):
    return PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=idx)


def chunk():
    return [SMILES_POOL[j % len(SMILES_POOL)] for j in range(CHUNK)]


def main():
    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    pg = placement_group([{"CPU": 7, "GPU": 1}] * MAX_G, strategy="STRICT_SPREAD")
    print(f"requesting {MAX_G} L4 nodes (placement group)...", flush=True)
    ray.get(pg.ready())
    print("all bundles scheduled; building actors + installing runtime_env...", flush=True)

    gpus, feats_by_node = [], []
    for b in range(MAX_G):
        gpus.append(GpuForward.options(scheduling_strategy=pgss(pg, b)).remote(CONFIG, CKPT_DIR))
        feats_by_node.append(
            [Featurizer.options(scheduling_strategy=pgss(pg, b)).remote() for _ in range(FEAT_PER_NODE)])

    names = ray.get([g.ping.remote() for g in gpus])
    ray.get([f.featurize.remote(["CCO"] * 8) for fs in feats_by_node for f in fs])
    print(f"{MAX_G} nodes warm: {names}", flush=True)

    results = []
    for G in range(1, MAX_G + 1):
        n_chunks = CHUNKS_PER_GPU * G
        counters = [0] * G
        t0 = time.perf_counter()
        pending = []
        for i in range(n_chunks):
            b = i % G
            f = feats_by_node[b][counters[b] % FEAT_PER_NODE]
            counters[b] += 1
            batch_ref = f.featurize.remote(chunk())
            pending.append(gpus[b].forward.remote(batch_ref))
        ray.get(pending)
        dt = time.perf_counter() - t0
        mols = n_chunks * CHUNK
        results.append({"gpus": G, "mol_s": mols / dt, "mols": mols, "wall_s": dt})
        base = results[0]["mol_s"]
        print(f"G={G}  {mols/dt:>9,.0f} mol/s  {mols/dt/170:>5.1f}x baseline  "
              f"{mols/dt/base:>4.2f}x vs 1-GPU (ideal {G})", flush=True)

    base = results[0]["mol_s"]
    out = {
        "feat_per_node": FEAT_PER_NODE, "chunk": CHUNK, "gpu": names[0],
        "results": results,
        "scaling": [{"gpus": r["gpus"], "mol_s": r["mol_s"],
                     "speedup_vs_1gpu": r["mol_s"] / base, "ideal": r["gpus"],
                     "efficiency_pct": 100 * (r["mol_s"] / base) / r["gpus"]} for r in results],
        "peak_mol_s": results[-1]["mol_s"], "peak_x_baseline": results[-1]["mol_s"] / 170,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("\nSCALING:", "  ".join(f"{r['gpus']}GPU={r['mol_s']/base:.2f}x" for r in results), flush=True)
    print(f"peak {results[-1]['mol_s']:,.0f} mol/s on {MAX_G} L4 = {results[-1]['mol_s']/170:.0f}x baseline")
    print(f"wrote {OUT}")
    remove_placement_group(pg)


if __name__ == "__main__":
    main()
