"""P1b: two-stage pipeline — K CPU featurizer actors -> 1 GPU forward actor.

Shows how far ONE L4 goes once RDKit featurization (the P0/P1 bottleneck) is scaled
across cores. Featurizer actors (num_cpus=1) each turn a chunk of SMILES into a PyG
Batch; a single GPU actor (num_gpus=1) forwards them. Ray passes each featurized
Batch by object reference straight to the GPU actor. Pure Ray actors — no Serve/HTTP
overhead — so this isolates the pipeline ceiling for a single GPU.
"""
import json
import time

import ray

SHIP_DIR = "/home/ray/default/kmol_ship"
CONFIG = "config.json"
CKPT_DIR = "checkpoints"
OUT = "/home/ray/default/kmol_port/pipeline_results.json"
TASK_PIP = {"pip": ["torch==2.5.1", "torch_geometric==2.6.1", "rdkit==2024.3.5", "numpy<2"]}

K = 12          # CPU featurizer actors
CHUNK = 256
TOTAL = 60000

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


@ray.remote(num_gpus=1, runtime_env=TASK_PIP)
class GpuForward:
    def __init__(self, config_path, ckpt_dir):
        import torch

        import kmolport
        torch.set_num_threads(1)
        self._torch = torch
        cfg = kmolport.load_config(config_path)
        self.model = kmolport.build_ensemble(cfg, checkpoint_dir=ckpt_dir, device="cuda")

    def forward(self, batch):
        torch = self._torch
        batch = batch.to("cuda")
        with torch.no_grad():
            out = self.model({"graph": batch})
        return int(out["logits"].shape[0])  # keep the wire return tiny

    def ping(self):
        return self._torch.cuda.get_device_name(0)


def main():
    ray.init(address="auto", runtime_env={"working_dir": SHIP_DIR})
    feats = [Featurizer.remote() for _ in range(K)]
    gpu = GpuForward.remote(CONFIG, CKPT_DIR)
    gpu_name = ray.get(gpu.ping.remote())
    # warm up all actors
    ray.get([f.featurize.remote(["CCO"] * 8) for f in feats])
    ray.get(gpu.forward.remote(ray.get(feats[0].featurize.remote(["CCO"] * 8))))

    n_chunks = TOTAL // CHUNK
    print(f"gpu={gpu_name}  featurizers={K}  chunk={CHUNK}  total={TOTAL}", flush=True)

    t0 = time.perf_counter()
    pending = []
    for i in range(n_chunks):
        chunk = [SMILES_POOL[j % len(SMILES_POOL)] for j in range(CHUNK)]
        batch_ref = feats[i % K].featurize.remote(chunk)   # CPU featurize (parallel)
        pending.append(gpu.forward.remote(batch_ref))       # GPU forward (serial, by-ref)
    ray.get(pending)
    dt = time.perf_counter() - t0
    mol_s = TOTAL / dt

    # featurization-only aggregate rate (no GPU), same K actors
    tf = time.perf_counter()
    fpending = [feats[i % K].featurize.remote(
        [SMILES_POOL[j % len(SMILES_POOL)] for j in range(CHUNK)]) for i in range(n_chunks)]
    ray.get(fpending)
    feat_dt = time.perf_counter() - tf
    feat_mol_s = TOTAL / feat_dt

    out = {"gpu": gpu_name, "featurizer_actors": K, "chunk": CHUNK, "total_mols": TOTAL,
           "pipeline_wall_s": dt, "pipeline_mol_s": mol_s,
           "featurize_only_mol_s": feat_mol_s,
           "x_baseline": mol_s / 170}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nPIPELINE (K={K} CPU featurizers -> 1 L4): {mol_s:,.0f} mol/s "
          f"= {mol_s/170:.1f}x baseline", flush=True)
    print(f"featurize-only aggregate ({K} cores): {feat_mol_s:,.0f} mol/s", flush=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
