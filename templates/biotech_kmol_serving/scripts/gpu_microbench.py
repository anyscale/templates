"""GPU throughput microbenchmark for the kMoL 5-model ensemble (no Ray).

Answers the email's REC 3 directly: the old "~5 ms/molecule" number was almost
certainly measured near batch size 1 (per-call overhead). This times the *batched*
forward of the whole ensemble at increasing batch sizes so we see the GPU's real
throughput and where it saturates.

Runs kMoL's own primitives: Predictor (loads all 5 checkpoints once) + the graph
featurizer + GeneralCollater (one PyG Batch). No Ray — this isolates model/GPU
throughput from any serving overhead.

Usage (with kMoL env python + PYTHONPATH set):
    python scripts/gpu_microbench.py configs/ensemble_serve.example.json
"""

import sys
import time

import numpy as np
import torch

from kmol.core.config import Config
from kmol.core.helpers import SuperFactory
from kmol.data.preprocessor import AbstractPreprocessor
from kmol.data.resources import AbstractCollater, DataPoint
from kmol.model.executors import Predictor

# A small pool of real drug-like SMILES, cycled to fill batches.
SMILES_POOL = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]
BATCH_SIZES = [1, 8, 16, 32, 64, 128, 256]
ITERS = 30      # timed iterations per batch size
WARMUP = 5


def build(config_path):
    cfg = Config.from_file(config_path, job_command="predict")
    pre = SuperFactory.create(AbstractPreprocessor, cfg.preprocessor, loaded_parameters={"config": cfg})
    col = SuperFactory.create(AbstractCollater, cfg.collater)
    pred = Predictor(config=cfg)
    labels = list(cfg.loader.get("target_column_names", []))
    ncol = cfg.loader["input_column_names"][0]
    return cfg, pre, col, pred, labels, ncol


def make_batch(pre, col, labels, ncol, n):
    pts = []
    for i in range(n):
        dp = DataPoint(id_=i, labels=labels, inputs={ncol: SMILES_POOL[i % len(SMILES_POOL)]},
                       outputs=np.zeros(max(len(labels), 1), dtype=np.float32))
        pre.preprocess(dp)
        pts.append(dp)
    return col.apply(pts)


def main(config_path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}  "
          f"gpu={torch.cuda.get_device_name(0) if dev=='cuda' else 'n/a'}")

    cfg, pre, col, pred, labels, ncol = build(config_path)
    print(f"ensemble models={len(pred.network.models)}  targets={len(labels)}")

    print(f"\n{'batch':>6} {'mol/s':>12} {'ms/mol':>10} {'ms/batch':>10}")
    for bs in BATCH_SIZES:
        batch = make_batch(pre, col, labels, ncol, bs)
        for _ in range(WARMUP):
            pred.run(batch)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            pred.run(batch)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / ITERS
        mol_s = bs / dt
        print(f"{bs:>6} {mol_s:>12,.0f} {dt/bs*1e3:>10.3f} {dt*1e3:>10.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/ensemble_serve.example.json")
