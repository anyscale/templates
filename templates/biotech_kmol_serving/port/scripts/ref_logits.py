"""Ground-truth reference logits from the REAL kMoL (py3.9 / torch 1.13 / PyG 2.3).

Runs kMoL's own Config + preprocessor + collater + Predictor on a fixed SMILES set
and dumps logits + variance to JSON. The py3.11 port is validated against this.

Usage (in the `kmol` conda env, cwd = kmol_serving):
    python ref_logits.py <your-config>.json ref_logits.json
"""
import json
import sys

import numpy as np
import torch

from kmol.core.config import Config
from kmol.core.helpers import SuperFactory
from kmol.data.preprocessor import AbstractPreprocessor
from kmol.data.resources import AbstractCollater, DataPoint
from kmol.model.executors import Predictor

# Fixed, drug-like SMILES set (same pool the microbench cycles).
SMILES = [
    "CCO", "c1ccccc1", "CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1CCC[C@H]1c1cccnc1", "Oc1ccc2CC3C(Cc2c1)C1CCCCC1CC3",
    "Clc1ccccc1C(=O)Nc1ccccc1", "COc1ccc2nc(sc2c1)N", "CC(=O)Nc1ccc(O)cc1",
]


def main(cfg_path: str, out_path: str) -> None:
    cfg = Config.from_file(cfg_path, job_command="predict")
    pre = SuperFactory.create(
        AbstractPreprocessor, cfg.preprocessor, loaded_parameters={"config": cfg}
    )
    col = SuperFactory.create(AbstractCollater, cfg.collater)
    pred = Predictor(config=cfg)

    labels = list(cfg.loader.get("target_column_names", []))
    ncol = cfg.loader["input_column_names"][0]
    n = max(len(labels), 1)

    pts = []
    for i, s in enumerate(SMILES):
        dp = DataPoint(id_=i, labels=labels, inputs={ncol: s},
                       outputs=np.zeros(n, dtype=np.float32))
        pre.preprocess(dp)
        pts.append(dp)

    batch = col.apply(pts)
    payload = pred.run(batch)

    logits = payload.logits.detach().cpu().numpy()
    var = getattr(payload, "logits_var", None)
    var = var.detach().cpu().numpy().tolist() if var is not None else None

    out = {
        "smiles": SMILES,
        "labels": labels,
        "logits": logits.tolist(),
        "variance": var,
        "meta": {
            "torch": torch.__version__,
            "pyg": __import__("torch_geometric").__version__,
            "n_models": len(pred.network.models),
            "device": str(cfg.get_device()),
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}  logits shape={logits.shape}  device={out['meta']['device']}")
    print("logits[0][:5]=", logits[0][:5])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
