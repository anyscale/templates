"""Generate synthetic (randomly-initialized) checkpoints for the ported ensemble.

Port-native — needs only kmolport (torch/PyG), NOT the original kMoL. Produces the
exact architecture and file format as real kMoL checkpoints
(`torch.load(path)["model"] == state_dict`), so real trained weights are a drop-in
swap; only the prediction *values* differ. Use these to reproduce the THROUGHPUT
numbers without real weights (throughput is architecture-determined).

Usage (from the port/ bundle root, with kmolport importable):
    PYTHONPATH=. python scripts/make_synthetic_checkpoints.py \
        configs/ensemble_serve.example.json checkpoints
"""
import json
import sys
from pathlib import Path

import torch

from kmolport.abstract_network import AbstractNetwork
from kmolport.helpers import SuperFactory


def main(config_path: str, out_dir: str = "checkpoints") -> None:
    cfg = json.loads(Path(config_path).read_text())
    if cfg["model"].get("type") != "ensemble":
        raise SystemExit("Expected model.type == 'ensemble' with a 'model_configs' list.")

    sub_configs = cfg["model"]["model_configs"]
    ckpt_paths = cfg["checkpoint_path"]
    if len(sub_configs) != len(ckpt_paths):
        raise SystemExit("model_configs and checkpoint_path must be the same length.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, (sub_cfg, ckpt_path) in enumerate(zip(sub_configs, ckpt_paths)):
        torch.manual_seed(1000 + i)  # per-sub-model diversity
        net = SuperFactory.create(AbstractNetwork, dict(sub_cfg))
        dest = out / Path(ckpt_path).name
        torch.save({"model": net.state_dict()}, dest)
        n = sum(p.numel() for p in net.parameters())
        print(f"[{i}] wrote {dest}  ({n:,} params)")
    print(f"Done: {len(ckpt_paths)} synthetic checkpoints in {out}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: make_synthetic_checkpoints.py <config.json> [out_dir]")
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "checkpoints")
