"""Generate synthetic (randomly-initialized) checkpoints for a kMoL ensemble config.

Purpose: prove the SERVING MECHANICS (load-once, batching, warm-up, autoscaling,
throughput/scaling — the review's REC 1-7) without the real trained weights. The
architecture and checkpoint format are identical to real kMoL checkpoints, so the
real weights are a drop-in swap later; only prediction *values* change.

A kMoL checkpoint is `torch.load(path)["model"] == state_dict` (see
kmol AbstractNetwork.load_checkpoint). We build each sub-model from the config's
`model.model_configs[i]` and save its freshly-initialized state_dict.

Usage:
    python scripts/make_synthetic_checkpoints.py configs/ensemble_serve.example.json
"""

import json
import sys
from pathlib import Path

import torch

from kmol.core.helpers import SuperFactory
from kmol.model.architectures import AbstractNetwork


def main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())

    model = cfg["model"]
    if model.get("type") != "ensemble":
        raise SystemExit("Expected model.type == 'ensemble' with a 'model_configs' list.")

    sub_configs = model["model_configs"]
    ckpt_paths = cfg["checkpoint_path"]
    if len(sub_configs) != len(ckpt_paths):
        raise SystemExit(
            f"model_configs ({len(sub_configs)}) and checkpoint_path ({len(ckpt_paths)}) "
            "must be the same length."
        )

    for i, (sub_cfg, ckpt_path) in enumerate(zip(sub_configs, ckpt_paths)):
        # Fresh random init per sub-model so the ensemble has (trivial) diversity.
        torch.manual_seed(1000 + i)
        net = SuperFactory.create(AbstractNetwork, dict(sub_cfg))
        out = Path(ckpt_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict()}, out)
        n_params = sum(p.numel() for p in net.parameters())
        print(f"[{i}] wrote {out}  ({n_params:,} params)")

    print(f"Done: {len(ckpt_paths)} synthetic checkpoints written.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/make_synthetic_checkpoints.py <config.json>")
    main(sys.argv[1])
