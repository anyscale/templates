"""Assemble the ported ensemble from a kMoL config and run inference."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .ensemble_network import EnsembleNetwork
from .featurizer import GraphFeaturizer, collate


def load_config(config_path: str) -> dict:
    return json.loads(Path(config_path).read_text())


def build_ensemble(
    config: dict, checkpoint_dir: Optional[str] = None, device: str = "cpu",
    strict: bool = False,
) -> EnsembleNetwork:
    """Build the 5-model ensemble and load the checkpoints. Returns eval() model.

    strict=True raises on any checkpoint/architecture key mismatch (use in
    production so a wrong checkpoint can't silently load a half-random model).
    """
    model_configs = config["model"]["model_configs"]
    model = EnsembleNetwork(model_configs)

    ckpt_paths = config["checkpoint_path"]
    if checkpoint_dir is not None:
        ckpt_paths = [str(Path(checkpoint_dir) / Path(p).name) for p in ckpt_paths]
    model.load_checkpoint(ckpt_paths, device=torch.device(device), strict=strict)

    model = model.to(device).eval()
    return model


@torch.no_grad()
def predict(
    model: EnsembleNetwork, featurizer: GraphFeaturizer, smiles_list: List[str], device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Featurize -> one PyG Batch -> forward. Returns (logits, logits_var)."""
    data_list = [featurizer.featurize(s) for s in smiles_list]
    batch = collate(data_list).to(device)
    out: Dict[str, torch.Tensor] = model({"graph": batch})
    return out["logits"], out["logits_var"]
