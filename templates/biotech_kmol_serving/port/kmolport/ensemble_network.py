"""EnsembleNetwork — verbatim from kMoL minus the EventManager/observer hooks
(the molecule inference path doesn't need them). Holds 5 sub-models in one
ModuleList; forward returns torch.mean over them plus torch.var as logits_var.
"""
from typing import Any, Dict, List, Optional

import torch

from .abstract_network import AbstractNetwork
from .helpers import SuperFactory


class EnsembleNetwork(AbstractNetwork):
    def __init__(self, model_configs: List[Dict[str, Any]]):
        super().__init__()
        self.models = torch.nn.ModuleList(
            [SuperFactory.create(AbstractNetwork, config) for config in model_configs]
        )

    @property
    def out_features(self):
        return self.models[0].out_features

    def load_checkpoint(self, checkpoint_paths: List[str], device: Optional[torch.device] = None,
                        strict: bool = False):
        n_models = len(self.models)
        n_checkpoints = len(checkpoint_paths)
        if n_models != n_checkpoints:
            raise ValueError(
                f"Number of checkpoint_path should equal number of models. Got {n_models}, {n_checkpoints}."
            )
        for model, checkpoint_path in zip(self.models, checkpoint_paths):
            model.load_checkpoint(checkpoint_path, device, strict=strict)

    def get_requirements(self):
        return list(set(sum([model.get_requirements() for model in self.models], [])))

    def forward(self, data: Dict[str, Any], loss_type: str = None) -> Dict[str, torch.Tensor]:
        outs = [model.forward(data) for model in self.models]
        outputs = torch.stack(outs, dim=0)

        if loss_type == "torch.nn.BCEWithLogitsLoss":
            outputs = torch.sigmoid(outputs)

        return {
            "logits": torch.mean(outputs, dim=0),
            "logits_var": torch.var(outputs, dim=0),
        }
