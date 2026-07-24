"""Minimal AbstractNetwork — the checkpoint loader + module base.

Trimmed from kMoL: dropped observers/EventManager/logger/exceptions that the
molecule inference path does not need. Checkpoint format and load_state_dict
semantics are preserved exactly (strict=False), which is why kMoL weights load
unchanged.
"""
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

import torch


class AbstractNetwork(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def get_requirements(self) -> List[str]:
        raise NotImplementedError

    def map(self, module: "AbstractNetwork", *args) -> Dict[str, Any]:
        requirements = module.get_requirements()
        if len(args) != len(requirements):
            raise AttributeError("Cannot map inputs to module")
        return {requirement: args[index] for index, requirement in enumerate(requirements)}

    def load_checkpoint(self, checkpoint_path: str, device: Optional[torch.device] = None,
                        strict: bool = False):
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is None")
        if device is None:
            device = torch.device("cpu")
        info = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = self.load_state_dict(info["model"], strict=False)
        if missing or unexpected:
            # A key mismatch means the checkpoint doesn't match this architecture — with
            # strict=False the model would be left partly random. Fail loudly in
            # production (strict=True); warn otherwise.
            msg = (f"checkpoint key mismatch for {checkpoint_path}: "
                   f"missing={missing} unexpected={unexpected}")
            if strict:
                raise RuntimeError("[kmolport] " + msg)
            print("[kmolport] WARNING " + msg)
        return info
