"""Minimal SuperFactory: the two behaviors kMoL's molecule path relies on.

- reflect(dotted_path): resolve a string like "torch_geometric.nn.LEConv" or
  "torch.nn.ReLU" to the object. The one kMoL-internal string used by the
  molecule GCN config is "kmol.model.layers.BatchNorm" -> our BatchNorm.
- create(base, config): pop "type" and instantiate. The only submodel type in
  the ensemble config is "graph_convolutional".
"""
import importlib
from typing import Any, Dict, Optional


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class SuperFactory:
    @staticmethod
    def reflect(path: str):
        if path == "kmol.model.layers.BatchNorm":
            from .layers import BatchNorm
            return BatchNorm
        module_path, _, name = path.rpartition(".")
        return getattr(importlib.import_module(module_path), name)

    @staticmethod
    def create(base, config: Dict[str, Any], loaded_parameters: Optional[Dict[str, Any]] = None):
        config = dict(config)
        type_ = config.pop("type")
        if type_ == "graph_convolutional":
            from .graph_convolutional_network import GraphConvolutionalNetwork
            cls = GraphConvolutionalNetwork
        else:
            # Fall back to dotted-path resolution (matches kMoL's behavior for
            # fully-qualified type strings).
            cls = SuperFactory.reflect(type_)
        params = dict(config)
        if loaded_parameters:
            params.update(loaded_parameters)
        return cls(**params)
