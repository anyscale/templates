"""Minimal layers: GraphConvolutionWrapper + BatchNorm.

Verbatim from kMoL except: dropped the top-level `torch_scatter` import (only
GraphNorm and the edge-feature path used it; this config uses neither, since
edge_features defaults to 0). Attribute names are unchanged so checkpoint keys
match exactly: convolution / norm_layer / residual_layer / activation / dropout.
"""
from typing import Dict, Optional

import torch
import torch_geometric
from torch.nn.functional import leaky_relu

from .helpers import SuperFactory


class BatchNorm(torch_geometric.nn.BatchNorm):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().forward(x)


class GraphConvolutionWrapper(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
        layer_type: str = "torch_geometric.nn.GCNConv",
        is_residual: bool = True,
        norm_layer: Optional[str] = None,
        activation: str = "torch.nn.ReLU",
        edge_features: int = 0,
        propagate_edge_features: bool = False,
        **kwargs,
    ):
        super().__init__()
        base_features = in_features + in_features // 2 if edge_features else in_features
        self.convolution = SuperFactory.reflect(layer_type)(base_features, out_features, **kwargs)

        self._propagate_edge_features = propagate_edge_features
        self._edge_features = edge_features
        if self._edge_features and not self._propagate_edge_features:
            self.edge_projection = torch.nn.Linear(edge_features, in_features // 2)

        self.norm_layer = SuperFactory.reflect(norm_layer)(out_features) if norm_layer else None
        self.residual_layer = torch.nn.Linear(base_features, out_features) if is_residual else None
        self.activation = SuperFactory.reflect(activation)()
        self.dropout = torch.nn.Dropout(p=dropout)

    def _get_layer_arguments(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        arguments = {"x": x, "edge_index": edge_index}
        if self._propagate_edge_features:
            arguments["edge_attr"] = edge_attr
        return arguments

    def _add_edge_features(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if self._edge_features and not self._propagate_edge_features:
            from torch_scatter import scatter  # only reached when edge_features>0
            last_atom_index = x.size(0) - 1
            if last_atom_index not in torch.unique(edge_index[0]):
                edge_index = torch.cat(
                    (edge_index, torch.LongTensor([[last_atom_index], [last_atom_index]]).to(edge_index.device)),
                    dim=1,
                )
                edge_attr = torch.cat(
                    (edge_attr, torch.zeros((1, edge_attr.size(1))).to(edge_attr.device)), dim=0
                )
            per_node_edge_features = scatter(edge_attr, edge_index[0], dim=0, reduce="sum")
            per_node_edge_features = leaky_relu(self.edge_projection(per_node_edge_features))
            x = torch.cat([x, per_node_edge_features], dim=1)
        return x

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self._add_edge_features(x, edge_index, edge_attr)
        identity = x
        arguments = self._get_layer_arguments(x, edge_index, edge_attr)
        x = self.convolution(**arguments)
        if self.residual_layer:
            x += self.residual_layer(identity)
        if self.norm_layer:
            x = self.norm_layer(x, batch)
        x = self.activation(x)
        x = self.dropout(x)
        return x
