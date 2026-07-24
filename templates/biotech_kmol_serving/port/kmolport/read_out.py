"""Minimal read-outs: max / sum / mean + CombinedReadOut.

The ensemble config uses the GCN default read_out ("max", "sum"). Dropped the
attention / set2set / mlp_sum readouts (extra deps, unused here). These readouts
are parameter-free, so they never appear in the checkpoint state_dict.
"""
from typing import Dict, List, Tuple, Union

import torch
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool


class MaxReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_max_pool(x, batch)


class SumReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_add_pool(x, batch)


class MeanReadOut(torch.nn.Module):
    def __init__(self, in_channels: int, **kwargs):
        super().__init__()
        self.out_dim = in_channels

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return global_mean_pool(x, batch)


READOUT_FUNCTIONS = {"max": MaxReadOut, "sum": SumReadOut, "mean": MeanReadOut}


class CombinedReadOut(torch.nn.Module):
    def __init__(self, read_out_list: Union[Tuple[str, ...], List[str]], read_out_kwargs: dict):
        super().__init__()
        self.read_outs = torch.nn.ModuleList([get_read_out(f, read_out_kwargs) for f in read_out_list])
        self.out_dim = sum(read_out.out_dim for read_out in self.read_outs)

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        return torch.cat([read_out(x, batch) for read_out in self.read_outs], dim=1)


def get_read_out(read_out: Union[str, Tuple[str, ...], List[str]], read_out_kwargs: Dict):
    if "in_channels" not in read_out_kwargs:
        raise ValueError("Can't instantiate read_out without `in_channels` argument")
    if isinstance(read_out, (tuple, list)):
        return CombinedReadOut(read_out, read_out_kwargs)
    read_out_fn = READOUT_FUNCTIONS.get(read_out)
    if read_out_fn is None:
        raise ValueError(f"Unknown read_out function : {read_out}")
    return read_out_fn(**read_out_kwargs)
