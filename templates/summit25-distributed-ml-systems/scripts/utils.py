"""Utility functions for distributed training with Ray.

These functions are imported by the distributed training notebook
and used by the DistributedWorker actor.
"""

import os
from datetime import timedelta

import ray
import torch
import torch.distributed as dist


def setup_torch_process_group_impl(
    rank: int,
    world_size: int,
    backend: str,
    master_addr: str,
    master_port: int,
    timeout_s: int = 1800,
):
    """Implementation of PyTorch process group initialization."""
    # Set environment variables for torch distributed
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    # For NCCL backend, set async error handling
    if backend == "nccl":
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        
        # Set CUDA device for this worker
        if torch.cuda.is_available():
            gpu_ids = ray.get_gpu_ids()
            if gpu_ids:
                torch.cuda.set_device(gpu_ids[0])
    
    # Initialize the process group
    dist.init_process_group(
        backend=backend,
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_s),
    )
    
    print(f"[Rank {rank}] Process group initialized successfully!")
    return True


def cleanup_impl(rank: int):
    """Implementation of process group cleanup."""
    if dist.is_initialized():
        print(f"[Rank {rank}] Destroying process group")
        dist.destroy_process_group()
    return True

