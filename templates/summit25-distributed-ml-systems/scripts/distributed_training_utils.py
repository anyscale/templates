"""Utility functions for distributed training with Ray.

These functions are imported by the distributed training notebook
and used by the DistributedWorker actor.
"""

import os
from datetime import timedelta

import ray
import torch
from collections import defaultdict
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




### Advanced: GPU Setup and Coordination

# **The Problem:** When using NCCL (NVIDIA's collective communication library) for GPU training, each worker needs visibility to **all GPUs on its node**, not just its own GPU. By default, Ray isolates each actor to see only its assigned GPU.

# **Why This Matters:**
# - NCCL uses peer-to-peer GPU communication for efficiency
# - **Workers on the same node** can use **fast NVLink/PCIe** instead of going through the network
# - Without visibility to other GPUs, NCCL falls back to slower communication paths

# **The Solution:** We gather GPU information from all workers, group them by node, and set `CUDA_VISIBLE_DEVICES` to include all GPUs on each node.
# call share_cuda_visible_devices on the worker list before setup_torch_process_group_impl

def share_cuda_visible_devices(workers: list):
    """Share CUDA_VISIBLE_DEVICES across workers on the same node."""

    # Step 1: Collect metadata from all workers using execute()
    metadata_list = ray.get([
        worker.execute.remote(get_worker_metadata)
        for worker in workers
    ])
    
    # Step 2: Group workers by node
    node_to_workers = defaultdict(list)
    for worker_idx, (node_id, gpu_ids) in enumerate(metadata_list):
        node_to_workers[node_id].append(worker_idx)

    node_to_gpu_ids = defaultdict(set)
    for worker_idx, (node_id, gpu_ids) in enumerate(metadata_list):
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(str(gpu_id))
    
    # Step 3: Set CUDA_VISIBLE_DEVICES on each worker using execute()
    set_refs = []
    for node_id, worker_indices in node_to_workers.items():
        gpu_ids_str = ",".join(sorted(node_to_gpu_ids[node_id]))
        
        for worker_idx in worker_indices:
            set_ref = workers[worker_idx].execute.remote(
                set_worker_cuda_devices,
                rank=worker_idx,
                gpu_ids_str=gpu_ids_str
            )
            set_refs.append(set_ref)
    
    # Wait for all workers to complete setting CUDA_VISIBLE_DEVICES
    ray.get(set_refs)


def get_worker_metadata():
    """Get metadata about this worker (node_id and GPU IDs)."""
    node_id = ray.get_runtime_context().get_node_id()
    gpu_ids = ray.get_gpu_ids()
    return node_id, gpu_ids

def set_worker_cuda_devices(rank: int, gpu_ids_str: str):
    """Set CUDA_VISIBLE_DEVICES for this worker."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    print(f"[Rank {rank}] Set CUDA_VISIBLE_DEVICES={gpu_ids_str}")
    return True