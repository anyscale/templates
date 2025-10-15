#!/usr/bin/env python
"""
Simplified example of creating a worker group of Ray actors and setting up
a torch distributed process group, following the pattern used by Ray Train.
"""

import os
import socket
from collections import defaultdict
from typing import List, Tuple

import ray
import torch
import torch.distributed as dist
from datetime import timedelta


# ============================================================================
# Worker Actor Class
# ============================================================================


@ray.remote
class DistributedWorker:
    """A Ray actor that can participate in torch distributed operations."""

    def __init__(self, rank: int, world_size: int, device: str):
        """Initialize the distributed worker with rank, world_size, and device."""
        self.rank = rank
        self.world_size = world_size
        self.device = device

    def get_address_and_port(self) -> Tuple[str, int]:
        """Get the IP address and an available port for this worker."""
        ip_address = ray.util.get_node_ip_address()

        # Find an available port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            s.listen(1)
            port = s.getsockname()[1]

        return ip_address, port

    def get_metadata(self) -> Tuple[str, List[int]]:
        """Get metadata about this worker (node_id and GPU IDs)."""
        node_id = ray.get_runtime_context().get_node_id()
        gpu_ids = ray.get_gpu_ids()
        return node_id, gpu_ids

    def set_cuda_visible_devices(self, gpu_ids_str: str):
        """Set CUDA_VISIBLE_DEVICES to enable visibility of all GPUs on this node."""
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
        print(f"[Rank {self.rank}] Set CUDA_VISIBLE_DEVICES={gpu_ids_str}")

    def setup_torch_process_group(
        self,
        backend: str,
        master_addr: str,
        master_port: int,
        timeout_s: int = 1800,
    ):
        """Initialize torch distributed process group on this worker."""
        # Set environment variables for torch distributed
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)

        # For NCCL backend, set async error handling
        if backend == "nccl":
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

            # Set CUDA device for this worker
            if torch.cuda.is_available():
                gpu_ids = ray.get_gpu_ids()
                if gpu_ids:
                    # Use the first GPU assigned to this worker
                    torch.cuda.set_device(gpu_ids[0])
                    print(f"[Rank {self.rank}] Set CUDA device to GPU {gpu_ids[0]}")

        # Initialize the process group
        init_method = "env://"

        print(f"[Rank {self.rank}] Initializing process group with backend={backend}")

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=timeout_s),
        )

        print(f"[Rank {self.rank}] Process group initialized successfully!")

        return True

    def broadcast_tensor(self, src_rank: int = 0) -> torch.Tensor:
        """Participate in a broadcast operation from src_rank to all workers."""
        if not dist.is_initialized():
            raise RuntimeError("Process group not initialized!")

        # Create tensor
        if self.rank == src_rank:
            # Source rank creates a tensor with specific values
            tensor = torch.tensor(
                [100.0, 200.0, 300.0, 400.0, 500.0], device=self.device
            )
            print(f"[Rank {self.rank}] Broadcasting tensor: {tensor.tolist()}")
        else:
            # Other ranks create empty tensor to receive
            tensor = torch.zeros(5, device=self.device)
            print(f"[Rank {self.rank}] Before broadcast: {tensor.tolist()}")

        # Perform the broadcast
        dist.broadcast(tensor, src=src_rank)

        print(f"[Rank {self.rank}] After broadcast: {tensor.tolist()}")

        return tensor.cpu()

    def cleanup(self):
        """Clean up the distributed process group."""
        if dist.is_initialized():
            print(f"[Rank {self.rank}] Destroying process group")
            dist.destroy_process_group()
        return True


# ============================================================================
# Main Orchestration Logic
# ============================================================================


def create_worker_group(
    num_workers: int, resources_per_worker: dict | None = None, device: str = "cpu"
) -> List:
    """Create a group of worker actors with specified resources."""
    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 1}

    print()
    print("=" * 70)
    print(f"Creating {num_workers} workers...")
    print("=" * 70)
    print()

    # Create worker actors with their rank and world_size
    # Note: We pass rank and world_size at initialization time so each worker
    # knows its identity in the group. This is cleaner than passing it during
    # setup_torch_process_group and matches the pattern used in Ray Train.
    workers = []
    for rank in range(num_workers):
        worker = DistributedWorker.options(**resources_per_worker).remote(
            rank=rank,
            world_size=num_workers,
            device=device,
        )
        workers.append(worker)

    print(f"Created {len(workers)} worker actors")
    print()

    return workers


def share_cuda_visible_devices(workers: List):
    """Share CUDA_VISIBLE_DEVICES across workers on the same node.

    This enables NCCL to establish efficient peer-to-peer GPU communication
    between workers on the same node.
    """
    print("=" * 70)
    print("Sharing CUDA visible devices across workers")
    print("=" * 70)
    print()

    # Get metadata from all workers
    metadata_futures = [worker.get_metadata.remote() for worker in workers]
    metadata_list = ray.get(metadata_futures)

    # Build mapping of node_id -> worker indices and GPU IDs
    node_to_workers = defaultdict(list)
    node_to_gpu_ids = defaultdict(set)

    for worker_idx, (node_id, gpu_ids) in enumerate(metadata_list):
        node_to_workers[node_id].append(worker_idx)
        for gpu_id in gpu_ids:
            node_to_gpu_ids[node_id].add(str(gpu_id))

    # Set CUDA_VISIBLE_DEVICES on each worker to include all GPUs on its node
    set_refs = []
    for node_id, worker_indices in node_to_workers.items():
        # Sort GPU IDs for consistent ordering
        all_gpu_ids = ",".join(sorted(node_to_gpu_ids[node_id]))

        print(
            f"Node {node_id}: Setting CUDA_VISIBLE_DEVICES={all_gpu_ids} for workers {worker_indices}"
        )

        for worker_idx in worker_indices:
            future = workers[worker_idx].set_cuda_visible_devices.remote(all_gpu_ids)
            set_refs.append(future)

    ray.get(set_refs)


def setup_torch_distributed(
    workers: List,
    backend: str = "gloo",
    timeout_s: int = 1800,
):
    """Set up torch distributed process group across all workers."""
    world_size = len(workers)

    print("=" * 70)
    print("Setting up torch distributed process group")
    print(f"Backend: {backend}, World size: {world_size}")
    print("=" * 70)

    # Step 1: Get master address and port from rank 0 worker
    print("Step 1: Getting master address and port from rank 0...")
    master_addr, master_port = ray.get(workers[0].get_address_and_port.remote())
    print(f"Master address: {master_addr}:{master_port}")

    # Step 2: Initialize process group on each worker asynchronously
    print("Step 2: Initializing process group on all workers...")
    setup_futures = []
    for worker in workers:
        future = worker.setup_torch_process_group.remote(
            backend=backend,
            master_addr=master_addr,
            master_port=master_port,
            timeout_s=timeout_s,
        )
        setup_futures.append(future)

    # Step 3: Wait for all workers to complete initialization
    results = ray.get(setup_futures)
    print(f"All workers initialized: {all(results)}")


def run_distributed_broadcast(workers: List, src_rank: int = 0):
    """Run a broadcast operation across all workers."""
    print("=" * 70)
    print(f"Running distributed broadcast from rank {src_rank}")
    print("=" * 70)

    # Execute broadcast on all workers asynchronously
    broadcast_futures = [
        worker.broadcast_tensor.remote(src_rank=src_rank) for worker in workers
    ]

    # Wait for all workers to complete
    tensors = ray.get(broadcast_futures)

    print("=" * 70)
    print("Broadcast Results:")
    print("=" * 70)
    for rank, tensor in enumerate(tensors):
        print(f"Rank {rank}: {tensor.tolist()}")

    # Verify all tensors are the same
    all_same = all(torch.equal(tensors[0], t) for t in tensors)
    print(f"All tensors match: {all_same}")


def cleanup_workers(workers: List):
    """Clean up worker actors."""
    print("=" * 70)
    print("Cleaning up workers...")
    print("=" * 70)

    cleanup_refs = [worker.cleanup.remote() for worker in workers]
    ray.get(cleanup_refs)

    print("All workers cleaned up")


# ============================================================================
# Main Entry Point
# ============================================================================


def main(use_gpu: bool = True):
    """Main function demonstrating the worker group pattern."""

    # Initialize Ray
    ray.init(log_to_driver=True)

    print("=" * 70)
    print("SIMPLIFIED WORKER GROUP WITH TORCH DISTRIBUTED")
    print("=" * 70)

    # Configuration
    num_workers = 4

    if use_gpu:
        backend = "nccl"
        resources_per_worker = {"num_gpus": 1}
        device = "cuda"
        print("Using GPU mode with NCCL backend")
        print(f"Available GPUs: {torch.cuda.device_count()}")
    else:
        backend = "gloo"
        resources_per_worker = {"num_cpus": 1}
        device = "cpu"
        print("Using CPU mode with Gloo backend")

    print(f"Number of workers: {num_workers}")

    try:
        # Step 1: Create worker group
        workers = create_worker_group(
            num_workers=num_workers,
            resources_per_worker=resources_per_worker,
            device=device,
        )

        # Step 2: Share CUDA visible devices (for GPU mode with NCCL)
        if use_gpu:
            share_cuda_visible_devices(workers)

        # Step 3: Setup torch distributed process group
        setup_torch_distributed(workers, backend=backend)

        # Step 4: Run a distributed collective operation (broadcast)
        run_distributed_broadcast(workers, src_rank=0)

        # Step 5: Cleanup
        cleanup_workers(workers)

        print("=" * 70)
        print("SUCCESS! All operations completed successfully.")
        print("=" * 70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Shutdown Ray
        ray.shutdown()


if __name__ == "__main__":
    main()
