"""This script inspects the memory usage of Ray tasks looking at both worker heap and shared memory."""

import ray
import numpy as np
import psutil


def print_memory_usage(name: str) -> None:
    process = psutil.Process()
    mem_info = process.memory_info()
    print(name)
    print(f"RSS : {mem_info.rss / 1024**2} MiB")
    print(f"Shared memory: {mem_info.shared / 1024**2} MiB")
    print(
        f"Heap memory (RSS - Shared): {(mem_info.rss - mem_info.shared) / 1024**2} MiB"
    )
    print("-" * 30, end="\n" * 2)


@ray.remote
def producer_task(size_mb: int = 4 * 1024) -> np.ndarray:
    print_memory_usage("producer_task: At start")
    array = np.random.rand((1024**2 * size_mb // 8)).astype(np.float64)
    print_memory_usage(f"producer_task: After creating array of size {size_mb // 1024:.2f} GiB")
    return array


@ray.remote
def consumer_task(array: np.ndarray) -> None:
    print_memory_usage("consumer_task: At start")
    assert isinstance(array, np.ndarray)
    assert not array.flags.owndata


if __name__ == "__main__":
    arr_ref = producer_task.remote()  # Produce a 4 GiB array.
    output_ref = consumer_task.remote(arr_ref)  # Consume the array.
    ray.wait([output_ref])  # Wait for the task to complete.
