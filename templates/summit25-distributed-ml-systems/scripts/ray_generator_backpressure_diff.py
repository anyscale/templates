"""This script demonstrates the difference in execution between a backpressured generator and a non-backpressured generator."""

from typing import Dict, Iterator
import numpy as np
import ray
import time

def generate_data(
    num_blocks: int, block_size_mb: int, time_per_block: int
) -> Iterator[Dict[str, np.ndarray]]:
    """Generates data blocks with a delay."""
    for _ in range(num_blocks):
        time.sleep(time_per_block)
        yield {"a": np.random.rand(1024**2 * block_size_mb // 8)}

def run_generator(num_blocks, block_size_mb, time_per_block, backpressure=None):
    """Runs the generator with optional backpressure."""
    # Define the remote function
    if backpressure:
        print(f"With backpressure of {backpressure} objects:")
        generator_remote_fn = ray.remote(
            _generator_backpressure_num_objects=backpressure
        )(generate_data)
    else:
        print("Without backpressure:")
        generator_remote_fn = ray.remote(generate_data)

    # Submit the generator task
    generator_ref = generator_remote_fn.remote(
        num_blocks, block_size_mb, time_per_block
    )

    # Wait for the total block generation time
    print(f"Waiting for {time_per_block * num_blocks + 10} seconds")
    time.sleep(time_per_block * num_blocks + 10)

    # Iterate over the generator
    previous_time = time.time()
    for idx, obj_ref in enumerate(generator_ref):
        block = ray.get(obj_ref)
        current_time = time.time()
        assert block["a"].shape == (1024**2 * block_size_mb // 8,)
        print(f"Fetched block {idx} with delay of {current_time - previous_time:.2f} seconds")
        previous_time = current_time

    print("Done", end="\n\n")

if __name__ == "__main__":
    num_blocks = 10
    time_per_block = 2
    block_size_mb = 100

    ray.init()

    run_generator(num_blocks, block_size_mb, time_per_block)
    run_generator(num_blocks, block_size_mb, time_per_block, backpressure=2)
