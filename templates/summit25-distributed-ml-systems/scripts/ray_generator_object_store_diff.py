"""Inspect diff in distributed object store usage between a Ray generator task and a non-generator task."""
import time
import ray
import ray.util.state
import numpy as np
import gc


def list_objects():
    """List and print Ray objects with their sizes in MiB."""
    for obj_state in ray.util.state.list_objects(
        filters=[("TASK_STATUS", "!=", "NIL")],
        raise_on_missing_output=False
    ):
        if obj_state.object_size > 1000:
            print(
                f"{obj_state.object_id.replace('f', '')}: {obj_state.object_size / 1024**2:.2f} MiB"
                f", REF TYPE: {obj_state.reference_type}"
            )
    print("-" * 100)


@ray.remote(num_returns="streaming")
def generate_blocks(num_blocks=10, block_size_mb=100):
    """Generate data blocks and yield them one by one."""
    for i in range(num_blocks):
        print(f"Generating block {i}")
        time.sleep(0.5)  # Simulate network latency
        yield {"a": np.random.rand(1024**2 * block_size_mb // 8), "id": i}


@ray.remote
def transform_block(block):
    """Transform data blocks by incrementing their values."""
    print(f"Transformed block {block['id']}")
    list_objects()  # Monitor object store usage
    return {"a": block["a"] + 1, "id": block["id"]}


@ray.remote
def build_all_data(num_blocks=10, block_size_mb=100):
    """Build data blocks and return them as a list."""
    time.sleep(0.5 * num_blocks)  # Simulate network latency
    return [
        {"a": np.random.rand(1024**2 * block_size_mb // 8), "id": i}
        for i in range(num_blocks)
    ]


@ray.remote
def transform_all_data(data):
    """Transform data blocks by incrementing their values."""
    print("Transformed all data")
    list_objects()  # Monitor object store usage
    return [{"a": block["a"] + 1, "id": block["id"]} for block in data]


if __name__ == "__main__":
    ray.init()

    # Generate, build, and transform data blocks
    print("\nInspecting the impact of a Ray generator task on the object store")
    for obj_ref in generate_blocks.remote():
        transform_block.remote(obj_ref)
    
    # clean up last reference
    del obj_ref
    gc.collect()

    print("\nInspecting the impact of a Ray non-generator task on the object store")
    ray.get(transform_all_data.remote(build_all_data.remote()))
