import ray
import time

@ray.remote
def process(x):
    """
    A simple Ray task that squares a number.
    Demonstrates basic distributed task execution with Ray.
    """
    print(f"Processing {x}")
    time.sleep(0.1)  # Simulate some work
    return x ** 2

# Run 100 tasks in parallel
results = ray.get([process.remote(i) for i in range(100)])
print(f"Processed {len(results)} numbers")
print(f"Sum of squares (0-99): {sum(results)}")
