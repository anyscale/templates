# parallel_process.py
import ray
import time
import numpy as np

@ray.remote
def process_image(image: np.ndarray) -> np.ndarray:
    """Simulates a slow 1-second filter."""
    time.sleep(1)
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]

start_time = time.time()

# 2. Launch tasks in parallel; returns list of ObjectRefs (futures)
result_refs = [process_image.remote(img) for img in images]

# 3. Wait for and retrieve finished results via ray.get()
results = ray.get(result_refs)
end_time = time.time()

# On an 8-core machine: ~1 second total runtime!
print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds.")

ray.shutdown()
