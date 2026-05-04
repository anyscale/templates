# sequential_process.py
import time
import numpy as np



def process_image(image: np.ndarray) -> np.ndarray:
    """Simulates a slow 1-second filter."""
    time.sleep(1)
    return 255 - image

images = [np.random.randint(0, 255, (10, 10, 3)) for _ in range(8)]

start_time = time.time()

# Sequential: 8 images × 1 sec/image = 8 seconds
results = [process_image(img) for img in images]

end_time = time.time()

print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds.")
