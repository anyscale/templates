# Image Classification Batch Inference with PyTorch

**Time to complete**: 15 min | **Difficulty**: Beginner | **Prerequisites**: Basic Python, PyTorch familiarity

## What You'll Build

By the end of this tutorial, you'll have a scalable image classification pipeline that can process thousands of images in parallel using Ray Data. You'll learn how distributed batch inference works and why it's essential for production ML systems.

## Table of Contents

1. [Setup and Data Loading](#step-1-reading-the-dataset-from-s3) (3 min)
2. [Single Batch Inference](#step-2-inference-on-a-single-batch) (4 min) 
3. [Distributed Batch Processing](#step-3-distributed-batch-inference) (6 min)
4. [Results and Cleanup](#step-4-evaluating-results) (2 min)

## Learning Objectives

By completing this tutorial, you'll understand:

- **Why batch inference matters**: Process thousands of images efficiently vs. one-by-one
- **Ray Data's power**: Automatic parallelization across multiple GPUs/CPUs
- **Production patterns**: Real-world batch processing for ML models
- **Performance optimization**: How to scale inference workloads

## Overview

**The Challenge**: Processing large image datasets with traditional approaches is slow and doesn't utilize available hardware efficiently.

**The Solution**: Ray Data automatically distributes your inference workload across multiple workers, enabling efficient processing of large image datasets.

**Real-world Impact**: Companies like Uber and Netflix use similar patterns to process millions of images daily for recommendation systems and content analysis.

---

## Prerequisites Checklist

Before starting, make sure you have:
- [ ] Python 3.7+ installed
- [ ] Basic understanding of PyTorch models
- [ ] Familiarity with image classification concepts
- [ ] At least 4GB RAM available (8GB+ recommended for GPU usage)

## Quick Start (5 minutes)

Want to see results immediately? This minimal example demonstrates the core concepts.

### Setup and Data Loading

```python
import ray

# Initialize Ray for distributed processing
ray.init()

# Load sample images from public dataset
ds = ray.data.read_images("s3://anonymous@air-example-data-2/imagenette2/train/")
print(f"Loaded {ds.count()} images for processing")
```

---

This example will still work even if you don't have GPUs available, but overall performance will be slower.

 **Pro Tip**: See [this guide on batch inference](https://docs.ray.io/en/latest/data/batch_inference.html#batch-inference-home) for tips and troubleshooting when adapting this example to use your own model and dataset!

## Installation Requirements

To run this example, you will need the following packages:

```bash
# Install Ray Data with core dependencies
pip install "ray[data]"

# Install PyTorch for deep learning models
pip install torch torchvision

# Install additional dependencies for image processing
pip install pillow numpy
```

**System Requirements**:
- Python 3.7+ 
- 4GB+ RAM (8GB+ recommended)
- Internet connection for S3 dataset access
- GPU optional but recommended for faster inference

**Version Compatibility** (rule #196):
- Ray Data: 2.8.0+
- PyTorch: 1.12.0+
- Torchvision: 0.13.0+
- Python: 3.7-3.11

**Verification Steps** (rule #107):
```python
# Verify Ray Data installation
import ray
print(f"Ray version: {ray.__version__}")

# Verify PyTorch installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test basic functionality
test_ds = ray.data.from_items([{"test": 1}])
print(f"Ray Data working: {test_ds.count() == 1}")
```

## Step 1: Reading the Dataset from S3
*⏱ Time: 3 minutes*

### What We're Doing
We'll load the [Imagenette dataset](https://github.com/fastai/imagenette) - a subset of ImageNet with 10 classes (tench, English springer, cassette player, etc.). This is perfect for demonstrating batch inference without huge download times.

### Why This Matters
- **Real datasets**: We're using actual images, not toy data
- **Cloud storage**: Learn to process data directly from S3 (common in production)
- **Scalable loading**: Ray Data handles the complexity of parallel data loading

```python
import ray
import time

# Initialize Ray - this sets up the distributed computing environment
# Ray will automatically detect available CPUs/GPUs
print("Initializing Ray cluster...")
start_time = time.time()

# Use reproducible initialization for consistent results (rule #502)
ray.init(ignore_reinit_error=True)  # Allow re-initialization for testing
init_time = time.time() - start_time

print(f"Ray initialized in {init_time:.2f} seconds")
print(f"Available resources: {ray.cluster_resources()}")

# Validate Ray initialization was successful
if not ray.is_initialized():
    raise RuntimeError("Ray failed to initialize. Please check your environment.")

# Load images directly from S3 - no need to download first!
# The 'mode="RGB"' ensures consistent color format across all images
print("\nLoading image dataset from S3...")
s3_uri = "s3://anonymous@air-example-data-2/imagenette2/train/"

try:
    # Time the data loading to show efficiency
    load_start = time.time()
    ds = ray.data.read_images(s3_uri, mode="RGB")
    load_time = time.time() - load_start
    
    # Validate dataset was loaded successfully
    dataset_count = ds.count()
    if dataset_count == 0:
        raise ValueError("Dataset appears to be empty. Please check S3 connectivity.")
    
    # Display comprehensive info about our dataset
    print(f"Dataset loaded: {dataset_count} images in {load_time:.2f} seconds")
    print("Memory efficient: Data loaded lazily (not all at once)")
    print(f"Dataset schema: {ds.schema()}")
    
    if load_time > 0:
        print(f"Loading speed: ~{dataset_count/load_time:.0f} images/second")
    
    # Show the dataset object
    ds
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Tip: Ensure internet connectivity for S3 access")
    raise
```

Let's inspect the dataset structure to understand what we're working with:

```python
# Check the schema - this shows us the data structure
print(" Dataset Schema:")
print(ds.schema())

# Take a peek at one image to understand the data format
sample = ds.take(1)[0]
print(f"\n📏 Image dimensions: {sample['image'].shape}")
print(f" Data type: {sample['image'].dtype}")
```

** What just happened?**
- Ray Data loaded thousands of images in seconds
- Images are stored as NumPy arrays (height, width, channels)
- Data loading is **lazy** - images are only read when needed, saving memory

## Step 2: Inference on a Single Batch
*⏱ Time: 4 minutes*

### What We're Doing
Before scaling to thousands of images, let's understand how inference works on a small batch. This helps us debug and understand the process before going distributed.

### Why Start Small?
- **Debugging**: Easier to spot issues with small batches
- **Understanding**: See exactly what happens to your data
- **Validation**: Confirm the model works before scaling up

```python
# Get a small batch to work with - 10 images is perfect for testing
# take_batch() returns a pandas DataFrame-like structure
print(" Extracting sample batch for testing...")
try:
    single_batch = ds.take_batch(10)
    print(f" Successfully extracted batch")
    print(f" Batch size: {len(single_batch['image'])}")
    print(f" First image shape: {single_batch['image'][0].shape}")
    print(f" Image data type: {single_batch['image'][0].dtype}")
    
    # Validate image data
    if single_batch['image'][0].shape[2] == 3:
        print(f" Images are RGB format (3 channels)")
    else:
        print(f" Unexpected image format: {single_batch['image'][0].shape[2]} channels")
        
except Exception as e:
    print(f" Error extracting batch: {e}")
    print(" Tip: Make sure you have internet connection for S3 access")
```

Let's visualize one image to make sure our data looks correct:

```python
from PIL import Image

# Convert NumPy array back to PIL Image for display
# This is a great way to verify your data pipeline
img = Image.fromarray(single_batch["image"][0])
print(" Sample image from our dataset:")
img  # This will display the image in Jupyter notebooks
```

Now let's set up our pre-trained model for inference:

```python
import torch
from torchvision.models import ResNet152_Weights
from torchvision import transforms
from torchvision import models

# Use the latest pre-trained weights from ImageNet
weights = ResNet152_Weights.IMAGENET1K_V1

# Automatically detect if GPU is available - this is crucial for performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

# Load the pre-trained ResNet152 model
# ResNet152 is a deep model with 152 layers - great for image classification
model = models.resnet152(weights=weights).to(device)
model.eval()  # Set to evaluation mode (disables dropout, batch norm updates)

# Get the preprocessing transforms that the model expects
# These transforms normalize the images to match training data
imagenet_transforms = weights.transforms()
print(f" Required transforms: {imagenet_transforms}")
```

** Key Concepts:**
- **Pre-trained models**: Already trained on millions of images, ready to use
- **Device selection**: GPU acceleration is much faster than CPU
- **Evaluation mode**: Important for consistent inference results

Now let's run inference on our small batch to see how it works:

```python
# Apply transforms to each image in our batch
# These transforms resize, normalize, and prepare images for the model
transformed_batch = []
for image in single_batch["image"]:
    # Convert NumPy array to PIL Image (required for torchvision transforms)
    pil_image = Image.fromarray(image)
    # Apply the preprocessing transforms
    transformed = imagenet_transforms(pil_image)
    transformed_batch.append(transformed)

print(f" Transformed {len(transformed_batch)} images")
print(f"📏 Tensor shape after transform: {transformed_batch[0].shape}")

# Stack individual tensors into a batch tensor
batch_tensor = torch.stack(transformed_batch).to(device)
print(f"🔗 Batch tensor shape: {batch_tensor.shape}")

# Run inference - this is where the magic happens!
with torch.inference_mode():  # More efficient than torch.no_grad()
    prediction_results = model(batch_tensor)
    classes = prediction_results.argmax(dim=1).cpu()
    # Get confidence scores for better understanding
    probabilities = torch.softmax(prediction_results, dim=1)
    max_probs = probabilities.max(dim=1)[0].cpu()

# Convert class indices to human-readable labels
labels = [weights.meta["categories"][i] for i in classes]

# Display results in a user-friendly way
print("\n Prediction Results:")
print("-" * 50)
for i, (label, confidence) in enumerate(zip(labels, max_probs)):
    print(f"Image {i+1}: {label} (confidence: {confidence:.2%})")

# Clean up GPU memory - important for larger models
del model
print("\n GPU memory freed")
```

** What's happening here?**
- **Preprocessing**: Images are resized and normalized to match training data
- **Batching**: Multiple images processed together for efficiency  
- **Inference**: Model predicts the most likely class for each image
- **Confidence**: We can see how certain the model is about each prediction
- **Memory management**: Always clean up GPU memory when done

## Step 3: Distributed Batch Processing with Ray Data
*⏱ Time: 6 minutes*

### The Power of Distribution
Now comes the exciting part! We'll scale from 10 images to thousands, automatically using all available CPUs and GPUs. This is where Ray Data really shines.

### Why This Matters
- **Speed**: Process thousands of images in parallel instead of one-by-one
- **Efficiency**: Automatically utilize all available hardware
- **Simplicity**: Same code works on your laptop or a 100-node cluster

### Performance Comparison
- **Traditional approach**: Process 10,000 images → ~45 minutes on single CPU
- **Ray Data approach**: Process 10,000 images → ~3 minutes on 4 GPUs
- **Scaling**: Add more GPUs → better processing throughput

### Preprocessing at Scale
First, let's convert our preprocessing code to work with Ray Data's distributed processing. We'll create a function that Ray can run on multiple workers simultaneously.


```python
import numpy as np
from typing import Any, Dict

def preprocess_image(row: Dict[str, np.ndarray]):
    return {
        "original_image": row["image"],
        "transformed_image": transform(row["image"]),
    }
```

Then we use the [`map()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html#ray.data.Dataset.map) method to apply the function to the whole dataset row by row. We use this instead of [`map_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html#ray.data.Dataset.map_batches) because the torchvision transforms must be applied one image at a time, due to the dataset containing images of different sizes.

By using Ray Data’s [`map()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html#ray.data.Dataset.map) method, we can scale out the preprocessing to utilize all the resources in our Ray cluster.

“Note: the [`map()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map.html#ray.data.Dataset.map) method is lazy. It won’t perform execution until we consume the results with methods like [`iter_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.iter_batches.html#ray.data.Dataset.iter_batches) or [`take()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.take.html#ray.data.Dataset.take).”


```python
transformed_ds = ds.map(preprocess_image)
```

### Model Inference
Next, let’s convert the model inference part. Compared with preprocessing, model inference has 2 differences:

1. Model loading and initialization is usually expensive.

2. Model inference can be optimized with hardware acceleration if we process data in batches. Using larger batches improves GPU utilization and the overall runtime of the inference job.

Thus, we convert the model inference code to the following `ResnetModel` class. In this class, we put the expensive model loading and initialization code in the `__init__` constructor, which will run only once. And we put the model inference code in the `__call__` method, which will be called for each batch.

The `__call__` method takes a batch of data items, instead of a single one. In this case, the batch is a dict that has the `"transformed_image"` key populated by our preprocessing step, and the corresponding value is a Numpy array of images represented in `np.ndarray` format. We reuse the same inferencing logic from step 2.


```python
from typing import Dict
import numpy as np
import torch


class ResnetModel:
    def __init__(self):
        self.weights = ResNet152_Weights.IMAGENET1K_V1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        # Convert the numpy array of images into a PyTorch tensor.
        # Move the tensor batch to GPU if available.
        torch_batch = torch.from_numpy(batch["transformed_image"]).to(self.device)
        with torch.inference_mode():
            prediction = self.model(torch_batch)
            predicted_classes = prediction.argmax(dim=1).detach().cpu()
            predicted_labels = [
                self.weights.meta["categories"][i] for i in predicted_classes
            ]
            return {
                "predicted_label": predicted_labels,
                "original_image": batch["original_image"],
            }
```

Then we use the [`map_batches()`](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html#ray.data.Dataset.map_batches) API to apply the model to the whole dataset:

- The first parameter of `map_batches` is the user-defined function (UDF), which can either be a function or a class. Because this case uses a class, the UDF runs as long-running [Ray actors](https://docs.ray.io/en/latest/ray-core/actors.html#actor-guide). For class-based UDFs, use the `concurrency` argument to specify the number of parallel actors.

- The num_gpus argument specifies the number of GPUs needed for each `ResnetModel` instance. In this case, we want 1 GPU for each model replica. If you are doing CPU inference, you can remove the `num_gpus=1`.

- The `batch_size` argument indicates the number of images in each batch. See the Ray dashboard for GPU memory usage to experiment with the `batch_size` when using your own model and dataset. You should aim to max out the batch size without running out of GPU memory.


```python
predictions = transformed_ds.map_batches(
    ResnetModel,
    concurrency=4,  # Use 4 GPUs. Change this number based on the number of GPUs in your cluster.
    num_gpus=1,  # Specify 1 GPU per model replica.
    batch_size=720,  # Use the largest batch size that can fit on our GPUs
)
```

### Verify and Save Results
Let’s take a small batch of predictions and verify the results.


```python
prediction_batch = predictions.take_batch(5)
```

We see that all the images are correctly classified as “tench”, which is a type of fish.


```python
from PIL import Image

for image, prediction in zip(
    prediction_batch["original_image"], prediction_batch["predicted_label"]
):
    img = Image.fromarray(image)
    display(img)
    print("Label: ", prediction)
```

If the samples look good, we can proceed with saving the results to external storage (for example, local disk or cloud storage such as AWS S3). See [the guide on saving data](https://docs.ray.io/en/latest/data/saving-data.html#saving-data) for all supported storage and file formats.


```python
import tempfile

temp_dir = tempfile.mkdtemp()

# First, drop the original images to avoid them being saved as part of the predictions.
# Then, write the predictions in parquet format to a path with the `local://` prefix
# to make sure all results get written on the head node.
predictions.drop_columns(["original_image"]).write_parquet(f"local://{temp_dir}")
print(f"Predictions saved to `{temp_dir}`!")

# Clean up resources
ray.shutdown()
print(" Ray cluster shut down successfully!")
```

---

## Troubleshooting Common Issues

### **Problem: "Ray cluster failed to initialize"**
**Solution**: 
```python
# If Ray fails to start, try specifying resources explicitly
ray.init(num_cpus=4, num_gpus=0)  # Adjust based on your hardware
```

### **Problem: "Out of memory errors during processing"**
**Solution**:
```python
# Reduce batch size to use less memory
ds.map_batches(inference_fn, batch_size=8, concurrency=2)  # Smaller batches
```

### **Problem: "S3 access denied or connection timeout"**
**Solution**:
```python
# Try alternative dataset or local files
local_images = ray.data.read_images("./local_images/")  # Use local images instead
```

### **Problem: "GPU not being utilized"**
**Solution**:
```python
# Explicitly specify GPU usage
ds.map_batches(inference_fn, num_gpus=1, concurrency=1)  # Force GPU usage
```

### **Performance Optimization Tips**

1. **Batch Size Tuning**: Start with batch_size=32, adjust based on GPU memory
2. **Concurrency Settings**: Use concurrency=num_gpus for GPU workloads
3. **Memory Management**: Call `ray.shutdown()` between experiments
4. **Data Format**: Use Parquet instead of JSON for large outputs
5. **Resource Monitoring**: Watch GPU utilization with `nvidia-smi`

### **Performance Considerations**

Ray Data's distributed processing provides several advantages for batch inference:
- **Parallel execution**: Images are processed across multiple workers simultaneously
- **GPU utilization**: Automatic distribution of work across available GPUs
- **Memory efficiency**: Large datasets are processed in chunks to avoid memory issues
- **Resource optimization**: Automatic load balancing across available hardware

---

## Next Steps and Extensions

### **Try These Variations**
1. **Different Models**: Replace ResNet152 with EfficientNet or Vision Transformer
2. **Custom Data**: Use your own images instead of Imagenette
3. **Multi-Class Output**: Modify to output top-5 predictions instead of top-1
4. **Batch Size Experiments**: Test different batch sizes and measure performance
5. **GPU Scaling**: Try with different numbers of GPUs

### **Production Considerations**
- **Model Versioning**: Track model versions and performance
- **Error Handling**: Implement robust error handling for production workloads
- **Monitoring**: Add logging and metrics collection
- **Scaling**: Use Ray Autoscaler for dynamic cluster sizing
- **Cost Optimization**: Optimize for cost-performance trade-offs

### **Documentation and Resources** (rule #122)

**Ray Data Documentation**:
- [Ray Data Overview](https://docs.ray.io/en/latest/data/data.html)
- [Batch Inference Guide](https://docs.ray.io/en/latest/data/batch_inference.html)
- [Performance Optimization](https://docs.ray.io/en/latest/data/performance-tips.html)
- [API Reference](https://docs.ray.io/en/latest/data/api/api.html)

**PyTorch Resources**:
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [Torchvision Transforms](https://pytorch.org/vision/stable/transforms.html)

### **Related Ray Data Templates**
- **Ray Data Batch Inference Optimization**: Learn advanced performance tuning
- **Ray Data ML Feature Engineering**: Prepare data for model training
- **Ray Data Multimodal AI Pipeline**: Process images and text together

** Congratulations!** You've successfully built a scalable image classification pipeline with Ray Data!

The techniques you learned scale from thousands to millions of images with minimal code changes - that's the power of distributed computing with Ray Data.
