"""
Baseline Implementation - Poorly Performing Ray Data Pipeline

This file demonstrates common performance mistakes in Ray Data pipelines:
1. Inefficient data ingestion without filtering
2. Poor transformation patterns (using .map() instead of .map_batches())
3. Resource misallocation and memory pressure
4. Inefficient output handling

Run this to see poor performance, then compare with optimized_implementation.py
"""

import os
import time
import tempfile
from typing import Dict, Any
import numpy as np
import ray
import torch
import torchvision
from torchvision import transforms, models
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - these are intentionally suboptimal
BATCH_SIZE = 32  # Too small for GPU efficiency
BATCH_SIZE_CPU = 16  # Even smaller for CPU
NUM_WORKERS = 2  # Too few workers
MODEL_NAME = "resnet50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Real ImageNet dataset from S3
IMAGENET_S3_URI = "s3://anonymous@air-example-data-2/imagenette2/train/"

def create_imagenet_dataset():
    """Create dataset from actual ImageNet data on S3"""
    logger.info(f"📊 Loading ImageNet dataset from: {IMAGENET_S3_URI}")
    
    try:
        # MISTAKE 1: No streaming, no block size optimization
        ds = ray.data.read_images(
            IMAGENET_S3_URI,
            mode="RGB",
            # MISTAKE 2: No override_num_blocks - poor parallelism
            # MISTAKE 3: No streaming - loads entire dataset into memory
        )
        
        logger.info(f"✅ Dataset loaded successfully")
        logger.info(f"📊 Dataset schema: {ds.schema()}")
        
        return ds
        
    except Exception as e:
        logger.error(f"Failed to load ImageNet dataset: {e}")
        logger.info("Falling back to sample dataset...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create a sample dataset with repeated images to simulate larger dataset"""
    logger.info("📊 Creating sample dataset (fallback)")
    
    # Sample ImageNet data paths (replace with your actual data)
    SAMPLE_IMAGES = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/Eopsaltria_australis_-_Mogo_Campground.jpg/1200px-Eopsaltria_australis_-_Mogo_Campground.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Common_dog_breeds_around_the_world.jpg/1200px-Common_dog_breeds_around_the_world.jpg",
    ]
    
    def download_sample_images():
        """Download sample images for demonstration"""
        import requests
        from io import BytesIO
        
        images = []
        for url in SAMPLE_IMAGES:
            try:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content)).convert('RGB')
                # Resize to standard ImageNet size
                img = img.resize((224, 224))
                images.append(np.array(img))
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                # Create a dummy image if download fails
                dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                images.append(dummy_img)
        
        return images
    
    images = download_sample_images()
    # Repeat images to create a larger dataset for demonstration
    repeated_images = images * 100  # 300 total images
    
    # Create dataset with image paths and labels
    data = []
    for i, img in enumerate(repeated_images):
        data.append({
            "image_id": f"img_{i:06d}",
            "image": img,
            "file_path": f"sample_image_{i % len(images)}.jpg"
        })
    
    return ray.data.from_items(data)

class PoorlyOptimizedResNet:
    """
    This class demonstrates several anti-patterns:
    1. Model loaded for every batch (inefficient)
    2. No proper error handling
    3. Inefficient preprocessing
    4. Poor memory management
    5. No device-specific optimization
    """
    
    def __init__(self):
        # MISTAKE 1: Loading model in __init__ but not caching it properly
        self.device = DEVICE
        # MISTAKE 2: Loading full ResNet50 without optimization
        self.model = models.resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        # MISTAKE 3: Inefficient transforms - recreating for each call
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # MISTAKE 4: No batch size optimization for device
        if self.device == "cuda":
            self.batch_size = BATCH_SIZE
        else:
            self.batch_size = BATCH_SIZE_CPU
        
        # MISTAKE 5: No proper cleanup
        self._load_imagenet_labels()
        
        logger.info(f"PoorlyOptimizedResNet initialized on {self.device}")
    
    def _load_imagenet_labels(self):
        """Load ImageNet class labels"""
        try:
            # This is inefficient - should be cached
            import urllib.request
            url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
            urllib.request.urlretrieve(url, "imagenet_classes.txt")
            with open("imagenet_classes.txt", "r") as f:
                self.labels = [line.strip() for line in f.readlines()]
        except:
            # Fallback to dummy labels
            self.labels = [f"class_{i}" for i in range(1000)]
    
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        MISTAKE 6: Inefficient batch processing
        - No proper error handling
        - Inefficient tensor conversion
        - No memory management
        - Processing one image at a time
        """
        try:
            # MISTAKE 7: Processing one image at a time instead of batching
            results = []
            for i in range(len(batch["image"])):
                # MISTAKE 8: Converting numpy to PIL to tensor (inefficient)
                img = Image.fromarray(batch["image"][i])
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                
                # MISTAKE 9: No batch inference - processing one at a time
                with torch.no_grad():
                    output = self.model(img_tensor)
                    predicted_class = output.argmax(dim=1).item()
                    confidence = torch.softmax(output, dim=1).max().item()
                
                # MISTAKE 10: Inefficient result construction
                image_id = batch.get("image_id", [f"img_{j}" for j in range(len(batch["image"]))])[i]
                file_path = batch.get("file_path", [f"image_{j}.jpg" for j in range(len(batch["image"]))])[i]
                
                results.append({
                    "image_id": image_id,
                    "predicted_class": predicted_class,
                    "predicted_label": self.labels[predicted_class],
                    "confidence": confidence,
                    "file_path": file_path
                })
            
            # MISTAKE 11: Inefficient result aggregation
            return {
                "image_id": [r["image_id"] for r in results],
                "predicted_class": [r["predicted_class"] for r in results],
                "predicted_label": [r["predicted_label"] for r in results],
                "confidence": [r["confidence"] for r in results],
                "file_path": [r["file_path"] for r in results]
            }
            
        except Exception as e:
            # MISTAKE 12: Poor error handling - just logging and continuing
            logger.error(f"Error processing batch: {e}")
            # Return empty results which will cause downstream issues
            batch_size = len(batch["image"])
            return {
                "image_id": [f"img_{i}" for i in range(batch_size)],
                "predicted_class": [],
                "predicted_label": [],
                "confidence": [],
                "file_path": [f"image_{i}.jpg" for i in range(batch_size)]
            }

def get_baseline_configuration():
    """Get baseline configuration - intentionally suboptimal"""
    if not ray.is_initialized():
        return {
            "num_workers": 2,
            "batch_size": BATCH_SIZE_CPU,
            "use_gpu": False,
            "device": "cpu"
        }
    
    cluster_resources = ray.cluster_resources()
    available_gpus = cluster_resources.get('GPU', 0)
    available_cpus = cluster_resources.get('CPU', 4)
    
    # MISTAKE 13: Always use suboptimal configuration regardless of available resources
    if available_gpus > 0:
        # MISTAKE 14: Use too few workers even with GPUs available
        num_workers = 2  # Should be min(available_gpus * 2, available_cpus)
        batch_size = BATCH_SIZE
        use_gpu = True
        device = "cuda"
        logger.info(f"🎮 GPU cluster detected: {available_gpus} GPUs, {available_cpus} CPUs")
        logger.warning("⚠️  Using suboptimal configuration: only 2 workers")
    else:
        # MISTAKE 15: Use too few workers for CPU-only cluster
        num_workers = 2  # Should be max(4, available_cpus // 2)
        batch_size = BATCH_SIZE_CPU
        use_gpu = False
        device = "cpu"
        logger.info(f"💻 CPU-only cluster detected: {available_cpus} CPUs")
        logger.warning("⚠️  Using suboptimal configuration: only 2 workers")
    
    return {
        "num_workers": num_workers,
        "batch_size": batch_size,
        "use_gpu": use_gpu,
        "device": device
    }

def run_baseline_pipeline():
    """Run the poorly optimized baseline pipeline"""
    logger.info("🚀 Starting Baseline Pipeline (Poorly Optimized)")
    logger.info("⚠️  This pipeline demonstrates common performance mistakes")
    
    start_time = time.time()
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Get baseline configuration (intentionally suboptimal)
    config = get_baseline_configuration()
    logger.info(f"🔧 Baseline Configuration: {config['num_workers']} workers, batch_size={config['batch_size']}, device={config['device']}")
    
    # Create dataset
    logger.info("📊 Creating ImageNet dataset...")
    ds = create_imagenet_dataset()
    logger.info(f"Dataset created with {ds.count()} images")
    
    # MISTAKE 16: No proper block size configuration
    logger.info("🔄 Starting inference pipeline...")
    
    # MISTAKE 17: Using .map() instead of .map_batches() for batch operations
    # MISTAKE 18: Too few workers
    # MISTAKE 19: No GPU allocation specification
    # MISTAKE 20: No proper batch size optimization
    predictions = ds.map(
        PoorlyOptimizedResNet,
        concurrency=config["num_workers"],  # Too few workers
        # MISTAKE 21: No num_gpus specification even when GPUs are available
        # MISTAKE 22: No proper batch size optimization
    )
    
    # MISTAKE 23: No progress monitoring
    logger.info("⏳ Processing predictions...")
    
    # MISTAKE 24: Materializing entire dataset at once (memory pressure)
    results = predictions.take_all()
    
    # MISTAKE 25: Inefficient output handling
    logger.info("💾 Saving results...")
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "baseline_results.parquet")
    
    # MISTAKE 26: No repartitioning before writing (will create many small files)
    predictions.write_parquet(output_path)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info(f"✅ Baseline pipeline completed in {total_time:.2f} seconds")
    logger.info(f"📁 Results saved to: {output_path}")
    logger.info(f"📊 Processed {len(results)} images")
    logger.info(f"🐌 Throughput: {len(results)/total_time:.2f} images/second")
    
    return {
        "total_time": total_time,
        "images_processed": len(results),
        "throughput": len(results)/total_time,
        "output_path": output_path,
        "configuration": config
    }

def analyze_baseline_performance():
    """Analyze the performance of the baseline pipeline"""
    logger.info("🔍 Analyzing baseline performance...")
    
    # Run the pipeline
    results = run_baseline_pipeline()
    
    # Performance analysis
    logger.info("\n" + "="*60)
    logger.info("📊 BASELINE PERFORMANCE ANALYSIS")
    logger.info("="*60)
    logger.info(f"Total Time: {results['total_time']:.2f} seconds")
    logger.info(f"Images Processed: {results['images_processed']}")
    logger.info(f"Throughput: {results['throughput']:.2f} images/second")
    logger.info(f"Average Time per Image: {results['total_time']/results['images_processed']*1000:.2f} ms")
    
    # Show configuration used
    config = results['configuration']
    logger.info(f"\n🔧 Configuration Used:")
    logger.info(f"• Device: {config['device']}")
    logger.info(f"• Workers: {config['num_workers']}")
    logger.info(f"• Batch Size: {config['batch_size']}")
    logger.info(f"• GPU Enabled: {config['use_gpu']}")
    
    # Identify performance issues
    logger.info("\n🚨 PERFORMANCE ISSUES IDENTIFIED:")
    logger.info("1. Small batch size - poor device utilization")
    logger.info("2. Too few workers - limited parallelism")
    logger.info("3. Using .map() instead of .map_batches() - no batching benefits")
    logger.info("4. No GPU allocation specification - may not use GPUs")
    logger.info("5. Inefficient preprocessing - converting numpy→PIL→tensor")
    logger.info("6. No proper error handling - pipeline may fail silently")
    logger.info("7. No output repartitioning - many small output files")
    logger.info("8. Poor memory management - materializing entire dataset")
    logger.info("9. No streaming - loads entire dataset into memory")
    logger.info("10. No block size optimization - poor parallelism")
    logger.info("11. Processing one image at a time - no vectorization")
    logger.info("12. Inefficient result construction - multiple loops")
    logger.info("13. No device-specific optimizations")
    logger.info("14. Suboptimal resource allocation")
    logger.info("15. No progress monitoring or error recovery")
    
    logger.info("\n💡 NEXT STEPS:")
    logger.info("Run optimized_implementation.py to see dramatic improvements!")
    
    return results

if __name__ == "__main__":
    try:
        analyze_baseline_performance()
    except Exception as e:
        logger.error(f"Baseline pipeline failed: {e}")
        raise
