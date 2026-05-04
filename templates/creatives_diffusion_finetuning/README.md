# Stable Diffusion LoRA Fine-Tuning with Ray Train

**⏱️ Time to complete**: 30 min

### Anyscale Technical Demo — Distributed Training on Anyscale Jobs

---

## The Problem

A creative AI company fine-tunes Stable Diffusion on their custom art style.
On a single GPU this takes **12+ hours**. Multi-GPU requires rewriting with
`torch.distributed`, managing NCCL, writing custom launch scripts. If a spot
instance is preempted at epoch 8 — they lose all progress.

## What We're Building

```
Ray Data (CPU)                     Ray Train (2x T4 GPU)
┌──────────────────┐               ┌──────────────────┐
│ 833 Pokemon images│──stream──────▶│ LoRA fine-tuning  │
│ decode + resize   │               │ DDP across 2 GPUs │
│ normalize + tokenize│             │ 0.1% params trained│
└──────────────────┘               │ checkpoint each epoch│
                                    └──────────────────┘
```

Standard PyTorch + diffusers + peft code. Ray Train distributes it.
Ray Data feeds it. Anyscale runs it as a Job.

## Step 1: Connect to the Ray Cluster

The cell below initializes a Ray connection and inspects the cluster resources provisioned by Anyscale.
This is the **zero-infrastructure** story: Anyscale spins up a heterogeneous cluster (CPU heads + GPU workers) from a simple YAML config — no Kubernetes manifests, no NCCL tuning, no SSH.

**Why it matters:** Your ML engineers write Python, not infrastructure-as-code. Time-to-first-experiment drops from days to minutes.


```python
import os
os.environ["HF_HOME"] = "/mnt/cluster_storage/hf_cache"

!pip install -q -r requirements.txt
```


```python
import sys, os

DEMO_ROOT = os.path.abspath(os.getcwd())
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

import ray

ray.init(
    ignore_reinit_error=True,
    runtime_env={"working_dir": DEMO_ROOT},
)

resources = ray.cluster_resources()
print("Ray cluster resources:")
for resource, count in sorted(resources.items()):
    if not resource.startswith('node:'):
        print(f"  {resource:<20} {count}")

gpus = resources.get('GPU', 0)
print(f"\nGPU workers ready: {int(gpus)} T4 GPUs")
```

## Step 2: Load and Explore the Training Data with Ray Data

Here we use **Ray Data** to load the Pokemon image dataset as a distributed, streaming dataset.
Unlike PyTorch DataLoader, Ray Data decouples data preprocessing from GPU training — CPUs handle decode/resize while GPUs stay saturated with training. No OOM from loading the full dataset into memory.

**Key differentiator:** Ray Data streams batches on-demand, enabling training on datasets far larger than GPU memory — critical for production creative AI workloads.


```python
from src.data_pipeline import load_pokemon_dataset
import io

ds = load_pokemon_dataset()
print(f"Dataset: {ds.count()} images")
print(f"Schema:  {ds.schema()}")

# Show sample images with captions
samples = ds.take(4)
from src.utils import show_image_grid
from PIL import Image
import numpy as np

images = []
titles = []
for s in samples:
    img = s['image']
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img['bytes']))
    images.append(img)
    titles.append(s['en_text'][:60])

show_image_grid(images, titles, cols=4)
```

## Code Walkthrough

Open `src/train_lora.py` — the key pieces:

**LoRA config** — only 0.1% of UNet parameters are trainable:
```python
lora_config = LoraConfig(r=4, target_modules=["to_q", "to_v", "to_k", "to_out.0"])
unet = get_peft_model(unet, lora_config)
```

**Standard diffusion training loop** — encode latents, sample noise, predict, MSE loss:
```python
latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
loss = F.mse_loss(noise_pred.float(), noise.float())  # fp32 for stability on T4
```

**Ray Train integration** — three additions to standard PyTorch:
```python
unet = ray.train.torch.prepare_model(unet)       # DDP
train_ds = ray.train.get_dataset_shard("train")   # data parallelism
ray.train.report({"loss": avg_loss})              # metrics + checkpointing
```

We didn't rewrite the training code. Ray Train wraps it.


## Step 3: Launch Distributed LoRA Fine-Tuning

The next cell preprocesses the dataset and kicks off distributed training across **2 T4 GPUs** for 3 epochs.
Notice the code is standard PyTorch + diffusers + PEFT — Ray Train adds only 3 lines (`prepare_model`, `get_dataset_shard`, `report`). Scaling from 2 GPUs to 8 or 32 is a one-line config change: `ScalingConfig(num_workers=N)`.

**Business value:** If a spot instance is preempted mid-training, Ray Train automatically restores from the last checkpoint. No lost compute, no manual restart scripts.


```python
# Preprocess dataset: resize, normalize, tokenize captions
from src.data_pipeline import preprocess_batch
from src.train_lora import run_training

train_ds = ds.map_batches(
    preprocess_batch, batch_size=32, num_cpus=1, batch_format="numpy",
)

print("Launching distributed training: 2 T4 GPUs, 3 epochs")
print("TIP: Open Ray Dashboard -> Train to watch:")
print("  - Per-worker GPU utilization")
print("  - Training loss decreasing across epochs")
print("  - Ray Data feeding GPU workers without bottleneck\n")

result = run_training(
    num_workers=2,
    num_epochs=3,
    train_ds=train_ds,
)

print(f"\nFinal loss: {result.metrics['loss']:.4f}")
print(f"Checkpoint: {result.checkpoint}")

```

## Step 4: Generate Images from the Fine-Tuned Model

Now we load the LoRA checkpoint produced by Ray Train and run inference with the fine-tuned Stable Diffusion model.
The checkpoint is a lightweight LoRA adapter (~5 MB) — not a full model copy — so storage and serving costs stay minimal. This checkpoint was written automatically by Ray Train at each epoch boundary.

**Why it matters:** Fast iteration from training to inference on the same platform means your creative teams can evaluate style transfer quality in minutes, not hours.


```python
# Generate images from the fine-tuned LoRA checkpoint
from src.generate import generate_images

PROMPTS = [
    "a pokemon with blue fire",
    "a cute pokemon in a forest",
    "a legendary water pokemon",
    "a dragon type pokemon",
]

print("Generating images from fine-tuned model...")

@ray.remote(num_gpus=1)
def get_samples():    
    finetuned_images = generate_images(
        checkpoint_path=result.checkpoint.path,
        prompts=PROMPTS,
        seed=42,
    )
    return finetuned_images

finetuned_images = ray.get(get_samples.remote())
```

## Step 5: Compare Base Model vs. Fine-Tuned Results

This cell renders a side-by-side comparison: vanilla Stable Diffusion 1.5 outputs vs. our LoRA fine-tuned model.
The visual difference demonstrates that LoRA training on only **0.1% of parameters** for 3 epochs is enough to shift the model's style — a massive efficiency win over full fine-tuning.

**Takeaway for prospects:** Your teams get style-specific generation without the cost of training a full model. Anyscale makes the distributed training invisible.


```python
# Side-by-side: base SD 1.5 (pre-generated) vs fine-tuned LoRA
from src.utils import show_comparison

show_comparison(
    base_dir=os.path.join(DEMO_ROOT, "assets", "base_model_samples"),
    finetuned_images=finetuned_images,
    prompts=PROMPTS,
)
```

## Step 6: From Notebook to Production in One Command

The final cell shows the production path. The exact same training code runs as an **Anyscale Job** — no Docker rewrites, no Kubernetes YAML, no Helm charts.
Scaling from 2 GPUs to a full cluster is an entrypoint flag change. Anyscale handles node provisioning, fault tolerance, and log aggregation.

**The pitch:** What you just saw in this notebook is production-ready. `anyscale job submit` is the only command between here and a recurring training pipeline.


```python
print("Path to production as an Anyscale Job:")
print("""
  anyscale job submit -f job_config.yaml --working-dir ./
""")

print("Scale up:")
print("""
  anyscale job submit -f job_config.yaml --working-dir ./ \\
    --override-entrypoint 'python scripts/run_training.py --num-workers 8 --num-epochs 10'
""")

print("What you get with Ray Train vs. DIY distributed:")
rows = [
    ("Multi-GPU scaling",   "ScalingConfig(num_workers=N)", "torch.distributed.launch + NCCL env vars"),
    ("Data loading",        "Ray Data streaming",          "Custom DistributedSampler + DataLoader"),
    ("Fault tolerance",     "Automatic checkpoint restore", "Manual checkpoint + restart scripts"),
    ("Dev → Production",    "Same code, anyscale job submit", "Dockerize + K8s manifests + Helm"),
]
print(f"\n  {'Concern':<24} {'Ray Train':<34} {'DIY Distributed'}")
print(f"  {'─'*24} {'─'*34} {'─'*40}")
for concern, ray_answer, diy_answer in rows:
    print(f"  {concern:<24} {ray_answer:<34} {diy_answer}")

```


```python

```
