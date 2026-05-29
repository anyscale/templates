# BEV Model Training for Robotics

**⏱️ Time to complete**: 60 min

Train Bird's-Eye View (BEV) perception models for robotics using Ray Data for distributed preprocessing and Ray Train for fault-tolerant distributed training. This template demonstrates production-grade patterns for scaling multi-camera perception workloads on the Anyscale platform.

## What You'll Learn

By the end of this template, you'll be able to:
- Stage large robotics perception datasets (NuScenes) in cluster-visible storage
- Build scalable Ray Data pipelines for multi-camera image preprocessing and BEV label rasterization
- Train camera-only BEV Transformer models with Ray Train using Distributed Data Parallel (DDP)
- Implement checkpointing and fault tolerance for long-running perception training jobs
- Understand why distributed systems patterns matter for robotics perception

## Prerequisites

- Familiarity with PyTorch and computer vision basics
- Understanding of distributed training concepts (data parallelism, gradient synchronization)
- Basic knowledge of robotics coordinate frames (helpful but not required)

---

## 1. Introduction and Environment Setup

Bird's-Eye View (BEV) models are foundational for modern robotics and autonomous driving. They transform multi-camera observations into a unified top-down spatial representation, making downstream perception, planning, and control easier.

Training BEV models at scale is a **distributed systems problem**:
- Robotics datasets are large and heterogeneous (multi-camera images, lidar, poses, calibration)
- Preprocessing is CPU-intensive (image decoding, resizing, rasterization)
- Training requires multi-GPU coordination with fault tolerance

This template shows how to solve these challenges using:
- **Ray Data** for distributed CPU-side preprocessing
- **Ray Train** for multi-GPU training with checkpointing

We use the **NuScenes v1.0-mini** dataset (10 scenes, ~400 samples) as a practical example.


```python
# nuscenes-devkit pins matplotlib<3.6 (no py311 wheel -> source build) and GUI
# opencv-python; install it without deps, then its runtime deps from requirements.txt
# (the base image already provides matplotlib 3.7 and opencv-python-headless 4.13).
!pip install -q --no-deps nuscenes-devkit==1.1.11
!pip install -q -r requirements.txt
```


```python
import os
from pathlib import Path
import sys

# Print Python version for debugging environment issues
print("Python:", sys.version)

# Define cluster-visible dataset root
# Use /mnt/cluster_storage so all Ray workers can access the same files
NUSCENES_ROOT = Path(os.environ.get("NUSCENES_ROOT", "/mnt/cluster_storage/nuscenes")).resolve()

# Keep downloads separate from extracted dataset
DOWNLOAD_DIR = NUSCENES_ROOT / "_downloads"

# Create directories idempotently
NUSCENES_ROOT.mkdir(parents=True, exist_ok=True)
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

print("NUSCENES_ROOT =", NUSCENES_ROOT)
print("DOWNLOAD_DIR  =", DOWNLOAD_DIR)
```

---

## 2. Download NuScenes Dataset with Resume Support

Large robotics datasets require robust downloading that can resume after interruptions (network issues, spot instance preemption, notebook restarts).


```python
import requests
from tqdm.auto import tqdm

def download_with_resume(url: str, dst: Path, chunk_size: int = 1024 * 1024) -> Path:
    """
    Download a URL to dst with HTTP Range resume support.

    If a partial .part file exists, resume from where the previous download stopped.
    This is critical for large robotics datasets on shared infrastructure.
    """
    dst = Path(dst)
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Check how many bytes already downloaded
    existing = tmp.stat().st_size if tmp.exists() else 0

    # Request byte range if resuming
    headers = {"Range": f"bytes={existing}-"} if existing > 0 else {}

    with requests.get(url, stream=True, headers=headers, allow_redirects=True, timeout=60) as r:
        r.raise_for_status()

        # If server ignores Range (status 200 instead of 206), restart cleanly
        if existing > 0 and r.status_code == 200:
            existing = 0
            tmp.unlink(missing_ok=True)

        # Compute total size for progress bar
        total = r.headers.get("Content-Length")
        total = int(total) + existing if total is not None else None

        # Append if resuming, otherwise start fresh
        mode = "ab" if existing > 0 else "wb"
        with open(tmp, mode) as f, tqdm(
            total=total, initial=existing, unit="B", unit_scale=True, desc=dst.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Atomically move completed file into place
    tmp.rename(dst)
    return dst
```


```python
# Download NuScenes v1.0-mini dataset (~4 GB)
NUSCENES_MINI_URL = os.environ.get(
    "NUSCENES_MINI_URL",
    "https://www.nuscenes.org/data/v1.0-mini.tgz"
)

tgz_path = DOWNLOAD_DIR / "v1.0-mini.tgz"

# Skip download if archive already exists (idempotent)
if tgz_path.exists() and tgz_path.stat().st_size > 0:
    print("Already downloaded:", tgz_path, f"({tgz_path.stat().st_size/1e9:.2f} GB)")
else:
    print("Downloading:", NUSCENES_MINI_URL)
    download_with_resume(NUSCENES_MINI_URL, tgz_path)
    print("Saved to:", tgz_path)
```

---

## 3. Extract and Validate Dataset Structure

After downloading, extract the archive safely and validate that the NuScenes dataset structure is complete.


```python
import tarfile

# Extract only the dirs used here — maps/, samples/ (keyframes), v1.0-mini/
# (tables). Skip sweeps/: tens of thousands of unused intermediate-frame files
# whose write to network storage is the bulk of extract time.
KEEP_DIRS = {"maps", "samples", "v1.0-mini"}

def safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    """
    Extract the needed NuScenes dirs, preventing path traversal attacks.

    Skips sweeps/ (unused here); verifies every extracted member resolves
    within the target directory before extraction.
    """
    path = path.resolve()
    members = []
    for member in tar.getmembers():
        name = member.name[2:] if member.name.startswith("./") else member.name
        if name.split("/", 1)[0] not in KEEP_DIRS:
            continue
        member_path = (path / member.name).resolve()

        # Block entries that would escape target directory
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Blocked path traversal in tar member: {member.name}")
        members.append(member)

    tar.extractall(path, members=members)

# Expected top-level directories after extraction (sweeps/ intentionally skipped)
expected = ["maps", "samples", "v1.0-mini"]

# Check if dataset already extracted
already = all((NUSCENES_ROOT / e).exists() for e in expected)

if already:
    print("Looks already extracted. Found:", expected)
else:
    print("Extracting to:", NUSCENES_ROOT)
    with tarfile.open(tgz_path, "r:gz") as tar:
        safe_extract(tar, NUSCENES_ROOT)
    print("Done.")
    print("Now present:", [p.name for p in NUSCENES_ROOT.iterdir()])
```


```python
# Initialize NuScenes API and validate dataset
%matplotlib inline

from nuscenes.nuscenes import NuScenes

# Initialize NuScenes object
# Setting verbose=True prints dataset loading info (useful for catching missing files)
nusc = NuScenes(version="v1.0-mini", dataroot=str(NUSCENES_ROOT), verbose=True)

# Print dataset scale
print("\nDataset Statistics:")
print(f"  Scenes: {len(nusc.scene)}")
print(f"  Samples: {len(nusc.sample)}")
print(f"  Sample data: {len(nusc.sample_data)}")
print(f"  Annotations: {len(nusc.sample_annotation)}")
```

---

## 4. Explore Multi-Sensor Data Layout

Before building the training pipeline, inspect NuScenes data to understand the multi-sensor setup and coordinate frames.


```python
# Select first scene and its first sample
scene = nusc.scene[0]
first_sample_token = scene["first_sample_token"]
sample = nusc.get("sample", first_sample_token)

print("Scene name:", scene.get("name"))
print("Sample token:", first_sample_token)
print("Available sensors:", sorted(sample["data"].keys()))
print("Num annotations:", len(sample["anns"]))
```


```python
# Visualize front camera with 3D bounding boxes
cam_token = sample["data"]["CAM_FRONT"]
nusc.render_sample_data(cam_token)
```


```python
# Visualize lidar in ego frame with map overlay
# This is the coordinate system BEV labels will use
lidar_token = sample["data"]["LIDAR_TOP"]
# nsweeps=1: keyframe lidar only (lives in samples/; sweeps/ is not extracted)
nusc.render_sample_data(lidar_token, nsweeps=1, underlay_map=True)
```


```python
# Project lidar points into camera image to verify calibration
# Important: if lidar doesn't align with camera pixels, BEV labels will be wrong
nusc.render_pointcloud_in_image(sample["token"], pointsensor_channel="LIDAR_TOP")
```

---

## 5. Build Lightweight Training Manifest

Create a token-based manifest that decouples training from heavyweight NuScenes objects. This is critical for distributed training scalability.

**Why manifests?**
- NuScenes scenes are large objects expensive to serialize to Ray workers
- Token-based manifests are lightweight (just strings + floats)
- Manifests enable deterministic train/val splits across runs


```python
import json
import random
from pyquaternion import Quaternion

def build_manifest_from_tokens(nusc, tokens, cam_channels):
    """
    Convert sample tokens into lightweight manifest items.

    Extract only what's needed for training:
    - Camera image file paths (absolute paths for workers)
    - Ego pose at lidar timestamp (consistent coordinate frame)
    - Simplified 3D annotation boxes (for BEV label rasterization)
    """
    items = []
    skipped = 0

    for tok in tokens:
        sample = nusc.get("sample", tok)

        # Anchor ego pose at lidar time for consistent BEV frame
        if "LIDAR_TOP" not in sample["data"]:
            skipped += 1
            continue

        lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        ego_translation = ego_pose["translation"]  # global [x, y, z]
        ego_rotation = ego_pose["rotation"]        # quaternion [w, x, y, z]

        # Collect absolute camera image paths
        cam_paths = []
        ok = True
        for ch in cam_channels:
            if ch not in sample["data"]:
                ok = False
                break
            sd = nusc.get("sample_data", sample["data"][ch])
            # Convert to absolute path so workers can open directly
            p = Path(sd["filename"])
            cam_paths.append(str(p if p.is_absolute() else (NUSCENES_ROOT / p).resolve()))

        if not ok or len(cam_paths) != len(cam_channels):
            skipped += 1
            continue

        # Collect simplified annotation records (global frame)
        anns = []
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            anns.append({
                "translation": ann["translation"],   # global [x, y, z]
                "size": ann["size"],                 # [w, l, h]
                "rotation": ann["rotation"],         # quaternion [w, x, y, z]
                "category_name": ann["category_name"],
            })

        items.append({
            "sample_token": tok,
            "cam_paths": cam_paths,
            "ego_translation": ego_translation,
            "ego_rotation": ego_rotation,
            "anns": anns,
        })

    print(f"Manifest built: {len(items)} samples (skipped {skipped})")
    return items

# Create a subset for fast iteration (size via BEV_SUBSET_SIZE env; default 200)
SUBSET_DIR = NUSCENES_ROOT / "subsets"
SUBSET_DIR.mkdir(parents=True, exist_ok=True)

subset_size = int(os.environ.get("BEV_SUBSET_SIZE", "200"))
tokens = []

# Walk scenes and collect sample tokens
for scene in nusc.scene:
    cur = scene["first_sample_token"]
    while cur and len(tokens) < subset_size:
        tokens.append(cur)
        s = nusc.get("sample", cur)
        cur = s["next"]
    if len(tokens) >= subset_size:
        break

# Save subset as JSON manifest
subset_path = SUBSET_DIR / f"nuscenes_mini_subset_{subset_size}.json"
subset_path.write_text(json.dumps({"version": "v1.0-mini", "tokens": tokens}, indent=2))

print("Wrote:", subset_path)
print("Num tokens:", len(tokens))
```


```python
# Build lightweight manifest
CAM_CHANNELS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]

manifest = build_manifest_from_tokens(nusc, tokens, CAM_CHANNELS)
print("First manifest item:", manifest[0]["sample_token"])
```


```python
# Split into train/val (80/20)
SEED = 123
rng = random.Random(SEED)
rng.shuffle(manifest)

n_total = len(manifest)
n_train = max(1, int(0.8 * n_total))

train_items = manifest[:n_train]
val_items = manifest[n_train:]

print(f"Total: {n_total}, Train: {len(train_items)}, Val: {len(val_items)}")
```

---

## 6. Define BEV Preprocessing Logic

Transform raw NuScenes samples into fixed-shape tensors for GPU training. This function runs on CPU via Ray Data.


```python
import numpy as np
import cv2
from PIL import Image
from nuscenes.utils.data_classes import Box

# ImageNet normalization for CNN backbone
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Image resize parameters (keep small for demo)
IMG_H = 128
IMG_W = 224
NUM_CAMS = 6

# BEV grid parameters (ego-centric, meters)
BEV_X_MIN, BEV_X_MAX = -25.0, 25.0   # forward/back
BEV_Y_MIN, BEV_Y_MAX = -25.0, 25.0   # left/right
BEV_RES = 1.0                        # meters per pixel

BEV_W = int(round((BEV_X_MAX - BEV_X_MIN) / BEV_RES))
BEV_H = int(round((BEV_Y_MAX - BEV_Y_MIN) / BEV_RES))

NUM_CLASSES = 3  # 0=background, 1=vehicle, 2=pedestrian

print(f"IMG: ({IMG_H}, {IMG_W})")
print(f"BEV: ({BEV_H}, {BEV_W}), res={BEV_RES} m/px")
```


```python
def category_to_class_id(cat: str) -> int:
    """Map NuScenes category to simple demo label space."""
    if cat.startswith("vehicle."):
        return 1
    if cat.startswith("human.pedestrian"):
        return 2
    return 0

def rasterize_bev_labels(anns, ego_translation, ego_rotation):
    """
    Rasterize 3D annotations into (BEV_H, BEV_W) ego-centric grid.

    Steps:
    1. Transform each 3D box from global frame to ego frame
    2. Extract bottom-face corners (BEV footprint)
    3. Convert ego-frame meters to BEV pixel coordinates
    4. Rasterize polygon with class ID
    """
    grid = np.zeros((BEV_H, BEV_W), dtype=np.uint8)

    ego_t = np.array(ego_translation, dtype=np.float32)
    ego_q = Quaternion(ego_rotation)

    for ann in anns:
        cls = category_to_class_id(ann["category_name"])
        if cls == 0:
            continue

        # Construct box in global frame
        box = Box(
            center=np.array(ann["translation"], dtype=np.float32),
            size=np.array(ann["size"], dtype=np.float32),
            orientation=Quaternion(ann["rotation"]),
        )

        # Transform to ego frame
        box.translate(-ego_t)
        box.rotate(ego_q.inverse)

        # Extract bottom-face corners
        corners = box.corners().T  # (8, 3)
        z = corners[:, 2]
        zmin = float(z.min())
        bottom = corners[z < (zmin + 1e-2), :2]  # (4, 2) typically

        if bottom.shape[0] < 3:
            continue

        x = bottom[:, 0]
        y = bottom[:, 1]

        # Fast reject if box outside BEV region
        if x.max() < BEV_X_MIN or x.min() > BEV_X_MAX or \
           y.max() < BEV_Y_MIN or y.min() > BEV_Y_MAX:
            continue

        # Convert ego meters to BEV pixels
        px = (x - BEV_X_MIN) / BEV_RES
        py = (BEV_Y_MAX - y) / BEV_RES  # y-up -> row-down

        poly = np.stack([px, py], axis=1).astype(np.int32)
        if poly.shape[0] >= 3:
            hull = cv2.convexHull(poly)
            cv2.fillConvexPoly(grid, hull, int(cls))

    return grid.astype(np.int64)

def load_and_preprocess_images(cam_paths):
    """
    Load, resize, and normalize multi-camera images.

    Returns: (NUM_CAMS, 3, IMG_H, IMG_W) float32 array
    """
    imgs = []
    for p in cam_paths:
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize((IMG_W, IMG_H), resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # HWC
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            arr = np.transpose(arr, (2, 0, 1))  # CHW
            imgs.append(arr)
    return np.stack(imgs, axis=0).astype(np.float32)

def preprocess_nuscenes_batch(batch: dict) -> dict:
    """
    Preprocess batch of manifest items into model-ready arrays.

    Ray Data passes dict-of-lists. Return fixed-shape NumPy arrays.
    """
    out_images = []
    out_labels = []

    cam_paths_list = batch["cam_paths"]
    ego_t_list = batch["ego_translation"]
    ego_r_list = batch["ego_rotation"]
    anns_list = batch["anns"]

    for cam_paths, ego_t, ego_r, anns in zip(cam_paths_list, ego_t_list, ego_r_list, anns_list):
        imgs = load_and_preprocess_images(cam_paths)
        lbls = rasterize_bev_labels(anns, ego_t, ego_r)

        out_images.append(imgs)  # (6, 3, H, W)
        out_labels.append(lbls)  # (BEV_H, BEV_W)

    return {
        "images": np.stack(out_images, axis=0),  # (B, 6, 3, IMG_H, IMG_W)
        "labels": np.stack(out_labels, axis=0),  # (B, BEV_H, BEV_W)
    }
```

---

## 7. Build Ray Data Datasets for Distributed Preprocessing

Create Ray Data datasets that parallelize CPU-heavy preprocessing across the cluster.


```python
import ray

# Build Ray Data datasets
# Preprocessing scales independently of GPU training
train_ds = (
    ray.data.from_items(train_items)
    .map_batches(preprocess_nuscenes_batch, batch_size=2, batch_format="numpy")
    .materialize()
)

val_ds = (
    ray.data.from_items(val_items)
    .map_batches(preprocess_nuscenes_batch, batch_size=2, batch_format="numpy")
    .materialize()
)

# Materialized above, so preprocessing runs once and is reused by
# count()/schema() here and by training (avoids 5-6x recompute).
print("Train ds count:", train_ds.count())
print("Val ds count:", val_ds.count())
print("Train schema:", train_ds.schema())
```


```python
# Sanity check: pull one batch and verify shapes
b = next(iter(train_ds.iter_batches(batch_size=1)))

print("images:", b["images"].shape, b["images"].dtype)  # (1, 6, 3, 128, 224) float32
print("labels:", b["labels"].shape, b["labels"].dtype)  # (1, 50, 50) int64
print("labels unique:", np.unique(b["labels"]))  # [0, 1, 2] or subset
```

---

## 8. Define Camera-Only BEV Transformer Model

Build a minimal BEV Transformer for pedagogical clarity. This model learns BEV structure implicitly from data without camera geometry.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CrossAttentionBlock(nn.Module):
    """Update BEV tokens by cross-attending to image tokens, then apply MLP."""
    def __init__(self, d_model: int, nhead: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, bev_tokens: torch.Tensor, img_tokens: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(bev_tokens)
        kv = self.norm_kv(img_tokens)
        x = self.attn(q, kv, kv, need_weights=False)[0]
        bev_tokens = bev_tokens + x
        bev_tokens = bev_tokens + self.mlp(self.norm2(bev_tokens))
        return bev_tokens

class SimpleBEVTransformer(nn.Module):
    """
    Camera-only BEV Transformer.

    Architecture:
    1. Shared ResNet-18 backbone extracts features from each camera
    2. Learnable BEV query tokens (one per BEV cell)
    3. Cross-attention: BEV queries attend to all camera features
    4. Lightweight segmentation head predicts per-cell logits

    Input: images (B, num_cams, 3, IMG_H, IMG_W)
    Output: logits (B, num_classes, BEV_H, BEV_W)
    """
    def __init__(
        self,
        img_h: int,
        img_w: int,
        num_cams: int,
        bev_h: int,
        bev_w: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.num_cams = num_cams
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_classes = num_classes
        self.d_model = d_model

        # Shared CNN backbone for all cameras
        resnet = torchvision.models.resnet18(weights=None)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        # Project backbone output to transformer dimension
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)

        # Compute feature grid size once
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_h, img_w)
            feat = self.proj(self.backbone(dummy))
            self.feat_h, self.feat_w = int(feat.shape[-2]), int(feat.shape[-1])

        # Learnable embeddings
        self.img_pos = nn.Parameter(torch.zeros(1, self.feat_h * self.feat_w, d_model))
        self.cam_embed = nn.Embedding(num_cams, d_model)

        # Learnable BEV query tokens
        self.bev_pos = nn.Parameter(torch.zeros(1, bev_h * bev_w, d_model))
        self.bev_query = nn.Parameter(torch.zeros(1, bev_h * bev_w, d_model))

        # Cross-attention blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model, num_classes, kernel_size=1),
        )

        self._init_params()

    def _init_params(self):
        nn.init.trunc_normal_(self.img_pos, std=0.02)
        nn.init.trunc_normal_(self.bev_pos, std=0.02)
        nn.init.trunc_normal_(self.bev_query, std=0.02)
        nn.init.normal_(self.cam_embed.weight, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N, C, H, W = images.shape
        assert N == self.num_cams

        # Batch all cameras for efficient CNN forward
        x = images.reshape(B * N, C, H, W)
        feat = self.backbone(x)  # (B*N, 512, fh, fw)
        feat = self.proj(feat)   # (B*N, d, fh, fw)

        # Flatten camera feature grids into token sequence
        d, fh, fw = feat.shape[1], feat.shape[2], feat.shape[3]
        feat = feat.reshape(B, N, d, fh, fw).permute(0, 1, 3, 4, 2)  # (B, N, fh, fw, d)
        img_tokens = feat.reshape(B, N * fh * fw, d)                 # (B, N*fh*fw, d)

        # Add camera-ID and spatial embeddings
        cam_ids = torch.arange(N, device=images.device, dtype=torch.long)
        cam_e = self.cam_embed(cam_ids)  # (N, d)
        cam_e = cam_e[:, None, :].expand(N, fh * fw, d).reshape(N * fh * fw, d)
        img_tokens = img_tokens + cam_e.unsqueeze(0)

        pos = self.img_pos.repeat(1, N, 1)
        img_tokens = img_tokens + pos

        # Initialize BEV tokens and cross-attend
        bev = self.bev_query + self.bev_pos
        bev = bev.expand(B, -1, -1)

        for blk in self.blocks:
            bev = blk(bev, img_tokens)

        # Reshape to BEV map and predict logits
        bev_map = bev.transpose(1, 2).reshape(B, d, self.bev_h, self.bev_w)
        logits = self.head(bev_map)
        return logits
```


```python
# Validate model forward pass
m = SimpleBEVTransformer(
    img_h=IMG_H, img_w=IMG_W,
    num_cams=NUM_CAMS,
    bev_h=BEV_H, bev_w=BEV_W,
    num_classes=NUM_CLASSES,
    d_model=256, nhead=8, num_layers=2, dropout=0.1,
)

with torch.no_grad():
    dummy = torch.zeros(1, NUM_CAMS, 3, IMG_H, IMG_W)
    out = m(dummy)

print("Model output:", out.shape)  # (1, 3, 50, 50)
print("Feature map (fh, fw):", (m.feat_h, m.feat_w))  # Small grid after stride-32 backbone
```

---

## 9. Define Ray Train Worker Loop with DDP

Implement the per-worker training function with DDP, mixed precision, checkpointing, and metric aggregation.


```python
from ray.train.torch import prepare_model

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_loop_per_worker(config: dict):
    """
    Per-worker training loop.

    Responsibilities:
    - Build and DDP-wrap model
    - Stream batches from Ray Data shards
    - Train + evaluate for multiple epochs
    - Resume from checkpoint if exists
    - Save checkpoint after each epoch (rank 0 only)
    - Report aggregated metrics across all workers
    """
    import os, tempfile
    import ray.cloudpickle as pickle
    import torch
    import torch.distributed as dist
    from ray import train
    from ray.train import Checkpoint

    # Reduce CUDA fragmentation. Must run in the worker process before CUDA
    # initializes; setting it on the driver has no effect on the workers.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    set_seed(int(config.get("seed", 123)))

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # Build model
    model = SimpleBEVTransformer(
        img_h=int(config["img_h"]),
        img_w=int(config["img_w"]),
        num_cams=int(config["num_cams"]),
        bev_h=int(config["bev_h"]),
        bev_w=int(config["bev_w"]),
        num_classes=int(config["num_classes"]),
        d_model=int(config.get("d_model", 256)),
        nhead=int(config.get("nhead", 8)),
        num_layers=int(config.get("num_layers", 2)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    # Wrap with DDP via Ray Train
    model = prepare_model(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 3e-4)),
        weight_decay=float(config.get("weight_decay", 1e-2)),
    )

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Resume from checkpoint
    start_epoch = 0
    step = 0
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as d:
            with open(os.path.join(d, "state.pkl"), "rb") as f:
                state = pickle.load(f)

        model.module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optim"])
        scaler.load_state_dict(state.get("scaler", scaler.state_dict()))
        start_epoch = int(state["epoch"]) + 1
        step = int(state.get("step", 0))

    # Get Ray Data shards
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    batch_size = int(config.get("batch_size", 1))
    grad_accum = int(config.get("grad_accum", 1))
    num_epochs = int(config.get("num_epochs", 3))

    def run_epoch(shard, train_mode: bool):
        """Run one epoch of training or evaluation."""
        nonlocal step
        model.train(train_mode)

        loss_sum = 0.0
        correct = 0
        total = 0
        n_batches = 0

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        # Stream batches from Ray Data shard
        for batch in shard.iter_torch_batches(
            batch_size=batch_size,
            dtypes={"images": torch.float32, "labels": torch.int64},
            device=device,
        ):
            images = batch["images"]  # (B, 6, 3, IMG_H, IMG_W)
            labels = batch["labels"]  # (B, BEV_H, BEV_W)

            # Forward pass with mixed precision; disable autograd during eval
            # to avoid building the graph (saves memory / OOM risk).
            with torch.set_grad_enabled(train_mode), torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda"), dtype=torch.float16
            ):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            if train_mode:
                # Gradient accumulation
                scaler.scale(loss / grad_accum).backward()
                step += 1
                if step % grad_accum == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            loss_sum += float(loss.detach().item())
            n_batches += 1

            # Pixel accuracy
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                correct += int((pred == labels).sum().item())
                total += int(labels.numel())

        # Aggregate metrics across workers
        loss_t = torch.tensor([loss_sum, n_batches, correct, total], device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)

        loss_sum_all, n_batches_all, correct_all, total_all = loss_t.tolist()
        avg_loss = loss_sum_all / max(1.0, n_batches_all)
        acc = correct_all / max(1.0, total_all)
        return avg_loss, acc

    # Train/eval loop with checkpointing
    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = run_epoch(train_shard, train_mode=True)
        val_loss, val_acc = run_epoch(val_shard, train_mode=False)

        metrics = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        # Checkpoint on rank 0
        ctx = train.get_context()
        rank = ctx.get_world_rank()

        if rank == 0:
            with tempfile.TemporaryDirectory(prefix="bev_ckpt_") as ckpt_dir:
                state = {
                    "model": model.module.state_dict(),
                    "optim": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "step": step,
                }
                with open(os.path.join(ckpt_dir, "state.pkl"), "wb") as f:
                    pickle.dump(state, f)
                ckpt = Checkpoint.from_directory(ckpt_dir)
                train.report(metrics, checkpoint=ckpt)
        else:
            train.report(metrics)
```

---

## 10. Launch Distributed Training with TorchTrainer

Configure and launch the distributed BEV training job with Ray Train.


```python
from ray.train import RunConfig, ScalingConfig, FailureConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

# Persistent storage for checkpoints and logs
PERSIST_ROOT = Path("/mnt/cluster_storage")
PERSIST_ROOT.mkdir(parents=True, exist_ok=True)

RUN_STORAGE_PATH = str(PERSIST_ROOT / "ray_train_runs" / "bev_robotics")
RUN_NAME = "bev-camera-only"

# Configure TorchTrainer
trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        # Reproducibility
        "seed": SEED,
        # Input/output shapes
        "img_h": IMG_H,
        "img_w": IMG_W,
        "num_cams": NUM_CAMS,
        "bev_h": BEV_H,
        "bev_w": BEV_W,
        "num_classes": NUM_CLASSES,
        # Model hyperparameters
        "d_model": 256,
        "nhead": 8,
        "num_layers": 2,
        "dropout": 0.1,
        # Optimizer hyperparameters
        "lr": 3e-4,
        "weight_decay": 1e-2,
        # Training hyperparameters
        "batch_size": 1,     # Small batch fits on modest GPUs
        "grad_accum": 1,
        "num_epochs": int(os.environ.get("BEV_NUM_EPOCHS", "3")),
    },
    scaling_config=ScalingConfig(
        num_workers=2,  # 2-worker DDP
        use_gpu=True,   # 1 GPU per worker
    ),
    run_config=RunConfig(
        name=RUN_NAME,
        storage_path=RUN_STORAGE_PATH,
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_acc",
            checkpoint_score_order="max",
        ),
        failure_config=FailureConfig(max_failures=1),  # Retry once on failure
    ),
    datasets={"train": train_ds, "val": val_ds},
)

print("Starting distributed training...")
print(f"  Workers: 2 (DDP)")
print(f"  GPUs: 1 per worker")
print(f"  Storage: {RUN_STORAGE_PATH}")

result = trainer.fit()
```

---

## 11. Analyze Training Results and Checkpoints

Inspect training outcomes and understand checkpoint structure for future resume.


```python
# Print final metrics
print("\nTraining Complete!")
print(f"  Final train loss: {result.metrics['train_loss']:.4f}")
print(f"  Final train acc: {result.metrics['train_acc']:.4f}")
print(f"  Final val loss: {result.metrics['val_loss']:.4f}")
print(f"  Final val acc: {result.metrics['val_acc']:.4f}")
```


```python
# Inspect checkpoint structure
print("\nCheckpoint Info:")
print(f"  Path: {result.checkpoint.path if result.checkpoint else 'None'}")
print(f"  Best val_acc: {result.metrics.get('val_acc', 'N/A')}")

# Show how to resume training
print("\nTo resume training:")
print("  1. TorchTrainer automatically loads latest checkpoint if run exists")
print("  2. Checkpoint contains: model state, optimizer state, scaler state, epoch, step")
print("  3. Training continues from last completed epoch")
```

---

## 12. Best Practices and Next Steps

### Key Takeaways

**Distributed systems patterns:**
- Ray Data scales CPU preprocessing independently of GPU training
- Token-based manifests decouple dataset indexing from execution
- Ray Train provides DDP, checkpointing, and fault tolerance with minimal code

**When to use this pipeline:**
- Multi-camera robotics perception (BEV, occupancy, depth estimation)
- Large datasets where preprocessing is CPU-bound
- Long-running training jobs requiring fault tolerance

**Production tips:**
- Use cluster-visible storage (`/mnt/cluster_storage`) for datasets and checkpoints
- Enable mixed precision (`torch.amp`) to reduce memory and speed up training
- Aggregate metrics with `torch.distributed.all_reduce()` for correct global values
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation

### Model Limitations

This camera-only BEV Transformer is intentionally simple:
- **No camera geometry** - Model learns BEV implicitly, not from known intrinsics/extrinsics
- **Coarse supervision** - Rasterized boxes, not dense depth or occupancy
- **Small scale** - NuScenes mini (10 scenes) for fast iteration

These limitations are deliberate to focus on **distributed execution patterns**, not modeling complexity.

### Next Steps

**Scale the pipeline:**
- Use full NuScenes (1000 scenes) - same code, larger manifest
- Increase workers and GPUs for faster training
- Enable gradient checkpointing for larger models

**Improve the model:**
- Add explicit camera geometry (view transformations, depth prediction)
- Use LSS (Lift-Splat-Shoot) or BEVFormer architectures
- Integrate lidar for multimodal BEV models
- Add temporal context across frames

**Production deployment:**
- Export trained model to ONNX for inference
- Deploy with Ray Serve for online perception
- Integrate with downstream planning/control

### Documentation

- [Ray Data User Guide](https://docs.ray.io/en/latest/data/data.html)
- [Ray Train User Guide](https://docs.ray.io/en/latest/train/train.html)
- [NuScenes Dataset](https://www.nuscenes.org/)
- [BEV Perception Survey](https://arxiv.org/abs/2206.07959)

---

## Summary

You have built a production-grade BEV training pipeline:

1. **Staged NuScenes dataset** in cluster storage with robust download/extraction
2. **Created lightweight manifest** to scale training independently of dataset objects
3. **Built Ray Data pipeline** for distributed multi-camera preprocessing and BEV label rasterization
4. **Trained camera-only BEV Transformer** with Ray Train using DDP, mixed precision, and checkpointing
5. **Implemented fault tolerance** with automatic checkpoint resume

The execution pattern demonstrated here—Ray Data for preprocessing, Ray Train for distributed training, checkpointing for fault tolerance—applies directly to more complex BEV architectures and larger robotics perception datasets.

You now have a solid foundation for training BEV models at scale on real infrastructure.
