"""
Shared utilities for the e-commerce recommendation system demo.
Kept in a single .py file so both notebooks and scripts can import it.
"""

import io
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORIES = ["Electronics", "Clothing", "Books", "Home & Garden", "Sports"]

CATEGORY_COLORS = {
    "Electronics":   (70,  130, 180),   # steel blue
    "Clothing":      (220,  90, 120),   # rose
    "Books":         (120, 170,  60),   # olive green
    "Home & Garden": (200, 150,  50),   # golden
    "Sports":        (80,  180, 140),   # teal
}

PRODUCTS = [
    # Electronics
    {"name": "Wireless Headphones",   "category": "Electronics",   "price": 79.99,  "description": "Over-ear noise-cancelling Bluetooth headphones with 30-hour battery life."},
    {"name": "Mechanical Keyboard",   "category": "Electronics",   "price": 99.99,  "description": "Compact TKL mechanical keyboard with tactile switches and RGB backlight."},
    {"name": "USB-C Hub",             "category": "Electronics",   "price": 39.99,  "description": "7-in-1 USB-C hub with HDMI, SD card, and 100W power delivery."},
    {"name": "Webcam 1080p",          "category": "Electronics",   "price": 59.99,  "description": "Full HD webcam with built-in microphone for video conferencing."},
    {"name": "Portable SSD 1TB",      "category": "Electronics",   "price": 109.99, "description": "Ultra-fast portable SSD with USB 3.2 Gen 2 and rugged aluminum case."},
    {"name": "Smart Watch",           "category": "Electronics",   "price": 199.99, "description": "Fitness smart watch with heart-rate monitor and 7-day battery."},
    {"name": "Bluetooth Speaker",     "category": "Electronics",   "price": 49.99,  "description": "Waterproof portable speaker with 360° sound and 12-hour battery."},
    {"name": "Laptop Stand",          "category": "Electronics",   "price": 29.99,  "description": "Adjustable aluminum laptop stand compatible with 10-16 inch laptops."},
    # Clothing
    {"name": "Running Shoes",         "category": "Clothing",      "price": 89.99,  "description": "Lightweight breathable running shoes with cushioned sole and wide toe box."},
    {"name": "Merino Wool Sweater",   "category": "Clothing",      "price": 69.99,  "description": "Soft and warm 100% merino wool crewneck sweater, machine washable."},
    {"name": "Waterproof Jacket",     "category": "Clothing",      "price": 129.99, "description": "Lightweight packable rain jacket with taped seams and adjustable hood."},
    {"name": "Slim-Fit Chinos",       "category": "Clothing",      "price": 54.99,  "description": "Stretch slim-fit chino trousers available in multiple neutral colors."},
    {"name": "Cotton T-Shirt 3-Pack", "category": "Clothing",      "price": 29.99,  "description": "Everyday crewneck cotton tees in classic white, black, and grey."},
    {"name": "Leather Belt",          "category": "Clothing",      "price": 34.99,  "description": "Full-grain leather dress belt with silver-tone pin buckle."},
    {"name": "Hiking Boots",          "category": "Clothing",      "price": 149.99, "description": "Waterproof mid-cut hiking boots with vibram sole and ankle support."},
    {"name": "Puffer Vest",           "category": "Clothing",      "price": 59.99,  "description": "Lightweight down-fill vest with snap front closure and two hand pockets."},
    # Books
    {"name": "Deep Learning Book",    "category": "Books",         "price": 44.99,  "description": "Comprehensive guide to deep learning foundations and modern architectures."},
    {"name": "Python Cookbook",       "category": "Books",         "price": 39.99,  "description": "Practical recipes for writing idiomatic, modern Python code."},
    {"name": "Distributed Systems",   "category": "Books",         "price": 49.99,  "description": "Principles and patterns of building reliable distributed systems at scale."},
    {"name": "Clean Code",            "category": "Books",         "price": 34.99,  "description": "A handbook of agile software craftsmanship principles and best practices."},
    {"name": "The Algorithm Design",  "category": "Books",         "price": 59.99,  "description": "Classic reference covering algorithm design techniques with worked examples."},
    {"name": "Designing Data-Intensive Apps", "category": "Books", "price": 54.99,  "description": "In-depth look at the systems behind modern data-intensive applications."},
    # Home & Garden
    {"name": "French Press Coffee",   "category": "Home & Garden", "price": 29.99,  "description": "Stainless steel French press with double-wall insulation, 34 oz."},
    {"name": "Cast Iron Skillet",     "category": "Home & Garden", "price": 39.99,  "description": "Pre-seasoned 10-inch cast iron skillet suitable for all cooktops."},
    {"name": "Bamboo Cutting Board",  "category": "Home & Garden", "price": 24.99,  "description": "Large reversible bamboo cutting board with juice groove and handle."},
    {"name": "Air Purifier",          "category": "Home & Garden", "price": 119.99, "description": "True HEPA air purifier for rooms up to 500 sq ft, whisper-quiet."},
    {"name": "Succulent Set 6-Pack",  "category": "Home & Garden", "price": 19.99,  "description": "Assorted live succulents in 2-inch pots, easy-care indoor plants."},
    {"name": "Scented Soy Candle",    "category": "Home & Garden", "price": 18.99,  "description": "Hand-poured lavender and vanilla soy wax candle, 50-hour burn time."},
    # Sports
    {"name": "Yoga Mat",              "category": "Sports",        "price": 34.99,  "description": "6mm thick non-slip yoga mat with carry strap, eco-friendly TPE."},
    {"name": "Resistance Bands Set",  "category": "Sports",        "price": 24.99,  "description": "Set of 5 fabric resistance bands with varying tension levels."},
    {"name": "Jump Rope",             "category": "Sports",        "price": 14.99,  "description": "Adjustable speed jump rope with ball-bearing handles and steel cable."},
    {"name": "Water Bottle 32oz",     "category": "Sports",        "price": 29.99,  "description": "Insulated stainless steel water bottle keeps drinks cold 24 hr / hot 12 hr."},
    {"name": "Foam Roller",           "category": "Sports",        "price": 22.99,  "description": "High-density foam roller for muscle recovery and myofascial release."},
    {"name": "Dumbbell Set",          "category": "Sports",        "price": 89.99,  "description": "Adjustable dumbbell set 5-25 lb per hand with compact storage tray."},
]

IMAGE_SIZE = (224, 224)

# Adjectives and category-specific suffixes used to generate product variants
_VARIANT_ADJECTIVES = [
    "Premium", "Pro", "Lite", "Ultra", "Essential", "Plus",
    "Max", "Mini", "Elite", "Classic", "Advanced", "Standard",
    "Signature", "Sport", "Studio", "Travel",
]

_VARIANT_SUFFIXES = {
    "Electronics": [
        "Includes extended warranty.",
        "Optimized for remote work.",
        "Designed for content creators.",
        "Features USB-C connectivity.",
        "With advanced noise reduction.",
        "Smart power management included.",
    ],
    "Clothing": [
        "Available in 8 colors.",
        "Made from sustainable materials.",
        "Reinforced stitching throughout.",
        "Features moisture-wicking fabric.",
        "UPF 50+ sun protection.",
        "Slim-fit design.",
    ],
    "Books": [
        "Includes digital access code.",
        "Updated 2024 edition.",
        "With practice exercises throughout.",
        "Annotated edition.",
        "Includes 100+ code examples.",
        "With online companion resources.",
    ],
    "Home & Garden": [
        "Dishwasher safe.",
        "BPA-free construction.",
        "Lifetime warranty included.",
        "Eco-friendly packaging.",
        "NSF certified.",
        "Made from recycled materials.",
    ],
    "Sports": [
        "With antimicrobial coating.",
        "Suitable for all fitness levels.",
        "Includes carrying bag.",
        "Made from recycled materials.",
        "Ergonomic grip design.",
        "Machine washable.",
    ],
}


# ---------------------------------------------------------------------------
# Synthetic image generation
# ---------------------------------------------------------------------------

def make_product_image(product: Dict, seed: int = 0) -> np.ndarray:
    """Generate a synthetic product image as a numpy array (H, W, 3) uint8.

    The image is a solid-colored rectangle with the product name overlaid,
    so it's visually distinct per category without requiring real photos.
    """
    rng = random.Random(seed)
    base_color = CATEGORY_COLORS[product["category"]]

    # Add slight per-product color jitter
    color = tuple(
        max(0, min(255, c + rng.randint(-30, 30))) for c in base_color
    )

    img = Image.new("RGB", IMAGE_SIZE, color=color)
    draw = ImageDraw.Draw(img)

    # Draw a centered rounded rectangle as a "product card"
    margin = 20
    draw.rounded_rectangle(
        [margin, margin, IMAGE_SIZE[0] - margin, IMAGE_SIZE[1] - margin],
        radius=15,
        fill=tuple(max(0, c - 40) for c in color),
    )

    # Write product name (word-wrap at ~15 chars)
    words = product["name"].split()
    lines, line = [], []
    for word in words:
        if sum(len(w) for w in line) + len(line) + len(word) <= 15:
            line.append(word)
        else:
            lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))

    # Try to use a basic font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    total_h = len(lines) * 24
    y = (IMAGE_SIZE[1] - total_h) // 2 - 10
    for line_text in lines:
        bbox = draw.textbbox((0, 0), line_text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((IMAGE_SIZE[0] - w) // 2, y), line_text, fill="white", font=font)
        y += 24

    # Category label at bottom
    cat = product["category"]
    bbox = draw.textbbox((0, 0), cat, font=small_font)
    w = bbox[2] - bbox[0]
    draw.text(((IMAGE_SIZE[0] - w) // 2, IMAGE_SIZE[1] - 35), cat, fill="white", font=small_font)

    return np.array(img)


_DEMO_IMAGES_DIR = Path(__file__).parent.parent / "data" / "demo_images"


def get_product_image(product: Dict) -> np.ndarray:
    """Load the bundled realistic product image, falling back to synthetic.

    Variant products created by ``expand_catalog`` carry a ``_base_name`` key
    pointing to their base product so they reuse that product's real image.
    """
    image_name = product.get("_base_name", product["name"])
    safe_name = image_name.replace(" ", "_").replace("/", "-")
    path = _DEMO_IMAGES_DIR / f"{safe_name}.jpg"
    if path.exists():
        return np.array(Image.open(path).convert("RGB"))
    return make_product_image(product, seed=hash(product["name"]) % 1000)


def image_to_bytes(img_array: np.ndarray, fmt: str = "JPEG") -> bytes:
    """Convert numpy (H,W,3) uint8 array to compressed image bytes."""
    buf = io.BytesIO()
    Image.fromarray(img_array).save(buf, format=fmt)
    return buf.getvalue()


def bytes_to_image(raw: bytes) -> np.ndarray:
    """Decode compressed image bytes back to numpy (H,W,3) uint8."""
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Basic text cleaning: strip, lowercase, collapse whitespace."""
    import re
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def make_training_text(product: Dict) -> str:
    """Combine product fields into a single string for embedding training."""
    return f"{product['name']}. {product['description']}"


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_catalog(
    products: Optional[List[Dict]] = None,
    output_dir: str = "data/raw",
    seed: int = 42,
) -> List[Dict]:
    """Generate the synthetic product catalog and save images to disk.

    Returns a list of records ready to be loaded into a Ray Dataset.
    """
    if products is None:
        products = PRODUCTS

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for i, p in enumerate(products):
        img_array = get_product_image(p)
        img_bytes = image_to_bytes(img_array)

        record = {
            "product_id": f"P{i:04d}",
            "name": p["name"],
            "category": p["category"],
            "description": p["description"],
            "price": p["price"],
            "image_bytes": img_bytes,
            "training_text": make_training_text(p),
        }
        records.append(record)

    print(f"Generated {len(records)} products in '{output_dir}'")
    return records


def expand_catalog(
    base_products: Optional[List[Dict]] = None,
    target_size: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """Expand the base catalog to *target_size* by generating product variants.

    Variant products share images with their base product via ``_base_name``,
    but have distinct names and descriptions — giving the embedding model
    diverse training signal without requiring additional real photos.
    """
    if base_products is None:
        base_products = PRODUCTS
    if not base_products:
        raise ValueError("base_products cannot be empty")

    rng = random.Random(seed)
    expanded = [dict(p) for p in base_products]  # start with originals

    i = 0
    while len(expanded) < target_size:
        base = base_products[i % len(base_products)]
        adj = rng.choice(_VARIANT_ADJECTIVES)
        suffix = rng.choice(_VARIANT_SUFFIXES[base["category"]])
        expanded.append({
            "name": f"{adj} {base['name']}",
            "category": base["category"],
            "price": round(base["price"] * rng.uniform(0.75, 1.4), 2),
            "description": base["description"].rstrip(".") + ". " + suffix,
            "_base_name": base["name"],
        })
        i += 1

    return expanded[:target_size]


# ---------------------------------------------------------------------------
# Preprocessing helpers (used inside Ray Data map_batches)
# ---------------------------------------------------------------------------

def preprocess_image_batch(batch: Dict) -> Dict:
    """Normalize image bytes to float32 tensor bytes (224x224x3, range [0,1]).

    Suitable as a Ray Data map_batches function.
    """
    processed = []
    for raw in batch["image_bytes"]:
        img = np.array(
            Image.open(io.BytesIO(raw)).convert("RGB").resize(IMAGE_SIZE)
        ).astype(np.float32) / 255.0
        # Store as bytes (float32 little-endian) to keep Parquet schema simple
        processed.append(img.tobytes())
    batch["image_tensor_bytes"] = processed
    return batch


def preprocess_text_batch(batch: Dict) -> Dict:
    """Clean training text. Tokenization happens inside the trainer."""
    batch["text_clean"] = [clean_text(t) for t in batch["training_text"]]
    return batch


def decode_image_tensor(raw: bytes) -> np.ndarray:
    """Inverse of preprocess_image_batch: bytes -> float32 (224,224,3)."""
    return np.frombuffer(raw, dtype=np.float32).reshape(*IMAGE_SIZE, 3)


# ---------------------------------------------------------------------------
# Ray helpers
# ---------------------------------------------------------------------------

def init_ray() -> None:
    """Initialize Ray with reduced logging (idempotent)."""
    import logging
    import ray
    import ray.data

    for name in ["ray", "ray.data", "ray.train", "ray.tune", "ray.serve",
                 "ray._private", "ray.runtime_env"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        logging_level=logging.WARNING,
    )
    ray.data.DataContext.get_current().enable_progress_bars = True


# ---------------------------------------------------------------------------
# Notebook convenience helpers
# ---------------------------------------------------------------------------

def attach_clean_text(records: List[Dict]) -> List[Dict]:
    """Add a `text_clean` field to each record (in-place), derived from
    `training_text`. Same cleaning rules as :func:`clean_text`.
    """
    for r in records:
        r["text_clean"] = clean_text(r["training_text"])
    return records


def sample_per_category(
    records: List[Dict],
    n_per_category: int,
    seed: int = 42,
) -> List[Dict]:
    """Return a class-balanced sample — up to *n_per_category* items per
    `category` — which gives contrastive training enough positives for every
    category even when the dataset is large.
    """
    from collections import defaultdict

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        buckets[r["category"]].append(r)

    rng = random.Random(seed)
    sampled: List[Dict] = []
    for bucket in buckets.values():
        k = min(n_per_category, len(bucket))
        sampled.extend(rng.sample(bucket, k))
    return sampled


def resolve_artifact_paths(here: Optional[str] = None) -> Dict[str, str]:
    """Pick where to read/write models + embeddings.

    On Anyscale clusters we prefer `/mnt/cluster_storage` so every node sees
    the same files; on a laptop we fall back to a local `models/` folder.
    Returns a dict with keys: model_dir, embeddings_path, metadata_path,
    train_result_dir.
    """
    here = here or os.path.abspath(".")
    shared = "/mnt/cluster_storage"
    use_shared = os.path.isdir(shared)
    base = shared if use_shared else os.path.join(here, "models")

    def _p(shared_name: str, local_name: str) -> str:
        return (
            os.path.join(shared, shared_name)
            if use_shared
            else os.path.join(here, "models", local_name)
        )

    return {
        "model_dir":        _p("ecomm_embedding_model",    "embedding_model"),
        "embeddings_path":  _p("ecomm_product_embeddings.npy", "product_embeddings.npy"),
        "metadata_path":    _p("ecomm_product_metadata.json",  "product_metadata.json"),
        "train_result_dir": _p("ecomm_ray_train_results", "ray_train_results"),
    }


def encode_image_base64(image_array: np.ndarray, fmt: str = "JPEG") -> str:
    """Encode a (H,W,3) numpy image to a base64 ASCII string — the payload
    shape our `/recommend` endpoint expects.
    """
    import base64
    return base64.b64encode(image_to_bytes(image_array, fmt=fmt)).decode()


def post_recommend(
    image_array: np.ndarray,
    url: str = "http://localhost:8000/recommend",
) -> Dict:
    """POST an image to the recommendation endpoint and return the JSON body."""
    import requests

    payload = {"image_base64": encode_image_base64(image_array)}
    return requests.post(url, json=payload).json()
