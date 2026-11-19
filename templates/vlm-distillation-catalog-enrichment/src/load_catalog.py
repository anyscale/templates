"""
Catalog loaders for the VLM batch enrichment demo.

Primary source:  McAuley-Lab/Amazon-Reviews-2023 (image URLs, served from Amazon CDN)
Backup source:   shunk031/amazon-berkeley-objects (images bundled in the HF dataset)

Both write a parquet with a uniform schema so the rest of the pipeline is
source-agnostic:

    product_id : str
    title      : str
    description: str
    image_url  : str   (HTTP for Amazon Reviews; "abo://<key>" for ABO local cache)
    source     : str   ("amazon-reviews-2023" | "amazon-berkeley-objects")
"""
import os
import random
from typing import Optional

import pandas as pd


AMAZON_REVIEWS_DATASET = "McAuley-Lab/Amazon-Reviews-2023"
ABO_DATASET = "shunk031/amazon-berkeley-objects"

# Categories mirror Amazon Reviews 2023 metadata splits
AMAZON_CATEGORIES = [
    "Electronics",
    "Home_and_Kitchen",
    "Clothing_Shoes_and_Jewelry",
    "Beauty_and_Personal_Care",
    "Sports_and_Outdoors",
]


def _extract_image_url(images_field) -> Optional[str]:
    """Pick the first 'large' image URL (or hi_res/thumb fallback) from Amazon's image struct.

    HF parquet stores `images` as a struct of parallel lists,
    e.g. ``{"large": [url1, url2], "hi_res": [...], "thumb": [...], "variant": [...]}``,
    not as a list of per-image dicts. We take the first URL of the best
    available size.
    """
    if not images_field or not isinstance(images_field, dict):
        return None
    for key in ("large", "hi_res", "thumb"):
        urls = images_field.get(key)
        if urls is not None and len(urls) > 0:
            return urls[0]
    return None


def _has_title_and_image(row: dict) -> bool:
    """Drop rows missing either a title or any usable image URL."""
    title = row.get("title")
    if not (title and title.strip()):
        return False
    return _extract_image_url(row.get("images")) is not None


def _coerce_description(desc_field) -> str:
    if isinstance(desc_field, list):
        return " ".join(str(x) for x in desc_field if x).strip()
    if isinstance(desc_field, str):
        return desc_field.strip()
    return ""


def _normalize_amazon_row_to_image(row: dict) -> dict:
    """One catalog row per product, using the single best image URL (use with `.map`)."""
    return {
        "product_id": row.get("parent_asin") or row.get("asin") or "",
        "title": row["title"].strip()[:512],
        "description": _coerce_description(row.get("description"))[:1024],
        "image_url": _extract_image_url(row["images"]),
        "source": "amazon-reviews-2023",
    }


def _normalize_amazon_row_to_images(row: dict, max_per_product: int = 8) -> list[dict]:
    """One catalog row per *image* — explodes a product into N rows (use with `.flat_map`).

    Takes 'large' URLs (falling back to hi_res / thumb), capped at
    ``max_per_product``. Each output row carries an ``image_idx`` so
    ``(product_id, image_idx)`` is a stable row key.
    """
    images = row.get("images") or {}
    if not isinstance(images, dict):
        return []
    urls = (
        images.get("large")
        or images.get("hi_res")
        or images.get("thumb")
        or []
    )[:max_per_product]
    if not urls:
        return []

    product_id = row.get("parent_asin") or row.get("asin") or ""
    title = row["title"].strip()[:512]
    description = _coerce_description(row.get("description"))[:1024]

    return [
        {
            "product_id": product_id,
            "image_idx": i,
            "title": title,
            "description": description,
            "image_url": url,
            "source": "amazon-reviews-2023",
        }
        for i, url in enumerate(urls)
    ]


def load_amazon_reviews_2023(
    category: str,
    n_rows: int,
    seed: int = 42,
):
    """
    Lazy Ray Dataset of N normalized product rows from Amazon Reviews 2023.

    Reads parquet directly from the HF Hub via ray.data.read_parquet —
    `datasets.load_dataset` no longer supports the script-based loader
    this dataset originally shipped (Amazon-Reviews-2023.py).

    Output columns: ``product_id, title, description, image_url, source``.
    Image bytes are NOT fetched here — the demo's naive vs heterogeneous
    pipelines diverge on *where* the fetch happens (GPU actor vs dedicated
    CPU stage), so the loader leaves URLs as URLs.
    """
    import ray
    from huggingface_hub import HfFileSystem

    if category not in AMAZON_CATEGORIES:
        raise ValueError(f"Unknown category {category!r}; pick one of {AMAZON_CATEGORIES}")

    hf_path = f"hf://datasets/{AMAZON_REVIEWS_DATASET}/raw_meta_{category}"
    print(f"[load] Building pipeline from {hf_path}...")

    return (
        ray.data.read_parquet(
            hf_path,
            file_extensions=["parquet"],
            filesystem=HfFileSystem(),
        )
        .limit(n_rows)
        .filter(_has_title_and_image)
        .map(_normalize_amazon_row_to_image)
        .random_shuffle(seed=seed)
    )


def load_amazon_berkeley_objects(
    n_rows: int,
    output_path: str,
    seed: int = 42,
    image_cache_dir: str = "/mnt/cluster_storage/vlm-distillation-catalog-enrichment/abo-images",
) -> str:
    """
    Backup loader: Amazon Berkeley Objects.

    ABO has images bundled in the HF dataset (PIL.Image), so we materialize
    them to a local cache and reference them via 'abo://<filename>' URLs
    that src/preprocess.py knows how to resolve.

    Use this if Amazon Reviews 2023 image URLs are unreachable from the
    cluster (e.g., demo wifi blocking Amazon CDN, or rate-limiting).
    """
    from datasets import load_dataset
    from PIL import Image

    print(f"[load] Loading {ABO_DATASET} (images bundled, this may take a few minutes)...")
    ds = load_dataset(ABO_DATASET, "all", split="train", trust_remote_code=True)

    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_rows * 2]  # buffer for skipped rows

    os.makedirs(image_cache_dir, exist_ok=True)
    records = []
    for idx in indices:
        row = ds[idx]
        item_id = row.get("item_id") or f"abo-{idx}"
        # ABO 'item_name' is a list of language-tagged dicts; take the en_US value.
        name_list = row.get("item_name") or []
        title = ""
        for entry in name_list:
            if isinstance(entry, dict) and entry.get("language_tag", "").startswith("en"):
                title = entry.get("value", "")
                break
        if not title:
            continue

        image: Optional[Image.Image] = row.get("image") or row.get("main_image")
        if image is None:
            continue

        cache_path = os.path.join(image_cache_dir, f"{item_id}.jpg")
        if not os.path.exists(cache_path):
            image.convert("RGB").save(cache_path, "JPEG", quality=85)

        records.append({
            "product_id": item_id,
            "title": title[:512],
            "description": (row.get("product_description") or "")[:1024],
            "image_url": f"abo://{item_id}.jpg",  # preprocess resolves to image_cache_dir
            "source": "amazon-berkeley-objects",
        })
        if len(records) >= n_rows:
            break

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"  [load] Wrote {len(df):,} products → {output_path}")
    print(f"  [load] Image cache: {image_cache_dir}")
    return output_path


def load_catalog(
    source: str = "amazon-reviews-2023",
    category: str = "Electronics",
    n_rows: int = 10_000,
    output_path: str = "/mnt/cluster_storage/vlm-distillation-catalog-enrichment/vlm-demo/catalog.parquet",
):
    """
    Unified loader. `source` is one of:
      - "amazon-reviews-2023"  (default, returns a lazy Ray Dataset of image URLs)
      - "amazon-berkeley-objects"  (backup, materializes images, returns parquet path)
    """
    if source == "amazon-reviews-2023":
        return load_amazon_reviews_2023(category, n_rows)
    elif source == "amazon-berkeley-objects":
        return load_amazon_berkeley_objects(n_rows, output_path)
    else:
        raise ValueError(f"Unknown source {source!r}")


def shard_catalog_to_parquet(
    category: str,
    n_rows: int,
    num_shards: int,
    output_dir: str,
    seed: int = 42,
) -> int:
    """Build a normalized + sharded catalog at ``output_dir/shard_NNNN.parquet``.

    Each shard becomes one unit of resumable work for the sharded pipeline
    (see ``scripts/run_pipeline_sharded.py`` and ``src.pipeline.run_with_checkpoints``).

    Returns the number of shards actually written (chunks with zero rows are skipped).
    """
    import numpy as np

    print(f"[shard] Loading {n_rows:,} rows from {AMAZON_REVIEWS_DATASET}/{category}...")
    df = load_amazon_reviews_2023(category, n_rows, seed=seed).to_pandas()
    print(f"[shard] Got {len(df):,} rows after filtering — splitting into {num_shards} shards")

    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for shard_id, chunk in enumerate(np.array_split(df, num_shards)):
        if len(chunk) == 0:
            continue
        path = os.path.join(output_dir, f"shard_{shard_id:04d}.parquet")
        chunk.to_parquet(path, index=False)
        written += 1
    print(f"[shard] Wrote {written} shard files → {output_dir}")
    return written
