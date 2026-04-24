"""
Offline index builder: generate 50K products → embed → build FAISS index.

Run once before starting the service:
    python -m src.build_index

Writes to /mnt/cluster_storage/ecommerce-demo/serving/:
    product_catalog.parquet  — product metadata (no embeddings)
    product_index.faiss      — FAISS IndexFlatIP over 384-dim embeddings
"""
import importlib.util
import os
import sys

import faiss
import numpy as np
import pandas as pd
import torch

# Reuse the Faker-based catalog generator from the batch-embeddings sibling demo.
# Imported via spec_from_file_location to avoid colliding with the local `src`
# package namespace when running as `python -m src.build_index`.
_BATCH_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "batch-embeddings")
)
_gen_data_path = os.path.join(_BATCH_ROOT, "src", "generate_data.py")
_spec = importlib.util.spec_from_file_location("batch_generate_data", _gen_data_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
generate_catalog = _mod.generate_catalog

CATALOG_PATH = "/mnt/cluster_storage/ecommerce-demo/serving/product_catalog.parquet"
INDEX_PATH = "/mnt/cluster_storage/ecommerce-demo/serving/product_index.faiss"
NUM_PRODUCTS = 50_000
EMBED_BATCH_SIZE = 512
MODEL_NAME = "all-MiniLM-L6-v2"


def _make_text(row: dict) -> str:
    desc = str(row.get("description", ""))[:500]
    return f"Category: {row['category']} | Title: {row['title']} | Description: {desc}"


def build_index(
    catalog_path: str = CATALOG_PATH,
    index_path: str = INDEX_PATH,
    num_products: int = NUM_PRODUCTS,
    force: bool = False,
) -> None:
    """
    Three-stage offline pipeline:
      1. Generate synthetic product catalog with Faker
      2. Embed all products with all-MiniLM-L6-v2
      3. Build a FAISS IndexFlatIP and write to disk

    Skips entirely if both output files already exist (use force=True to rebuild).
    """
    if not force and os.path.exists(catalog_path) and os.path.exists(index_path):
        idx = faiss.read_index(index_path)
        print(f"Index already exists — {idx.ntotal:,} vectors at {index_path}")
        print(f"Catalog: {catalog_path}")
        print("Skipping build. Pass force=True to rebuild.")
        return

    os.makedirs(os.path.dirname(catalog_path), exist_ok=True)

    # ── Stage 1: Generate catalog ──────────────────────────────────────────
    print(f"\n[1/3] Generating {num_products:,} synthetic products...")
    if os.path.exists(catalog_path):
        print(f"  Found existing catalog at {catalog_path}, skipping generation.")
    else:
        generate_catalog(num_products, catalog_path)

    df = pd.read_parquet(catalog_path)
    print(f"  Loaded {len(df):,} products")

    # ── Stage 2: Embed ─────────────────────────────────────────────────────
    print(f"\n[2/3] Embedding with {MODEL_NAME}...")
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    texts = [_make_text(row) for row in df.to_dict("records")]
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # ── Stage 3: FAISS index ───────────────────────────────────────────────
    print(f"\n[3/3] Building FAISS IndexFlatIP (dim={embeddings.shape[1]})...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"  Index: {index.ntotal:,} vectors → {index_path}")
    print(f"  Catalog: {len(df):,} products → {catalog_path}")
    print("\nIndex build complete. Ready to start the service.")


if __name__ == "__main__":
    build_index()
