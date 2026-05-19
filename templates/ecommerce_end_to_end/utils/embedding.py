"""Batch embedding utilities for the product catalog."""

import json
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import ray


def embed_catalog_in_parallel(
    ds: "ray.data.Dataset",
    model_dir: str,
    pool_size: int = 2,
    batch_size: int = 8,
) -> List[Dict]:
    """Run the fine-tuned model across `ds` with an actor pool.

    Each actor loads the model exactly once and then streams batches through
    it, so cost amortises across the whole catalog instead of per-row.
    Scaling up is a matter of bumping `pool_size`.
    """
    assert Path(model_dir).exists() and any(Path(model_dir).iterdir()), (
        f"Model not found at {model_dir} — run the fine-tuning stage first"
    )
    return ds.map_batches(
        ProductEmbedder,
        fn_constructor_kwargs={"model_dir": model_dir},
        batch_size=batch_size,
        num_cpus=1,
        compute=ray.data.ActorPoolStrategy(size=pool_size),
        batch_format="numpy",
    ).take_all()


def save_embeddings_and_metadata(
    rows: List[Dict],
    embeddings_path: str,
    metadata_path: str,
) -> tuple:
    """Split the embedder output into a dense vector matrix (saved as .npy)
    and a small JSON sidecar with just the fields the serving layer needs.
    Keeping them separate avoids loading product text into memory at query time.
    """
    embeddings = np.array([r["embedding"] for r in rows])
    metadata = [
        {"product_id": r["product_id"], "name": r["name"], "category": r["category"]}
        for r in rows
    ]
    Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return embeddings, metadata


class ProductEmbedder:
    """Ray Data actor: loads the fine-tuned model once, encodes batches of text."""

    def __init__(self, model_dir: str):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_dir)

    def __call__(self, batch: dict) -> dict:
        texts = batch["text_clean"].tolist()
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return {
            "product_id": batch["product_id"],
            "name":       batch["name"],
            "category":   batch["category"],
            "embedding":  list(embs),
        }


def run_batch_embedding(
    preprocessed_dir: str,
    model_dir: str,
    embeddings_path: str,
    metadata_path: str,
    batch_size: int = 8,
) -> tuple:
    """Run Stage 3: embed entire catalog with fine-tuned model.

    Returns:
        (embeddings, metadata) — numpy array of shape (N, D) and list of dicts.
    """
    t0 = time.time()
    print("=" * 60)
    print("STAGE 3 — RAY DATA: BATCH EMBEDDING")
    print("=" * 60)

    ds = (
        ray.data.read_parquet(preprocessed_dir)
        .select_columns(["product_id", "name", "category", "text_clean"])
    )
    print(f"\nRows to embed: {ds.count()}")
    print("Running batch embedding …")

    rows = (
        ds.map_batches(
            ProductEmbedder,
            fn_constructor_kwargs={"model_dir": model_dir},
            batch_size=batch_size,
            num_cpus=1,
            compute=ray.data.ActorPoolStrategy(size=2),
            batch_format="numpy",
        )
        .take_all()
    )

    embeddings = np.array([r["embedding"] for r in rows])
    metadata = [
        {"product_id": r["product_id"], "name": r["name"], "category": r["category"]}
        for r in rows
    ]

    np.save(embeddings_path, embeddings)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBatch embedding complete! ({time.time()-t0:.1f}s)")
    print(f"  Embeddings : {embeddings_path}  shape={embeddings.shape}")
    print(f"  Metadata   : {metadata_path}  ({len(metadata)} products)")
    print("=" * 60)
    return embeddings, metadata
