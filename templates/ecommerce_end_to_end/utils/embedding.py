"""Persistence helpers for the batch embedding stage.

The actor + ``map_batches`` call that actually computes embeddings lives
inline in ``README.ipynb`` (Stage 3) so readers can see the canonical Ray
Data pattern instead of a one-liner import. This module keeps just the
post-processing step that splits actor output into a dense ``.npy`` matrix
and a small JSON sidecar — boilerplate that doesn't add to the demo.
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


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
