"""
ProductRanker Ray Serve deployment.

Single deployment that handles the full recommendation pipeline:
  1. Encode query      → all-MiniLM-L6-v2 (PyTorch, GPU)
  2. Retrieve          → FAISS IndexFlatIP (in-process, CPU)
  3. Re-rank           → cross-encoder/ms-marco-MiniLM-L-6-v2 (PyTorch, GPU)

Both models are loaded once at replica __init__. Cold start happens per
replica, not per request — this is the key Ray Serve lifecycle to highlight.
"""
import time

import faiss
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from ray import serve
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.models import RecommendRequest, RecommendResponse, ProductResult

CATALOG_PATH = "/mnt/cluster_storage/ecommerce-demo/serving/product_catalog.parquet"
INDEX_PATH = "/mnt/cluster_storage/ecommerce-demo/serving/product_index.faiss"
ENCODER_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_app = FastAPI(title="E-Commerce Product Ranker", version="1.0")


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
)
@serve.ingress(_app)
class ProductRanker:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # FAISS index + product catalog — loaded once at replica init
        self.index = faiss.read_index(INDEX_PATH)
        self.products = pd.read_parquet(CATALOG_PATH).to_dict("records")

        # Both PyTorch models onto GPU
        self.encoder = SentenceTransformer(ENCODER_MODEL, device=device)
        self.reranker = CrossEncoder(RERANKER_MODEL, device=device)

        print(
            f"[ProductRanker] Ready on {device} | "
            f"{self.index.ntotal:,} products indexed"
        )

    @_app.get("/health")
    async def health(self):
        return {"status": "ok", "index_size": self.index.ntotal}

    @_app.post("/recommend")
    async def recommend(self, request: RecommendRequest) -> RecommendResponse:
        stages = {}
        wall_start = time.time()

        # Step 1: encode query → 384-dim normalized vector
        t = time.time()
        query_vec = self.encoder.encode(
            [request.query], normalize_embeddings=True
        )[0].astype(np.float32)
        stages["encode_ms"] = round((time.time() - t) * 1000, 1)

        # Step 2: FAISS retrieval → top-100 candidates
        t = time.time()
        scores, indices = self.index.search(query_vec.reshape(1, -1), 100)
        candidates = [
            dict(self.products[i])
            for i in indices[0]
            if i >= 0
        ]
        stages["retrieve_ms"] = round((time.time() - t) * 1000, 1)

        # Step 3: cross-encoder re-ranking → top num_results
        t = time.time()
        pairs = [
            (request.query, f"{c['title']} {c.get('description', '')[:300]}")
            for c in candidates
        ]
        rerank_scores = self.reranker.predict(pairs)
        for candidate, score in zip(candidates, rerank_scores):
            candidate["score"] = float(score)
        ranked = sorted(candidates, key=lambda x: x["score"], reverse=True)
        stages["rerank_ms"] = round((time.time() - t) * 1000, 1)

        # Apply optional filters
        if request.max_price is not None:
            ranked = [r for r in ranked if r.get("price", 0) <= request.max_price]
        if request.category_filter:
            ranked = [
                r for r in ranked
                if r.get("category", "").lower() == request.category_filter.lower()
            ]

        results = [
            ProductResult(
                product_id=r["product_id"],
                title=r["title"],
                category=r["category"],
                price=r["price"],
                relevance_score=round(r["score"], 4),
            )
            for r in ranked[: request.num_results]
        ]

        return RecommendResponse(
            query=request.query,
            results=results,
            latency_ms=round((time.time() - wall_start) * 1000, 1),
            stages=stages,
        )
