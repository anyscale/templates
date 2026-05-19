"""
E-Commerce Recommendation System — Ray Serve Application
=========================================================

Stage 3 of 3:  Online recommendation endpoint

Pipeline (multi-model composition)
-----------------------------------
  POST /recommend  {"image_base64": "<base64-encoded-image>"}
       │
       ▼
  ImageToText            ← Salesforce/blip-image-captioning-base
       │  caption text
       ▼
  ProductRecommender     ← fine-tuned all-MiniLM-L6-v2 + cosine index
       │
       ▼
  JSON response          {"caption": "...", "recommendations": [...]}

Run locally (development)
--------------------------
    serve run serve_app:app
    # or:
    python serve_app.py

Run as an Anyscale Service
---------------------------
    anyscale service deploy -f service.yaml

Test
----
    python client.py

Ray version: 2.x  (Ray ≥ 2.20)
Base image:  anyscale/ray:2.47.1-slim-py312   (CPU-only)
See https://docs.anyscale.com/reference/base-images for the latest images.

References
----------
- Ray Serve Model Composition: https://docs.ray.io/en/latest/serve/model_composition.html
- Ray Serve Key Concepts:      https://docs.ray.io/en/latest/serve/key-concepts.html
- Anyscale Services:           https://docs.anyscale.com/services/
"""

import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BLIP_MODEL = "Salesforce/blip-image-captioning-base"
_HERE = Path(__file__).parent.resolve()

# Model resolution order (first match wins):
#  1. Env var EMBEDDING_MODEL_DIR (explicit override)
#  2. Cluster storage fine-tuned model (written by notebook when /mnt/cluster_storage exists)
#  3. Local models/embedding_model (pushed or generated locally)
#  4. HuggingFace model ID — pre-cached in container image (Dockerfile bakes it in)
_CLUSTER_MODEL = Path("/mnt/cluster_storage") / "ecomm_embedding_model"
_LOCAL_MODEL = _HERE / "models/embedding_model"
_default_model_dir = (
    str(_CLUSTER_MODEL) if _CLUSTER_MODEL.exists()
    else str(_LOCAL_MODEL) if _LOCAL_MODEL.exists()
    else "sentence-transformers/all-MiniLM-L6-v2"
)

_CLUSTER_EMBEDDINGS = Path("/mnt/cluster_storage") / "ecomm_product_embeddings.npy"
_LOCAL_EMBEDDINGS = _HERE / "models/product_embeddings.npy"
_default_embeddings = (
    str(_CLUSTER_EMBEDDINGS) if _CLUSTER_EMBEDDINGS.exists()
    else str(_LOCAL_EMBEDDINGS)
)

_CLUSTER_METADATA = Path("/mnt/cluster_storage") / "ecomm_product_metadata.json"
_LOCAL_METADATA = _HERE / "models/product_metadata.json"
_default_metadata = (
    str(_CLUSTER_METADATA) if _CLUSTER_METADATA.exists()
    else str(_LOCAL_METADATA)
)

EMBEDDING_MODEL_DIR = os.environ.get("EMBEDDING_MODEL_DIR", _default_model_dir)
EMBEDDINGS_PATH = os.environ.get("EMBEDDINGS_PATH", _default_embeddings)
METADATA_PATH = os.environ.get("METADATA_PATH", _default_metadata)

TOP_K = int(os.environ.get("TOP_K", "5"))

# ---------------------------------------------------------------------------
# Pydantic schema
# ---------------------------------------------------------------------------


class RecommendRequest(BaseModel):
    image_base64: str  # standard base64-encoded image bytes


# ---------------------------------------------------------------------------
# Deployment 1: Image → Caption (BLIP)
# ---------------------------------------------------------------------------


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2},
    health_check_period_s=30,
    health_check_timeout_s=60,
)
class ImageToText:
    """Load BLIP and convert raw image bytes to a natural-language caption."""

    def __init__(self):
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ImageToText] Loading {BLIP_MODEL} on {self.device} …")

        self.processor = BlipProcessor.from_pretrained(BLIP_MODEL)
        self.model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
        self.model.to(self.device).eval()

        # Warm-up pass to avoid first-request latency spike
        from PIL import Image as PILImage

        dummy = PILImage.new("RGB", (224, 224), color=(128, 128, 128))
        inputs = self.processor(images=dummy, return_tensors="pt").to(self.device)
        import torch

        with torch.no_grad():
            self.model.generate(**inputs, max_new_tokens=30)

        print(f"[ImageToText] Ready.")

    def caption(self, image_bytes: bytes) -> str:
        """Generate a caption for the given raw image bytes."""
        import torch
        from PIL import Image as PILImage

        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=40,
                num_beams=3,
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    def check_health(self):
        pass  # model loaded in __init__; if it failed, the actor died


# ---------------------------------------------------------------------------
# Deployment 2: Text → Top-K products (sentence-transformer + cosine index)
# ---------------------------------------------------------------------------


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 2},
    health_check_period_s=30,
    health_check_timeout_s=60,
)
class ProductRecommender:
    """
    Embed a query text with the fine-tuned sentence-transformer, then return
    the top-K most similar products via cosine similarity over a pre-computed
    embedding matrix.

    Paths are accepted as constructor arguments so Ray Serve workers (which
    do not inherit the driver's ``os.environ`` from a notebook) still load the
    same model and index files as the process that called ``bind()``.
    """

    def __init__(
        self,
        embedding_model_dir: str | None = None,
        embeddings_path: str | None = None,
        metadata_path: str | None = None,
    ):
        from sentence_transformers import SentenceTransformer

        model_dir = (
            embedding_model_dir
            if embedding_model_dir is not None
            else EMBEDDING_MODEL_DIR
        )
        emb_path = (
            embeddings_path if embeddings_path is not None else EMBEDDINGS_PATH
        )
        meta_path = metadata_path if metadata_path is not None else METADATA_PATH

        print(f"[ProductRecommender] Loading embedding model from {model_dir} …")
        self.model = SentenceTransformer(model_dir)

        print(f"[ProductRecommender] Loading product index …")
        self.embeddings = np.load(emb_path).astype(np.float32)  # (N, D)
        # L2-normalise for fast cosine via dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        self.embeddings_norm = self.embeddings / norms

        with open(meta_path) as f:
            self.metadata = json.load(f)  # list of {product_id, name, category}

        print(f"[ProductRecommender] Ready. {len(self.metadata)} products indexed.")

    def recommend(self, query_text: str, top_k: int = TOP_K) -> list[dict]:
        """Return the top-K products most similar to query_text."""
        q_emb = self.model.encode([query_text], convert_to_numpy=True)[0].astype(
            np.float32
        )
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

        sims = self.embeddings_norm @ q_norm  # (N,)
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            m = self.metadata[idx]
            results.append(
                {
                    "rank": rank,
                    "product_id": m["product_id"],
                    "name": m["name"],
                    "category": m["category"],
                    "similarity": float(round(float(sims[idx]), 4)),
                }
            )
        return results

    def check_health(self):
        pass


# ---------------------------------------------------------------------------
# Deployment 3: HTTP ingress (orchestrator)
# ---------------------------------------------------------------------------

_fastapi = FastAPI(
    title="E-Commerce Product Recommender",
    description="Upload an image → get product recommendations",
    version="1.0.0",
)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 1},
)
@serve.ingress(_fastapi)
class RecommendationService:
    """
    HTTP ingress that chains ImageToText → ProductRecommender.

    Both sub-deployments are called via async DeploymentHandle so the
    orchstrator never blocks.
    """

    def __init__(
        self,
        image_to_text: DeploymentHandle,
        product_recommender: DeploymentHandle,
    ):
        self.image_to_text = image_to_text
        self.product_recommender = product_recommender
        print("[RecommendationService] Ready.")

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    @_fastapi.get("/health")
    def health(self) -> dict:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # POST /recommend
    # ------------------------------------------------------------------

    @_fastapi.post("/recommend")
    async def recommend(self, body: RecommendRequest) -> dict[str, Any]:
        """
        Accept a base64-encoded image, caption it with BLIP, then return
        the top-K most similar products.

        Request body:
            {"image_base64": "<base64 string>"}

        Response:
            {
                "caption": "a pair of wireless headphones",
                "recommendations": [
                    {"rank": 1, "product_id": "P0000", "name": "...",
                     "category": "Electronics", "similarity": 0.92},
                    ...
                ]
            }
        """
        # 1. Decode base64 → raw bytes
        try:
            image_bytes = base64.b64decode(body.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")

        # 2. Image → caption  (async remote call to ImageToText)
        caption: str = await self.image_to_text.caption.remote(image_bytes)

        # 3. Caption → recommendations  (async remote call to ProductRecommender)
        recommendations: list[dict] = await self.product_recommender.recommend.remote(
            caption, TOP_K
        )

        return {
            "caption": caption,
            "recommendations": recommendations,
        }


# ---------------------------------------------------------------------------
# Application binding
# ---------------------------------------------------------------------------

app = RecommendationService.bind(
    image_to_text=ImageToText.bind(),
    product_recommender=ProductRecommender.bind(
        embedding_model_dir=EMBEDDING_MODEL_DIR,
        embeddings_path=EMBEDDINGS_PATH,
        metadata_path=METADATA_PATH,
    ),
)


# ---------------------------------------------------------------------------
# Local development entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import ray

    ray.init(ignore_reinit_error=True)
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(app, blocking=True)
