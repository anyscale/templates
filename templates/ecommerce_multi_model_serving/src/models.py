"""
Pydantic request/response schemas for the recommendation API.
"""
from typing import List, Optional

from pydantic import BaseModel


class RecommendRequest(BaseModel):
    query: str
    num_results: int = 10
    max_price: Optional[float] = None
    category_filter: Optional[str] = None


class ProductResult(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    relevance_score: float


class RecommendResponse(BaseModel):
    query: str
    results: List[ProductResult]
    latency_ms: float
    stages: dict  # encode_ms, retrieve_ms, rerank_ms
