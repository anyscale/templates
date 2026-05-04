"""
CPU preprocessing stage for the embedding pipeline.
Runs on CPU worker nodes via Ray Data map_batches.
"""
import re
from typing import Optional

import pandas as pd


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_field(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return _normalize_whitespace(_strip_html(str(value)))


def _combine_text_fields(title: str, description: str, category: str) -> str:
    """Combine product fields into a single text representation for embedding."""
    parts = [f"Category: {category}", f"Title: {title}", f"Description: {description}"]
    combined = " | ".join(parts)
    # Truncate to ~512 tokens (approx 2048 chars) to match model max input
    return combined[:2048]


def preprocess_product(batch: dict) -> dict:
    """
    CPU-bound preprocessing stage.

    Input batch fields: product_id, title, description, category, brand, price
    Output adds: combined_text
    Drops rows with missing required fields.
    """
    titles = batch["title"]
    descriptions = batch["description"]
    categories = batch["category"]
    product_ids = batch["product_id"]
    brands = batch["brand"]
    prices = batch["price"]

    output = {
        "product_id": [],
        "title": [],
        "description": [],
        "category": [],
        "brand": [],
        "price": [],
        "combined_text": [],
    }
    dropped = 0

    for i in range(len(titles)):
        title = _clean_field(titles[i])
        description = _clean_field(descriptions[i])
        category = _clean_field(categories[i])

        if not title or not description or not category:
            dropped += 1
            continue

        output["product_id"].append(product_ids[i])
        output["title"].append(title)
        output["description"].append(description)
        output["category"].append(category)
        output["brand"].append(brands[i])
        output["price"].append(prices[i])
        output["combined_text"].append(_combine_text_fields(title, description, category))

    if dropped > 0:
        print(f"  [preprocess] Dropped {dropped} malformed rows in this batch")

    return output
