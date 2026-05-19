"""
Quick test client for the Ray Serve recommendation endpoint.

Usage:
    # Start the service first:
    #   serve run serve_app:app
    #
    # Then in another terminal:
    python client.py
"""

import base64
import io
import json
import sys

import requests
from PIL import Image

SERVE_URL = "http://localhost:8000"


def encode_image(path_or_pil) -> str:
    """Return base64-encoded JPEG string from a path or PIL Image."""
    if isinstance(path_or_pil, str):
        with open(path_or_pil, "rb") as f:
            raw = f.read()
    else:
        buf = io.BytesIO()
        path_or_pil.save(buf, format="JPEG")
        raw = buf.getvalue()
    return base64.b64encode(raw).decode("utf-8")


def test_health():
    print("=== Health check ===")
    resp = requests.get(f"{SERVE_URL}/health", timeout=5)
    print(f"Status: {resp.status_code}  Body: {resp.json()}")
    print()


def test_recommend_demo():
    """Send a real product image and call /recommend."""
    from utils import get_product_image, PRODUCTS

    print("=== /recommend — demo product image ===")

    # Use the first product (Wireless Headphones) as query image
    product = PRODUCTS[0]
    img_arr = get_product_image(product)
    img_pil = Image.fromarray(img_arr)

    payload = {"image_base64": encode_image(img_pil)}

    try:
        resp = requests.post(
            f"{SERVE_URL}/recommend",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        print(
            "ERROR: Could not connect. "
            "Make sure `serve run serve_app:app` is running first."
        )
        sys.exit(1)

    print(f"Caption: {result['caption']!r}")
    print(f"\nTop {len(result['recommendations'])} recommendations:")
    for r in result["recommendations"]:
        print(
            f"  {r['rank']}. [{r['category']:18s}] {r['name']:35s}  "
            f"sim={r['similarity']:.3f}"
        )
    print()


if __name__ == "__main__":
    test_health()
    test_recommend_demo()
    print("✅ All tests passed!")
