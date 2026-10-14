"""
CPU image fetch + decode stage.

Runs on Ray Data CPU workers via map_batches. Pulls image bytes from HTTP,
decodes to PIL.Image, resizes to 384x384 (Qwen2.5-VL input sweet spot),
re-encodes as JPEG, and emits raw bytes ready for the VLM.

Mirrors the inline helper in `notebooks/demo_walkthrough.ipynb`.
"""
import io

from PIL import Image


TARGET_SIZE = (384, 384)


def fetch_and_decode(batch: dict) -> dict:
    """CPU stage: HTTP fetch → PIL decode → 384x384 JPEG bytes ready for vLLM.

    Input columns:  product_id, title, image_url, ...
    Output columns: product_id, title, image_url, image_bytes
                    (rows with failed fetch/decode are dropped)
    """
    import requests

    out = {k: [] for k in ("product_id", "title", "image_url", "image_bytes")}
    for i in range(len(batch["product_id"])):
        try:
            resp = requests.get(
                batch["image_url"][i],
                timeout=5.0,
                headers={"User-Agent": "vlm-distillation-catalog-enrichment/1.0"},
            )
            if resp.status_code != 200:
                continue
            img = Image.open(io.BytesIO(resp.content)).convert("RGB").resize(TARGET_SIZE)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=90)
        except Exception:
            continue
        out["product_id"].append(batch["product_id"][i])
        out["title"].append(batch["title"][i])
        out["image_url"].append(batch["image_url"][i])
        out["image_bytes"].append(buf.getvalue())
    return out
