"""
Ray Data pipeline for Pokemon image-caption dataset.

Loads svjack/pokemon-blip-captions-en-zh (833 images, ungated) and preprocesses
for Stable Diffusion LoRA fine-tuning: resize, normalize, tokenize.
"""
import io

import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_SIZE = 512
TOKENIZER_NAME = "openai/clip-vit-large-patch14"

_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # → [-1, 1]
])


def load_pokemon_dataset():
    """Load the Pokemon BLIP captions dataset via Ray Data."""
    import ray.data
    from datasets import load_dataset

    # svjack variant is ungated (lambdalabs original is gated on HF)
    hf_ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
    ds = ray.data.from_huggingface(hf_ds)
    return ds


def preprocess_batch(batch: dict) -> dict:
    """CPU preprocessing: decode images, resize, normalize, tokenize captions.

    Input columns:  image (dict with 'bytes' key), en_text (str)
    Output columns: pixel_values (float32 [C,H,W]), input_ids (int64 [77])
    """
    from transformers import CLIPTokenizer

    tokenizer = CLIPTokenizer.from_pretrained(TOKENIZER_NAME)

    pixel_values = []
    for img in batch["image"]:
        # Ray Data serializes HF Image columns as {"bytes": b"...", "path": "..."}
        if isinstance(img, dict):
            img = Image.open(io.BytesIO(img["bytes"]))
        elif isinstance(img, bytes):
            img = Image.open(io.BytesIO(img))
        elif not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        pixel_values.append(_transform(img).numpy())

    # svjack dataset uses 'en_text' column name
    captions = list(batch.get("en_text", batch.get("text", [])))

    tokens = tokenizer(
        captions,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    return {
        "pixel_values": np.stack(pixel_values).astype(np.float32),
        "input_ids": tokens["input_ids"].astype(np.int64),
    }
