"""
One-time script: generate base SD 1.5 images (without LoRA) for side-by-side comparison.
Run once, commit the PNGs to assets/base_model_samples/.

Usage:
    python scripts/pre_generate_base_images.py
"""
import os
import torch
from diffusers import StableDiffusionPipeline

PROMPTS = [
    "a pokemon with blue fire",
    "a cute pokemon in a forest",
    "a legendary water pokemon",
    "a dragon type pokemon",
]

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "base_model_samples",
)

SEED = 42
MODEL_ID = "runwayml/stable-diffusion-v1-5"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")

    for prompt in PROMPTS:
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        img = pipe(prompt, num_inference_steps=30, generator=generator).images[0]

        safe_name = prompt.replace(" ", "_").replace("'", "")[:50]
        path = os.path.join(OUTPUT_DIR, f"base_{safe_name}.png")
        img.save(path)
        print(f"  Saved: {path}")

    print(f"\n{len(PROMPTS)} base images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
