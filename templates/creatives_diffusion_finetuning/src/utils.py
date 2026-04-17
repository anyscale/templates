"""
Utility functions for the diffusion fine-tuning demo.
"""
import os
import time
from contextlib import contextmanager
from typing import List

from PIL import Image


@contextmanager
def timer(label: str):
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{label}] {elapsed:.1f}s")


def show_image_grid(images: List[Image.Image], titles: List[str], cols: int = 4):
    """Display a grid of PIL images with titles in the notebook."""
    import matplotlib.pyplot as plt

    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for ax in axes:
        ax.axis("off")

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, wrap=True)

    plt.tight_layout()
    plt.show()


def show_comparison(
    base_dir: str,
    finetuned_images: List[Image.Image],
    prompts: List[str],
):
    """Show base model images (from assets/) alongside fine-tuned images."""
    import matplotlib.pyplot as plt

    n = len(prompts)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    if n == 1:
        axes = [axes]

    for i, prompt in enumerate(prompts):
        # Base model image (pre-generated, in assets/)
        safe_name = prompt.replace(" ", "_").replace("'", "")[:50]
        base_path = os.path.join(base_dir, f"base_{safe_name}.png")
        if os.path.exists(base_path):
            base_img = Image.open(base_path)
        else:
            # Placeholder if base image not found
            base_img = Image.new("RGB", (512, 512), (200, 200, 200))

        axes[i][0].imshow(base_img)
        axes[i][0].set_title("Base SD 1.5", fontsize=11)
        axes[i][0].axis("off")

        axes[i][1].imshow(finetuned_images[i])
        axes[i][1].set_title("Fine-tuned (LoRA)", fontsize=11)
        axes[i][1].axis("off")

        # Prompt label on the left
        axes[i][0].set_ylabel(f'"{prompt}"', fontsize=9, rotation=0,
                              labelpad=120, va="center")

    plt.suptitle("Base Model vs. LoRA Fine-Tuned", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()
