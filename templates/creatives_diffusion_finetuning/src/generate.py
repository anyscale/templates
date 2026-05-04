"""
Image generation from a fine-tuned LoRA checkpoint.

Loads the base SD 1.5 model, applies LoRA weights, and generates
images for side-by-side comparison with the base model.
"""
from typing import List

import torch
from PIL import Image


def generate_images(
    checkpoint_path: str,
    prompts: List[str],
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_inference_steps: int = 30,
    seed: int = 42,
) -> List[Image.Image]:
    """Generate images from a fine-tuned LoRA checkpoint.

    Returns one image per prompt, all using the same seed for reproducibility.
    """
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")

    # Load LoRA weights from checkpoint
    pipe.unet = PeftModel.from_pretrained(pipe.unet, checkpoint_path, autocast_adapter_dtype=False)

    images = []
    for prompt in prompts:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        img = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).images[0]
        images.append(img)
        print(f"  Generated: \"{prompt}\"")

    return images
