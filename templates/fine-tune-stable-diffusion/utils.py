import argparse
import os
import tempfile
from typing import List, Optional

from filelock import FileLock
from PIL import Image as PILImage
import pyarrow.fs

import ray
import ray.train


def upload_to_cloud(local_path: str, cloud_uri: str):
    pyarrow.fs.copy_files(source=local_path, destination=cloud_uri, use_threads=False)


def download_from_cloud(cloud_uri: str, local_path: str):
    os.makedirs(local_path, exist_ok=True)
    with FileLock(local_path + ".lock"):
        pyarrow.fs.copy_files(source=cloud_uri, destination=local_path)


@ray.remote(num_gpus=1, accelerator_type="A10G")
def generate(
    prompts: List[str],
    args: argparse.Namespace,
    model_checkpoint_path: Optional[str] = None,
) -> List[PILImage.Image]:
    """Load the SDXL pipeline and generate images as a GPU worker task.

    Arguments:
        prompts: A list of string prompts to generate images for.
        args: Parsed arguments object from the command line args.
            These are needed if the model revision/floating point precision are configured.
        model_checkpoint_path: Path to the LoRA fine-tuned model weights.
            If None, then this will generate images with the base SDXL model.

    Returns:
        images: List of generated images as PIL Image objects.
    """
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_path = (
        args.pretrained_model_name_or_path
        if args.pretrained_vae_model_name_or_path is None
        else args.pretrained_vae_model_name_or_path
    )
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder="vae" if args.pretrained_vae_model_name_or_path is None else None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )

    if model_checkpoint_path is not None:
        with tempfile.TemporaryDirectory(prefix="/mnt/local_storage/") as local_checkpoint_dir:
            # Download model checkpoint to this temporary local directory.
            download_from_cloud(model_checkpoint_path, local_checkpoint_dir)

            # Load fine-tuned LoRA weights
            pipeline.load_lora_weights(local_checkpoint_dir)

    pipeline = pipeline.to("cuda")
    images = pipeline(prompt=prompts).images
    return images
