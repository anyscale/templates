"""Preprocessing script for Stable Diffusion v2 pre-training.

The script performs the following steps:
1. Load images and text captions from a remote storage system using the read_data function.
2. Transform images and text captions using the SDTransformer class.
3. Encode images and text captions into latent spaces using the SDLatentEncoder class.
"""

import gc
import io
import logging
import math
import time
from pathlib import Path
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
import ray.data
import torch
import torchvision  # type: ignore
import typer
from diffusers.models import AutoencoderKL
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer  # type: ignore

### Logging ###
logger = logging.getLogger("ray.data")

### Types ###
ResolutionDtype = Literal[256, 512]

### Constants ###
CAPTION_LATENTS_KEY = "caption_latents"
CAPTION_KEY = "caption"
IMAGE_LATENTS_256_KEY = "latents_256_bytes"
IMAGE_LATENTS_512_KEY = "latents_512_bytes"


########################
# Step 1: Data Loading #
########################

def read_data(
    input_uri: str,
    caption_col: str,
    caption_dtype: str,
    height_col: str,
    height_dtype: str,
    width_col: str,
    width_dtype: str,
    image_col: str,
    image_dtype: str,
    image_hash_col: str,
    image_hash_dtype: str,
    concurrency: Optional[int] = None,
    limit: Optional[int] = None,
) -> ray.data.Dataset:
    """Construct a ray data dataset from a parquet dataset."""
    schema = pa.schema(
        [
            pa.field(image_hash_col, getattr(pa, image_hash_dtype)()),
            pa.field(caption_col, getattr(pa, caption_dtype)()),
            pa.field(height_col, getattr(pa, height_dtype)()),
            pa.field(width_col, getattr(pa, width_dtype)()),
            pa.field(image_col, getattr(pa, image_dtype)()),
        ]
    )

    ds = ray.data.read_parquet(
        input_uri,
        schema=schema,
        concurrency=concurrency,
    )
    if limit:
        ds = ds.limit(limit)
    return ds


##########################
# Step 2: Transformation #
##########################

#### Utils ####
class LargestCenterSquare:
    """Largest center square crop for images."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        """Crop the largest center square from an image."""
        # First, resize the image such that the smallest side is self.size while preserving aspect ratio.
        img = torchvision.transforms.functional.resize(
            img=img,
            size=self.size,
        )

        # Then take a center crop to a square.
        w, h = img.size
        c_top = (h - self.size) // 2
        c_left = (w - self.size) // 2
        img = torchvision.transforms.functional.crop(
            img=img,
            top=c_top,
            left=c_left,
            height=self.size,
            width=self.size,
        )
        return img


#### Transformer ####
class SDTransformer:
    """Image and text transforms."""

    def __init__(
        self,
        resolution: int,
        model_name: str = "stabilityai/stable-diffusion-2-base",
    ) -> None:
        # Image transforms.
        self.resolution = resolution
        normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.crop = LargestCenterSquare(resolution)
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), normalize]
        )
        # Text tokenizer.
        self.text_tokenizer = CLIPTokenizer.from_pretrained(
            model_name, subfolder="tokenizer"
        )

    def image_transform(self, image: Image.Image) -> np.ndarray:
        """Transform image to a square-sized normalized tensor."""
        image = self.crop(image)
        out = self.transforms(image)
        return convert_tensor_to_array(out)

    def text_tokenize(self, text: str) -> np.ndarray:
        """Tokenize text using the CLIP tokenizer into a fixed-length sequence."""
        return self.text_tokenizer(
            text,
            padding="max_length",
            max_length=self.text_tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )["input_ids"][0]

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Transform images and text captions."""
        final_batch: dict[str, list] = {
            "hash": [],
            "caption_ids": [],
            f"image_{self.resolution}": [],
        }
        logger.info("Running image and text transforms...")
        for hash_, jpg, width, height, caption in zip(
            batch["hash"],
            batch["jpg"],
            batch["width"],
            batch["height"],
            batch["caption"],
        ):
            if jpg == b"":
                logger.info(f"Skipping hash {hash_} due to empty image.")
                continue
            if math.isnan(width) or width == 0:
                logger.info(f"Skipping hash {hash_} due to invalid width.")
                continue
            if math.isnan(height) or height == 0:
                logger.info(f"Skipping hash {hash_} due to invalid height.")
                continue

            try:
                image = Image.open(io.BytesIO(jpg))
            except IOError:
                logger.info(
                    f"Skipping hash {hash_} due to invalid image.", exc_info=False
                )
                continue

            if image.mode != "RGB":
                logger.info(f"Converting image to RGB.")
                image = image.convert("RGB")

            image_arr = self.image_transform(image)
            caption_ids = self.text_tokenize(caption)

            final_batch["hash"].append(hash_)
            final_batch[f"image_{self.resolution}"].append(image_arr)
            final_batch["caption_ids"].append(caption_ids)

        logger.info("Finished executing image and text transforms.")
        return {k: np.array(v) for k, v in final_batch.items()}


####################
# Step 3: Encoding #
####################

#### Utils ####
def convert_tensor_to_array(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """Convert a torch tensor to a numpy array."""
    array = tensor.detach().cpu().numpy()
    return array.astype(dtype)


def supports_float16(device: torch.device) -> bool:
    """Check if the device supports float16."""
    if device == torch.device("cuda"):
        properties = torch.cuda.get_device_properties(device)
        return properties.major >= 7  # Volta or newer
    else:
        return False


def resolve_device() -> torch.device:
    """Resolve to the first available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#### Encoder ####
class SDLatentEncoder:
    """Latent encoder to encode images and text."""

    def __init__(
        self,
        resolution: int = 256,
        device: Optional[str] = None,
        model_name: str = "stabilityai/stable-diffusion-2-base",
    ) -> None:
        self.device = torch.device(device) if device else resolve_device()
        self.resolution = resolution

        # Image and text encoders.
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae",
            torch_dtype=(
                torch.float16 if supports_float16(self.device) else torch.float32
            ),
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_name,
            subfolder="text_encoder",
            torch_dtype=(
                torch.float16 if supports_float16(self.device) else torch.float32
            ),
        )

        # Move the encoders to device.
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        logger.info("Initialized SDLatentEncoder.")
        logger.info(f"Device: {self.device}")
        logger.info(f"Resolution: {self.resolution}")

    @property
    def image_latents_key(self) -> str:
        """Return the key for image latents based on resolution."""
        if self.resolution == 256:
            return IMAGE_LATENTS_256_KEY
        elif self.resolution == 512:
            return IMAGE_LATENTS_512_KEY
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")

    def encode_images(self, images: np.ndarray) -> np.ndarray:
        """Encode images into a latent space."""
        logger.info(f"Encoding images with shape: {images.shape}")
        input_images = torch.tensor(images, device=self.device)

        if supports_float16(self.device):
            latent_dist = self.vae.encode(input_images.half())["latent_dist"]
        else:
            logger.info("Running VAE.encode")
            latent_dist = self.vae.encode(input_images)["latent_dist"]

        image_latents = latent_dist.sample() * 0.18215
        logger.info(f"Image latents shape: {image_latents.shape}")
        return convert_tensor_to_array(image_latents)

    def encode_text(self, caption_ids: np.ndarray) -> np.ndarray:
        """Encode text captions into a latent space."""
        logger.info(f"Encoding text with shape: {caption_ids.shape}")
        caption_ids_tensor = torch.tensor(caption_ids, device=self.device)
        caption_latents_tensor = self.text_encoder(caption_ids_tensor)[0]
        logger.info(f"Caption latents shape: {caption_latents_tensor.shape}")
        return convert_tensor_to_array(caption_latents_tensor)

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Encode images and text captions."""
        with torch.no_grad():
            # Step 1: Encode images.
            input_images = batch[f"image_{self.resolution}"]
            image_latents = self.encode_images(input_images)
            # Shape = [batch_size, 4, 32, 32]
            batch[self.image_latents_key] = image_latents

            del batch[f"image_{self.resolution}"]
            gc.collect()

            # Step 2: Encode captions.
            caption_ids = batch["caption_ids"]
            # Shape = [batch_size, 77, 1024]
            batch[CAPTION_LATENTS_KEY] = self.encode_text(caption_ids)

            del batch["caption_ids"]
            gc.collect()

        return batch


#######################
# Visualizing Outputs #
#######################

def visualize_image(image_bytes: bytes, caption: str) -> None:
    """Visualize an image with a caption."""
    img = Image.open(io.BytesIO(image_bytes))
    plt.imshow(img)
    plt.axis("off")
    x_bounds, y_bounds = plt.xlim(), plt.ylim()
    # Add a caption at the bottom of the image.
    plt.text(
        x_bounds[0],
        y_bounds[0] + 50,
        f"Caption: {caption}",
        color="black",
        backgroundcolor="white",
        fontsize=12,
    )
    plt.show()


def visualize_image_latents(image_latents: np.ndarray) -> None:
    """Visualize image latents."""
    nchannels = image_latents.shape[0]
    fig, axes = plt.subplots(1, nchannels, figsize=(10, 10))
    # Set figure title as image latents.
    for idx, ax in enumerate(axes):
        ax.imshow(image_latents[idx], cmap="gray")
        ax.axis("off")
    # Add a title in the center top of the image.
    fig.suptitle("Image Latents", fontsize=16, x=0.5, y=0.625)
    fig.tight_layout()
    plt.show()


def visualize_text_embeddings(text_embeddings: np.ndarray) -> None:
    """Visualize text embeddings."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(text_embeddings, cmap="gray")
    ax.set_title("Caption Embeddings (Text Latents)")
    plt.show()


def visualize_input_and_output(
    resolution: ResolutionDtype,
    df_output: pd.DataFrame,
    df_input: pd.DataFrame,
) -> None:
    """Visualize input and output data."""
    for _, row in df_output.iterrows():
        hash_ = row["hash"]
        print("Input:")
        print(100 * "-")
        input_record = df_input[df_input["hash"] == hash_].squeeze()
        visualize_image(input_record["jpg"], input_record["caption"])
        print("Output:")
        print(100 * "-")
        visualize_image_latents(row[f"latents_{resolution}_bytes"])
        visualize_text_embeddings(row["caption_latents"])
        print("\n\n")


########################################
# Putting all data processing together #
########################################

def get_laion_streaming_dataset(
    # Reader.
    input_uri: str,
    caption_col: str,
    caption_dtype: str,
    height_col: str,
    height_dtype: str,
    width_col: str,
    width_dtype: str,
    image_col: str,
    image_dtype: str,
    image_hash_col: str,
    image_hash_dtype: str,
    resolution: ResolutionDtype,
    # Transformer.
    num_transformers: int,
    num_cpus_per_transformer: int,
    # Encoder.
    num_encoders: int,
    num_gpus_per_encoder: Optional[int],
    encoder_accelerator_type: Optional[str],
    encoder_batch_size: Optional[int],
    limit: Optional[int],
) -> ray.data.Dataset:
    """Stream data through a transformer and encoder pipeline."""
    ds = read_data(
        input_uri=input_uri,
        caption_col=caption_col,
        caption_dtype=caption_dtype,
        height_col=height_col,
        height_dtype=height_dtype,
        width_col=width_col,
        width_dtype=width_dtype,
        image_col=image_col,
        image_dtype=image_dtype,
        image_hash_col=image_hash_col,
        image_hash_dtype=image_hash_dtype,
        limit=limit,
        concurrency=num_transformers,
    )

    ds = ds.map_batches(
        SDTransformer,
        fn_constructor_kwargs={"resolution": resolution},
        concurrency=num_transformers,
        num_cpus=num_cpus_per_transformer,
    )

    ds = ds.map_batches(
        SDLatentEncoder,
        fn_constructor_kwargs={"resolution": resolution},
        num_gpus=num_gpus_per_encoder,
        num_cpus=1 if not num_gpus_per_encoder else 0,
        batch_size=encoder_batch_size,
        concurrency=num_encoders,
        accelerator_type=encoder_accelerator_type,
    )
    return ds


########################################################################
# CLI to run the main function, storing the output, and visualizing it #
########################################################################

app = typer.Typer()

@app.command()
def process(
    input_uri: str = (
        "https://anyscale-materials.s3.us-west-2.amazonaws.com/stable-diffusion/laion-art-sample.parquet"
    ),
    visualize_output: bool = True,
    output_path: Optional[str] = None,
    resolution: int = 512,
    num_rows_per_output_file: int = 1000,
    num_transformers: int = 1,
    num_cpus_per_transformer: int = 1,
    num_encoders: int = 1,
    num_gpus_per_encoder: Optional[int] = None,
    encoder_accelerator_type: Optional[str] = None,
    encoder_batch_size: Optional[int] = None,
    limit: int = 5,
):
    """Preprocess images and text for Stable Diffusion v2 model pre-training."""
    start_t = time.time()
    ds = get_laion_streaming_dataset(
        # Read.
        input_uri=input_uri,
        caption_col="caption",
        caption_dtype="string",
        height_col="height",
        height_dtype="float64",
        width_col="width",
        width_dtype="float64",
        image_col="jpg",
        image_dtype="binary",
        image_hash_col="hash",
        image_hash_dtype="int64",
        resolution=cast(ResolutionDtype, resolution),
        # Transform.
        num_transformers=num_transformers,
        num_cpus_per_transformer=num_cpus_per_transformer,
        # Encode.
        num_encoders=num_encoders,
        num_gpus_per_encoder=num_gpus_per_encoder,
        encoder_accelerator_type=encoder_accelerator_type,
        encoder_batch_size=encoder_batch_size,
        limit=limit,
    )

    if output_path:
        output_uri = output_path
    else:
        output_dir = Path("/mnt/cluster_storage/tmp/") / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_uri = str(output_dir.resolve())

    ds.write_parquet(
        output_uri,
        num_rows_per_file=num_rows_per_output_file,
    )
    total_t = time.time() - start_t
    print(f"Finished in {total_t} seconds.")

    if visualize_output:
        print("Visualizing input raw data compared to output processed data.")
        ray.data.set_progress_bars(False)
        ctx = ray.data.DataContext.get_current()
        ctx.execution_options.verbose_progress = False
        df_output = ray.data.read_parquet(output_uri).limit(2).to_pandas()
        df_input = (
            ray.data.read_parquet(
                input_uri,
            )
            .limit(2)
            .to_pandas()
        )
        visualize_input_and_output(
            resolution=cast(ResolutionDtype, resolution),
            df_output=df_output,
            df_input=df_input,
        )


if __name__ == "__main__":
    app()
