"""Preprocessing script for Stable Diffusion pre-training.

The script performs the following steps:
1. Load images and text captions from a remote storage system using the read_data function.
2. Transform images and text captions using the SDTransformer class.
3. Encode images and text captions into latent spaces using the SDLatentEncoder class.

For an detailed explanation of the below code, check out our
[guide here](https://www.anyscale.com/blog/processing-2-billion-images-for-stable-diffusion-model-training-definitive-guides-with-ray-series).
"""

import io
import logging
import time
from pathlib import Path
from typing import Literal, Optional, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore
import pyarrow as pa  # type: ignore
import ray.data
import typer
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sd_data_transforms import (
    SDTransformer,
    SDLatentEncoder,
    CAPTION_LATENTS_KEY,
    IMAGE_LATENTS_256_KEY,
    IMAGE_LATENTS_512_KEY,
)

### Logging ###
logger = logging.getLogger("ray.data")

### Type Definitions ###
ResolutionDtype = Literal[256, 512]


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
    """Construct a Ray Data dataset from a Parquet dataset."""
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



########################################################################
# CLI to run the main function, storing the output, and visualizing it #
##################################################
# Main CLI: Entry point for preprocessing script #
##################################################

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
        ctx = ray.data.DataContext.get_current()
        ctx.enable_progress_bars = False
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
