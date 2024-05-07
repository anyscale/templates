"""End-to-end online preprocessing and model training script for Stable Diffusion v2.

The script performs the following steps:
1. Load images and text captions from a remote storage system using the read_data function.
2. Transform images and text captions using the SDTransformer class.
3. Encode images and text captions into latent spaces using the SDLatentEncoder class.
4. Build a Stable Diffusion model using the StableDiffusion class.
5. Run the scalable training procedure with Ray Train using the train entry point.
"""

import gc
import io
import logging
import math
import os
import re
import shutil
import tempfile
from contextlib import nullcontext
from functools import partial
from typing import Literal, Optional, cast, ContextManager, Mapping, Union, Any

import lightning.pytorch as pl  # type: ignore
import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.fs
import ray.train
import torch
import torch.nn.functional as F
import torchvision  # type: ignore
import typer
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.models import AutoencoderKL
from lightning.pytorch import seed_everything  # type: ignore
from lightning.pytorch.callbacks import LearningRateMonitor  # type: ignore
from lightning.pytorch.utilities.types import OptimizerLRScheduler  # type: ignore
from PIL import Image
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayFSDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer, get_device
from s3fs import S3FileSystem  # type: ignore
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import CLIPTextModel, CLIPTokenizer, PretrainedConfig, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

### Type Definitions ###
ResolutionDtype = Literal[256, 512]

### Constants ###
CAPTION_LATENTS_KEY = "caption_latents"
IMAGE_LATENTS_256_KEY = "latents_256_bytes"
IMAGE_LATENTS_512_KEY = "latents_512_bytes"


############################################
#### Step 1: Data Loading ####
############################################
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


############################################
#### Step 2: Transformation ####
############################################


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


############################################
#### Step 3: Encoding ####
############################################


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
    """Resolve to the first available device: CUDA, MPS, or CPU."""
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


##########################################################
### Putting all data processing together ###
##########################################################
def get_training_columns(resolution: ResolutionDtype) -> list[str]:
    key_list = [CAPTION_LATENTS_KEY]
    if resolution == 256:
        key_list.append(IMAGE_LATENTS_256_KEY)
    elif resolution == 512:
        key_list.append(IMAGE_LATENTS_512_KEY)
    return key_list


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
    training_batch_size: int,
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

    cols = set(get_training_columns(resolution))

    def select_training_columns(batch):
        return {k: v for k, v in batch.items() if k in cols}

    ds = ds.map_batches(
        select_training_columns,
        zero_copy_batch=True,
        batch_size=training_batch_size,
    )

    return ds


#############################################
# Step 4: Build Stable Diffusion model.
#############################################

### Small model configuration ###
small_unet_model_config = {
    "_class_name": "UNet2DConditionModel",
    "_diffusers_version": "0.2.2",
    "act_fn": "silu",
    "attention_head_dim": 8,
    "block_out_channels": [160, 320, 640, 640],
    "center_input_sample": False,
    "cross_attention_dim": 1024,
    "down_block_types": [
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    ],
    "downsample_padding": 1,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "in_channels": 4,
    "layers_per_block": 2,
    "mid_block_scale_factor": 1,
    "norm_eps": 1e-05,
    "norm_num_groups": 32,
    "out_channels": 4,
    "sample_size": 64,
    "up_block_types": [
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ],
}


### Model definition ###
class StableDiffusion(pl.LightningModule):
    """Stable Diffusion U-Net model."""

    def __init__(
        self,
        lr: float,
        resolution: ResolutionDtype,
        weight_decay: float,
        num_warmup_steps: int,
        init_from_pretrained: bool,
        model_name: str,
        use_small_unet: bool,
        use_xformers: bool,
        fsdp: bool,
    ) -> None:
        self.lr = lr
        self.resolution = resolution
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        super().__init__()
        self.save_hyperparameters()

        # Initialize U-Net.
        if init_from_pretrained:
            self.unet = UNet2DConditionModel.from_pretrained(
                model_name, subfolder="unet"
            )
        else:
            if use_small_unet:
                model_config = small_unet_model_config
            else:
                model_config = PretrainedConfig.get_config_dict(
                    model_name, subfolder="unet"
                )[0]
            self.unet = UNet2DConditionModel(**model_config)

        if use_xformers:
            print("Enable xformers memeff attention.")
            self.unet.enable_xformers_memory_efficient_attention()

        if fsdp:
            self.unet = torch.compile(self.unet)

        # Define the training noise schedulers.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )
        # Setup loss function.
        self.loss_fn = F.mse_loss
        self.current_training_steps = 0

    @property
    def image_latents_key(self) -> str:
        """Return the key for image latents based on resolution."""
        if self.resolution == 256:
            return IMAGE_LATENTS_256_KEY
        elif self.resolution == 512:
            return IMAGE_LATENTS_512_KEY
        else:
            raise ValueError(f"Unsupported resolution: {self.resolution}")

    def on_fit_start(self) -> None:
        """Move cumprod tensor to GPU in advance to avoid data movement on each step."""
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(
            get_device()
        )

    def _sample_timesteps(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.randint(
            0, len(self.noise_scheduler), (latents.shape[0],), device=latents.device
        )

    def forward(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model."""
        # Extract inputs.
        latents = batch[self.image_latents_key]
        conditioning = batch[CAPTION_LATENTS_KEY]
        # Sample the diffusion timesteps.
        timesteps = self._sample_timesteps(latents)
        # Add noise to the inputs (forward diffusion).
        noise = torch.randn_like(latents)
        noised_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        # Forward through the model.
        outputs = self.unet(noised_latents, timesteps, conditioning)["sample"]
        return outputs, noise

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step of the model."""
        outputs, targets = self.forward(batch)
        loss = self.loss_fn(outputs, targets)
        self.log(
            "train/loss_mse", loss.item(), prog_bar=False, on_step=True, sync_dist=False
        )
        self.current_training_steps += 1
        return loss

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step of the model."""
        outputs, targets = self.forward(batch)
        loss = self.loss_fn(outputs, targets)
        self.log(
            "validation/loss_mse",
            loss.item(),
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),  # type: ignore [union-attr]
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        # Set a large training step here to keep lr constant after warm-up.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=100000000000,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


##############################################
# Step 5: Run the scalable training procedure.
##############################################

### Utils ###


### Callbacks ###
def strategy_context(
    fsdp: bool = False, model: Optional[torch.nn.Module] = None
) -> ContextManager:
    """Context manager to summon full params for FSDP."""
    if fsdp and model is not None:
        return FSDP.summon_full_params(model, writeback=False, recurse=False)
    else:
        return nullcontext()


class RayTrainReportCallback(pl.callbacks.Callback):
    def __init__(
        self,
        every_n_train_steps: int = 5000,
        checkpoint_sharding_strategy: str = "full",
        fsdp: bool = False,
    ) -> None:
        super().__init__()
        self.every_n_train_steps = every_n_train_steps
        self.checkpoint_sharding_strategy = checkpoint_sharding_strategy
        self.fsdp = fsdp

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[Union[torch.Tensor, Mapping[str, Any]]],
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Report metrics and save checkpoints."""
        step = pl_module.current_training_steps
        if (step + 1) % self.every_n_train_steps != 0:
            return

        # Create a local temporary directory to save the checkpoint.
        temp_checkpoint_dir = os.path.join(
            tempfile.gettempdir(),
            ray.train.get_context().get_trial_name(),
            f"step={step}",
        )
        os.makedirs(temp_checkpoint_dir, exist_ok=True)

        # Fetch metrics.
        callback_metrics = trainer.callback_metrics
        metrics = {k: v.item() for k, v in callback_metrics.items()}

        # (Optional) Add customized metrics.
        metrics["epoch"] = trainer.current_epoch
        metrics["step"] = trainer.global_step

        # Save checkpoint to local.
        ckpt_path = os.path.join(temp_checkpoint_dir, "checkpoint.ckpt")

        with strategy_context(fsdp=self.fsdp, model=trainer.model):
            trainer.save_checkpoint(ckpt_path, weights_only=False)

        # Create a Checkpoint object.
        if self.checkpoint_sharding_strategy == "full":
            checkpoint = (
                Checkpoint.from_directory(temp_checkpoint_dir)
                if ray.train.get_context().get_world_rank() == 0
                else None
            )
        elif self.checkpoint_sharding_strategy == "sharded":
            checkpoint = (
                Checkpoint.from_directory(temp_checkpoint_dir)
                if ray.train.get_context().get_local_rank() == 0
                else None
            )
        else:
            raise ValueError(f"Not supported {self.checkpoint_sharding_strategy=}")

        # Report to train session.
        ray.train.report(metrics=metrics, checkpoint=checkpoint)

        # Add a barrier to ensure all workers finished reporting here.
        torch.distributed.barrier()

        # Clean up the checkpoint, because it's already copied to storage.
        if ray.train.get_context().get_local_rank() == 0:
            shutil.rmtree(temp_checkpoint_dir)


### Collate function ###
def move_to_device_collate_fn(
    batch: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move the batch to the device."""
    for k, v in batch.items():
        batch[k] = torch.tensor(v).to(device)  # type: ignore [assignment]
    return cast(dict[str, torch.Tensor], batch)


### Training function per worker ###
def train_func(config: dict) -> None:
    """Training function for Stable Diffusion model."""
    seed = config["seed"]
    seed_everything(seed)
    trial_name = ray.train.get_context().get_trial_name()

    # Prepare Ray datasets.
    collate_fn = partial(move_to_device_collate_fn, device=ray.train.torch.get_device())

    train_ds = ray.train.get_dataset_shard("train")
    train_dataloader = train_ds.iter_torch_batches(
        batch_size=config["batch_size_per_worker"],
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_batches=config["prefetch_batches"],
    )

    validation_ds = ray.train.get_dataset_shard("validation")
    validation_dataloader = validation_ds.iter_torch_batches(
        batch_size=config["batch_size_per_worker"],
        collate_fn=collate_fn,
        drop_last=True,
        prefetch_batches=config["prefetch_batches"],
    )

    # Initialize Stable Diffusion model.
    torch.set_float32_matmul_precision("high")

    if config["fsdp"]:
        strategy = RayFSDPStrategy(
            sharding_strategy=config["sharding_policy"],
            state_dict_type=config["checkpoint_sharding_strategy"],
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            use_orig_params=True,
        )
    else:
        strategy = RayDDPStrategy()

    # Initialize Lightning callbacks.
    ray_train_reporter = RayTrainReportCallback(
        config["checkpoint_every_n_steps"], config["checkpoint_sharding_strategy"]
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [ray_train_reporter, lr_monitor]

    # Initialize Lightning Trainer.
    lightning_log_dir = os.path.join(
        tempfile.gettempdir(), "lightning_logs", trial_name
    )
    os.makedirs(lightning_log_dir, exist_ok=True)
    trainer = pl.Trainer(
        max_steps=config["max_steps"],
        val_check_interval=config["val_check_interval"],
        check_val_every_n_epoch=None,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        accelerator="gpu",
        devices="auto",
        precision="bf16-mixed",
        strategy=strategy,
        plugins=[RayLightningEnvironment()],
        callbacks=callbacks,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        default_root_dir=lightning_log_dir,
    )

    checkpoint = ray.train.get_checkpoint()
    model_kwargs = dict(
        init_from_pretrained=config["init_from_pretrained"],
        model_name=config["model_name"],
        use_xformers=config["use_xformers"],
        fsdp=config["fsdp"],
        lr=config["lr"],
        use_small_unet=config["use_small_unet"],
        resolution=config["resolution"],
        weight_decay=config["weight_decay"],
        num_warmup_steps=config["num_warmup_steps"],
    )
    if checkpoint:
        # Continue training from a previous checkpoint.
        with checkpoint.as_directory() as ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "checkpoint.ckpt")

            if config["resume_from_checkpoint"]:
                # Case 1: Start a new run.
                # Only restore the model weights, training starts from step 0.
                model = StableDiffusion.load_from_checkpoint(
                    ckpt_path, map_location=torch.device("cpu"), **model_kwargs
                )

                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader,
                )
            else:
                # Case 2: Restore from an interrupted or crashed run.
                # Restore both the model weights and the trainer states (optimizer, steps, callbacks).
                model = StableDiffusion(**model_kwargs)

                trainer.fit(
                    model,
                    train_dataloaders=train_dataloader,
                    val_dataloaders=validation_dataloader,
                    ckpt_path=ckpt_path,
                )
    else:
        # Start a new run from scratch.
        model = StableDiffusion(**model_kwargs)

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=validation_dataloader,
        )


#############################################
# Main CLI: Entry point for training script.
#############################################
app = typer.Typer()

artifact_storage = os.environ["ANYSCALE_ARTIFACT_STORAGE"]
user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
anyscale_storage_path = f"{artifact_storage}/{user_name}"
anyscale_storage_path = anyscale_storage_path.replace("s3://", "")


@app.command()
def train(
    experiment_name: str = "sd-demo",
    seed: int = 2024,
    init_from_pretrained: bool = False,
    model_name: str = "stabilityai/stable-diffusion-2-base",
    use_xformers: bool = False,
    batch_size_per_worker: int = 1,
    max_steps: int = 20,
    fsdp: bool = False,
    sharding_policy: Optional[str] = None,
    checkpoint_sharding_strategy: str = "full",
    resume_from_checkpoint: Optional[str] = None,
    restore_from_uri: Optional[str] = None,
    max_failures: int = 3,
    checkpoint_every_n_steps: int = 10,
    val_check_interval: int = 40,
    accumulate_grad_batches: int = 1,
    storage_path: str = anyscale_storage_path,
    prefetch_batches: int = 1,
    resolution: int = 256,
    num_training_workers: int = 1,
    num_transformers: int = 1,
    num_cpus_per_transformer: int = 1,
    num_encoders: int = 1,
    num_gpus_per_encoder: Optional[int] = None,
    encoder_accelerator_type: Optional[str] = None,
    encoder_batch_size: Optional[int] = None,
    limit: Optional[int] = None,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    num_warmup_steps: int = 4,
    use_small_unet: bool = True,
    accelerator_type: Optional[str] = None,
    train_data_uri: str = "s3://anyscale-materials/stable-diffusion/laion_art_sample_train.parquet",
    validation_data_uri: str = "s3://anyscale-materials/stable-diffusion/laion_art_sample_valid.parquet",
):
    """Train a Stable Diffusion model."""
    ray.data.set_progress_bars(False)
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = False

    train_ds = get_laion_streaming_dataset(
        # read
        input_uri=train_data_uri,
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
        # transform
        num_transformers=num_transformers,
        num_cpus_per_transformer=num_cpus_per_transformer,
        # encode
        num_encoders=num_encoders,
        num_gpus_per_encoder=num_gpus_per_encoder,
        encoder_accelerator_type=encoder_accelerator_type,
        encoder_batch_size=encoder_batch_size,
        limit=limit,
        training_batch_size=batch_size_per_worker,
    )

    validation_ds = get_laion_streaming_dataset(
        # read
        input_uri=validation_data_uri,
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
        # transform
        num_transformers=num_transformers,
        num_cpus_per_transformer=num_cpus_per_transformer,
        # encode
        num_encoders=num_encoders,
        num_gpus_per_encoder=num_gpus_per_encoder,
        encoder_accelerator_type=encoder_accelerator_type,
        encoder_batch_size=encoder_batch_size,
        limit=limit,
        training_batch_size=batch_size_per_worker,
    )

    ray_datasets = {"train": train_ds, "validation": validation_ds}

    if restore_from_uri:
        print(f"Restore experiment from {restore_from_uri}...")
        assert TorchTrainer.can_restore(restore_from_uri)
        trainer = TorchTrainer.restore(
            restore_from_uri,
            datasets=ray_datasets,
            train_loop_per_worker=train_func,
        )
    else:
        checkpoint = None
        if resume_from_checkpoint:
            checkpoint = Checkpoint(resume_from_checkpoint)

        s3_fs = S3FileSystem()
        fs = pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))

        trainer = TorchTrainer(
            train_func,
            train_loop_config={
                "seed": seed,
                "lr": lr,
                "weight_decay": weight_decay,
                "num_warmup_steps": num_warmup_steps,
                "init_from_pretrained": init_from_pretrained,
                "model_name": model_name,
                "use_xformers": use_xformers,
                "use_small_unet": use_small_unet,
                "batch_size_per_worker": batch_size_per_worker,
                "max_steps": max_steps,
                "fsdp": fsdp,
                "sharding_policy": sharding_policy,
                "checkpoint_sharding_strategy": checkpoint_sharding_strategy,
                "resume_from_checkpoint": resume_from_checkpoint,
                "checkpoint_every_n_steps": checkpoint_every_n_steps,
                "val_check_interval": val_check_interval,
                "accumulate_grad_batches": accumulate_grad_batches,
                "prefetch_batches": prefetch_batches,
                "resolution": cast(ResolutionDtype, resolution),
            },
            scaling_config=ScalingConfig(
                num_workers=num_training_workers,
                use_gpu=True,
                accelerator_type=accelerator_type,
            ),
            run_config=RunConfig(
                name=experiment_name,
                storage_path=storage_path,
                storage_filesystem=fs,
                failure_config=FailureConfig(max_failures=max_failures),
            ),
            datasets=ray_datasets,  # type: ignore [arg-type]
            resume_from_checkpoint=checkpoint,
        )
    trainer.fit()

    # Show the produced model checkpoints under storage path.
    fs = S3FileSystem()
    paths = fs.glob(f"{storage_path }/**/checkpoint.ckpt")
    print("Produced Model Checkpoints:")
    print("===========================")
    for p in paths:
        print(p)


if __name__ == "__main__":
    app()
