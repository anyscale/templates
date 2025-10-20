"""Training script for Stable Diffusion model v2.

The script performs the following steps:
1. Load preprocessed data from S3 or ABFSS using the load_precomputed_dataset function.
2. Build a Stable Diffusion model using the StableDiffusion class.
3. Run the scalable training procedure with Ray Train using the train entry point.

Supports both S3 and Azure Blob File System (ABFSS) for data storage and checkpoints.

For ABFSS support:
1. Install required packages:
   pip install adlfs azure-identity

2. Authentication uses DefaultAzureCredential which automatically handles:
   - Managed Identity (recommended for AKS)
   - Azure CLI credentials
   - Environment variables
   - Other Azure credential sources

Example ABFSS URL: abfss://container@account.dfs.core.windows.net/path
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import nullcontext
from functools import partial
from typing import ContextManager, Literal, Optional, cast, Mapping, Any, Union

import lightning.pytorch as pl  # type: ignore
import numpy as np
import pyarrow.fs
import ray.train
import torch
import torch.nn.functional as F
import typer
from diffusers import DDPMScheduler, UNet2DConditionModel
from lightning.pytorch import seed_everything  # type: ignore
from lightning.pytorch.callbacks import LearningRateMonitor  # type: ignore
from lightning.pytorch.utilities.types import OptimizerLRScheduler  # type: ignore
from ray.train import Checkpoint, FailureConfig, RunConfig, ScalingConfig
from ray.train.lightning import RayDDPStrategy, RayFSDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer, get_device
from s3fs import S3FileSystem  # type: ignore
try:
    from adlfs import AzureBlobFileSystem  # type: ignore
    ADLFS_AVAILABLE = True
except ImportError:
    ADLFS_AVAILABLE = False

try:
    from azure.identity import DefaultAzureCredential  # type: ignore
    AZURE_IDENTITY_AVAILABLE = True
except ImportError:
    AZURE_IDENTITY_AVAILABLE = False
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PretrainedConfig, get_linear_schedule_with_warmup  # type: ignore

logger = logging.getLogger(__name__)

### Type definitions ###
ResolutionDtype = Literal[256, 512]

### Constants ###
CAPTION_LATENTS_KEY = "caption_latents"
IMAGE_LATENTS_256_KEY = "latents_256_bytes"
IMAGE_LATENTS_512_KEY = "latents_512_bytes"

##################################################
# Step 1: Load preprocessed data from S3 or ABFSS #
##################################################

### Import ABFSS utilities ###
from abfss_utils import (
    is_abfss_path,
    create_azure_filesystem,
    upload_to_abfss,
    get_local_storage_path,
)


def create_filesystem(storage_path: str) -> pyarrow.fs.FileSystem:
    """Create appropriate filesystem based on the storage path."""
    if is_abfss_path(storage_path):
        if not ADLFS_AVAILABLE:
            raise ImportError(
                "adlfs is required for ABFSS support. Install it with: pip install adlfs"
            )
        # Extract account name from ABFSS path
        # Format: abfss://container@account.dfs.core.windows.net/path
        import re
        match = re.match(r"abfss?://[^@]+@([^.]+)\.dfs\.core\.windows\.net", storage_path)
        if not match:
            raise ValueError(f"Invalid ABFSS path format: {storage_path}")

        account_name = match.group(1)
        azure_fs = create_azure_filesystem(account_name)
        return pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(azure_fs))
    else:
        # Default to S3
        s3_fs = S3FileSystem()
        return pyarrow.fs.PyFileSystem(pyarrow.fs.FSSpecHandler(s3_fs))


def get_training_columns(resolution: ResolutionDtype) -> list[str]:
    key_list = [CAPTION_LATENTS_KEY]
    if resolution == 256:
        key_list.append(IMAGE_LATENTS_256_KEY)
    elif resolution == 512:
        key_list.append(IMAGE_LATENTS_512_KEY)
    return key_list


def convert_precision(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    for k, v in batch.items():
        batch[k] = v.astype(np.float16)
    return batch


### Data loading ###
def load_precomputed_dataset(
    data_uri: str, num_data_loading_workers: int, resolution: ResolutionDtype
) -> ray.data.Dataset:
    """Load from offline precomputed datasets."""
    key_list = get_training_columns(resolution)
    ds = ray.data.read_parquet(
        data_uri,
        columns=key_list,
        shuffle="file",
        concurrency=num_data_loading_workers,
    )

    return ds.map_batches(
        convert_precision,
        batch_size=None,
        concurrency=num_data_loading_workers,
    )


########################################
# Step 2: Build Stable Diffusion model #
########################################

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


###############################################
# Step 3: Run the scalable training procedure #
###############################################

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
        precision="16-mixed",  # Use float16 instead of bfloat16 for broader GPU compatibility
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
# Main CLI: Entry point for training script #
#############################################

app = typer.Typer()

artifact_storage = os.environ["ANYSCALE_ARTIFACT_STORAGE"]
user_name = re.sub(r"\s+", "__", os.environ.get("ANYSCALE_USERNAME", "user"))
anyscale_storage_path = f"{artifact_storage}/{user_name}"
# Remove protocol prefix for storage path - works for both s3:// and abfss://
if anyscale_storage_path.startswith("s3://"):
    anyscale_storage_path = anyscale_storage_path.replace("s3://", "")
elif is_abfss_path(anyscale_storage_path):
    # Keep ABFSS paths as-is since they need the full URI
    pass


@app.command()
def train(
    experiment_name: str = "sd-demo",
    seed: int = 2024,
    init_from_pretrained: bool = False,
    model_name: str = "stabilityai/stable-diffusion-2-base",
    use_xformers: bool = False,
    batch_size_per_worker: int = 1,
    max_steps: int = 100,
    fsdp: bool = False,
    sharding_policy: Optional[str] = None,
    checkpoint_sharding_strategy: str = "full",
    resume_from_checkpoint: Optional[str] = None,
    restore_from_uri: Optional[str] = None,
    max_failures: int = 3,
    checkpoint_every_n_steps: int = 50,
    val_check_interval: int = 100,
    accumulate_grad_batches: int = 1,
    storage_path: str = anyscale_storage_path,
    prefetch_batches: int = 1,
    resolution: int = 256,
    num_training_workers: int = 1,
    num_data_loading_workers: int = 1,
    lr: float = 0.0001,
    weight_decay: float = 0.01,
    num_warmup_steps: int = 10,
    use_small_unet: bool = True,
    accelerator_type: Optional[str] = None,
    train_data_uri: str = "s3://anyscale-materials/stable-diffusion/laion_art_sample_processed_train_256.parquet",
    validation_data_uri: str = "s3://anyscale-materials/stable-diffusion/laion_art_sample_processed_valid_256.parquet",
):
    """Train a Stable Diffusion model."""
    ray.data.set_progress_bars(False)
    ctx = ray.data.DataContext.get_current()
    ctx.execution_options.verbose_progress = False

    train_ds = load_precomputed_dataset(
        train_data_uri,
        num_data_loading_workers=num_data_loading_workers,
        resolution=cast(ResolutionDtype, resolution),
    )

    validation_ds = load_precomputed_dataset(
        validation_data_uri,
        num_data_loading_workers=num_data_loading_workers,
        resolution=cast(ResolutionDtype, resolution),
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

        # Handle ABFSS vs S3 storage paths differently
        if is_abfss_path(storage_path):
            if not ADLFS_AVAILABLE:
                raise ImportError(
                    "adlfs is required for ABFSS support. Install it with: pip install adlfs"
                )
            import re
            match = re.match(r"abfss?://[^@]+@([^.]+)\.dfs\.core\.windows\.net", storage_path)
            if not match:
                raise ValueError(f"Invalid ABFSS path format: {storage_path}")

            account_name = match.group(1)
            # Test that we can create the filesystem
            test_fs = create_azure_filesystem(account_name)
            logger.info("Successfully validated ABFSS authentication")

            # For ABFSS, Ray Train doesn't support it directly for storage_path
            # Use local storage for Ray Train checkpoints, we'll handle ABFSS copying separately
            logger.warning("Ray Train doesn't support ABFSS storage_path directly.")
            logger.warning("Using local storage for checkpoints. Will upload to ABFSS after training.")

            # Create a local storage path for Ray Train checkpoints
            local_storage_path = get_local_storage_path()
            os.makedirs(local_storage_path, exist_ok=True)

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
                    storage_path=local_storage_path,
                    failure_config=FailureConfig(max_failures=max_failures),
                ),
                datasets=ray_datasets,  # type: ignore [arg-type]
                resume_from_checkpoint=checkpoint,
            )
        else:
            # For S3, use the existing approach
            fs = create_filesystem(storage_path)

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
    if is_abfss_path(storage_path):
        upload_to_abfss(storage_path, experiment_name)
    else:
        # Default S3 behavior
        try:
            fs = S3FileSystem()
            paths = fs.glob(f"{storage_path}/**/checkpoint.ckpt")
            print("Produced Model Checkpoints:")
            print("===========================")
            for p in paths:
                print(p)
        except Exception as e:
            print(f"Warning: Could not list checkpoints: {e}")


if __name__ == "__main__":
    app()
