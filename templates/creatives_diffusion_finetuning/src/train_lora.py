"""
Stable Diffusion LoRA fine-tuning with Ray Train.

train_func runs inside each distributed worker. run_training launches the
TorchTrainer across N GPUs. Standard PyTorch + diffusers + peft — Ray Train
handles DDP, checkpointing, and fault tolerance.
"""
import os
import tempfile

import ray.train
import ray.train.torch
import torch
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer

MODEL_ID = "runwayml/stable-diffusion-v1-5"
LORA_RANK = 4
LORA_ALPHA = 4
LORA_TARGETS = ["to_q", "to_v", "to_k", "to_out.0"]
CHECKPOINT_DIR = "/mnt/cluster_storage/sd-lora-checkpoints"


def train_func(config: dict):
    """Per-worker training function.

    Loads SD 1.5 components, applies LoRA to the UNet, and runs
    the standard diffusion training loop: encode latents → sample
    noise → predict noise → MSE loss.

    Uses mixed precision via torch.autocast — model weights in fp32,
    forward pass in fp16, grad scaler for stable backprop on T4.
    """
    from diffusers import DDPMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model

    model_id = config.get("model_id", MODEL_ID)
    num_epochs = config.get("num_epochs", 3)
    batch_size = config.get("batch_size", 1)
    lr = config.get("lr", 1e-4)

    device = ray.train.torch.get_device()

    # ── Load model components in fp32 ────────────────────────────────
    # Weights stay fp32; torch.autocast handles fp16 compute
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    unet = pipe.unet
    vae = pipe.vae.to(device)
    text_encoder = pipe.text_encoder.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    # Freeze everything except LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ── Apply LoRA ─────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGETS,
        lora_dropout=0.1,
    )
    unet = get_peft_model(unet, lora_config, autocast_adapter_dtype=False)
    unet = unet.to(device)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"[Worker {ray.train.get_context().get_world_rank()}] "
          f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Wrap for DDP — only LoRA params have gradients
    unet = ray.train.torch.prepare_model(unet)

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=lr
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── Data ───────────────────────────────────────────────────────────
    train_ds = ray.train.get_dataset_shard("train")

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_ds.iter_torch_batches(
            batch_size=batch_size,
            device=device,
            dtypes={"pixel_values": torch.float32, "input_ids": torch.long},
        ):
            pixel_values = batch["pixel_values"]       # [B, 3, 512, 512]
            input_ids = batch["input_ids"]              # [B, 77]

            with torch.no_grad():
                # VAE encode in fp32
                latents = vae.encode(pixel_values).latent_dist.sample() * 0.18215

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Text embeddings (frozen encoder)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Mixed precision forward pass
            with torch.autocast("cuda", dtype=torch.float16):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # MSE loss in fp32
            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), noise.float()
            )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Save LoRA weights on last epoch
        checkpoint = None
        if epoch == num_epochs - 1:
            tmp = tempfile.mkdtemp()
            unwrapped = unet.module if hasattr(unet, "module") else unet
            unwrapped.save_pretrained(tmp)
            checkpoint = ray.train.Checkpoint.from_directory(tmp)

        ray.train.report(
            {"loss": avg_loss, "epoch": epoch},
            checkpoint=checkpoint,
        )
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}")


def run_training(
    num_workers: int,
    num_epochs: int,
    train_ds,
    model_id: str = MODEL_ID,
    batch_size: int = 1,
    lr: float = 1e-4,
):
    """Launch distributed LoRA fine-tuning with Ray Train."""
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config={
            "model_id": model_id,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr,
        },
        scaling_config=ScalingConfig(
            num_workers=num_workers,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
        run_config=RunConfig(
            name="sd-lora-finetuning",
            storage_path=CHECKPOINT_DIR,
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
        datasets={"train": train_ds},
    )

    result = trainer.fit()
    print(f"\nTraining complete — loss: {result.metrics['loss']:.4f}")
    return result
