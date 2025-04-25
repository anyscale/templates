import torch
from diffusers import StableDiffusionPipeline
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Create output directory
os.makedirs("model_repository/stable_diffusion/1", exist_ok=True)

# Load a specific model version that's known to work well with ONNX conversion
model_id = "runwayml/stable-diffusion-v1-5"  # This is often the most compatible
model_path = Path("model_repository/stable_diffusion/1")

pipe = StableDiffusionPipeline.from_pretrained(model_id)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Get model dimensions before attempting export
# This is crucial to ensure matrix dimensions match
unet = pipe.unet
text_encoder = pipe.text_encoder
hidden_size = text_encoder.config.hidden_size

# Log dimensions for debugging
logger.info(f"Text encoder output dimension: {hidden_size}")
logger.info(f"UNet cross attention dimension: {unet.config.cross_attention_dim}")

# Only proceed if dimensions match
if hidden_size != unet.config.cross_attention_dim:
    logger.error("ERROR: Model component dimensions don't match! Cannot convert.")
    exit(1)

# Create correctly sized inputs based on actual model dimensions
batch_size = 1

# Create dummy inputs with correct dimensions
dummy_text_input = torch.ones((batch_size, 77), dtype=torch.int64, device=device)
dummy_sample = torch.randn(batch_size, 4, 64, 64, device=device)
timestep = torch.tensor([999], device=device)
encoder_hidden_states = torch.randn(batch_size, 77, hidden_size, device=device)

# Export text encoder
torch.onnx.export(
    torch.ones((batch_size, 77), dtype=torch.int64, device=device),
    dummy_text_input,
    model_path / 'text_encoder.onnx',
    opset_version=14,
    input_names=["input_ids"],
    output_names=["last_hidden_state"],
)

logger.info("Text encoder exported successfully")

# Export UNet
torch.onnx.export(
    unet,
    (dummy_sample, timestep, encoder_hidden_states),
    model_path / 'unet.onnx',
    opset_version=14,
    input_names=["sample", "timestep", "encoder_hidden_states"],
    output_names=["out_sample"],
)

logger.info("UNet exported successfully")

# Export VAE decoder
dummy_latent = torch.randn(batch_size, 4, 64, 64, device=device)

torch.onnx.export(
    pipe.vae.decoder,
    dummy_latent,
    model_path / 'vae_decoder.onnx',
    opset_version=14,
    input_names=["latent"],
    output_names=["image"],
)

logger.info("VAE decoder exported successfully")
logger.info("Export complete. Models saved to model_repository/stable_diffusion/1/")
