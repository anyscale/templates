"""Ray Data transform classes for Stable Diffusion preprocessing.

These classes are in a separate importable module (not __main__) so that
Ray's cloudpickle can serialize class references without pickling the
entire module globals, which avoids issues with non-serializable objects
like PyTorch's GenericModule in newer torch versions.
"""

import gc
import io
import logging
import math
from typing import Optional

import numpy as np
import torch
import torchvision  # type: ignore
from PIL import Image

logger = logging.getLogger("ray.data")

### Constants ###
CAPTION_LATENTS_KEY = "caption_latents"
IMAGE_LATENTS_256_KEY = "latents_256_bytes"
IMAGE_LATENTS_512_KEY = "latents_512_bytes"


#### Utils ####
class LargestCenterSquare:
    """Largest center square crop for images."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, img: Image.Image) -> Image.Image:
        """Crop the largest center square from an image."""
        img = torchvision.transforms.functional.resize(
            img=img,
            size=self.size,
        )
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


class SDTransformer:
    """Image and text transforms."""

    def __init__(
        self,
        resolution: int,
        model_name: str = "CompVis/stable-diffusion-v1-4",
    ) -> None:
        from transformers import CLIPTokenizer

        self.resolution = resolution
        normalize = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.crop = LargestCenterSquare(resolution)
        self.transforms = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), normalize]
        )
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


class SDLatentEncoder:
    """Latent encoder to encode images and text."""

    def __init__(
        self,
        resolution: int = 256,
        device: Optional[str] = None,
        model_name: str = "CompVis/stable-diffusion-v1-4",
    ) -> None:
        from diffusers.models import AutoencoderKL
        from transformers import CLIPTextModel

        self.device = torch.device(device) if device else resolve_device()
        self.resolution = resolution

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
            input_images = batch[f"image_{self.resolution}"]
            image_latents = self.encode_images(input_images)
            batch[self.image_latents_key] = image_latents

            del batch[f"image_{self.resolution}"]
            gc.collect()

            caption_ids = batch["caption_ids"]
            batch[CAPTION_LATENTS_KEY] = self.encode_text(caption_ids)

            del batch["caption_ids"]
            gc.collect()

        return batch
