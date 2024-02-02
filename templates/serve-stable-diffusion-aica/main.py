import logging
from io import BytesIO

import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

from ray import serve
from ray.serve.handle import DeploymentHandle

# Create a FastAPI instance to handle HTTP parsing and validation.
# Learn more: https://docs.ray.io/en/latest/serve/http-guide.html#fastapi-http-deployments.
app = FastAPI()

# Set up the Ray Serve logger.
logger = logging.getLogger("ray.serve")

# A serve deployment contains business logic or an ML model to handle incoming requests.
# It consists of a number of replicas that are individual copies of the class or function defined within it.
@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

    # Raise an error if a user sends a request to the root path.
    @app.get("/")
    async def generate(self):
        raise HTTPException(
            status_code=400, detail="'/' not supported, use '/imagine' instead."
        )

    # Handle API requests to the `/imagine` route.
    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        if not prompt:
            raise HTTPException(status_code=400, detail="Missing 'prompt' parameter.")

        image = await self.handle.generate.remote(prompt, img_size=img_size)

        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


@serve.deployment(
    ray_actor_options={"num_gpus": 1, 
        "num_cpus": 1,  # Set the number of GPUs and CPUs required for each model replica.
        "accelerator_type": "A10G"},   # Set accelerator type based on GPU type to use (T4, A10G, L4, V100, A100-40G or A100-80G)
    max_concurrent_queries=2,   # Maximum number of queries that are sent to a replica of this deployment without receiving a response.
    autoscaling_config={
        "min_replicas": 1,    # The minimum number of model replicas to keep active in the Ray cluster.
        "max_replicas": 99,    # The maximum number of model replicas to keep active in the Ray cluster.
        "target_num_ongoing_requests_per_replica": 1,   # Target number of ongoing requests in a replica. Serve compares the actual number agasint this value and upscales or downscales.
    },
)
class StableDiffusionV2:
    def __init__(self):
        # Load the stable diffusion model inside a Ray Serve Deployment.
        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")
        

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"
        # Use Serve logger to emit structured (JSON) logs as the best practice for production use cases.
        logger.info("Prompt: [%s]", prompt)
        image = self.pipe(prompt, height=img_size, width=img_size).images[0]

        return image

# Bind the deployments to arguments that will be passed into its constructor. 
# This defines a Ray Serve application that we can run locally or deploy to production.
stable_diffusion_app = APIIngress.bind(StableDiffusionV2.bind())
