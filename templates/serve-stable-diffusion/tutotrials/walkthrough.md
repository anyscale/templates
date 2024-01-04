# Walkthrough of the template

## Overview

| Template Specification | Description |
| ---------------------- | ----------- |
| Summary | This app provides users a one click production option for serving a pre-trained Stable Diffusion model from HuggingFace.  It leverages [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to deploy locally and built in IDE integration on an Anyscale Workspace to iterate and add additional logic to the application. You can then use UI or a simple CLI to deploy to production with Anyscale Services. |
| Time to Run | Around 2 minutes to setup the models and generate your first image(s). Less than 10 seconds for every subsequent round of image generation (depending on the image size). |
| Minimum Compute Requirements | At least 1 GPU node with 1 NVIDIA A10 GPU. |
| Cluster Environment | This template uses a docker image built on top of the latest Anyscale-provided Ray 2.9 image using Python 3.9: [`anyscale/ray:latest-py39-cu118`](https://docs.anyscale.com/reference/base-images/overview). See the appendix below for more details. |

## Code definition: model loading and inference logic
This includes setting up:
- The stable diffusion model loaded inside a Ray Serve Deployment.
- The *number of model replicas* to keep active in our Ray cluster. These model replicas can process incoming requests concurrently. Note: autoscaling of model replicas is enabled below.

```
@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    autoscaling_config={"min_replicas": 1, "max_replicas": 99},
)
class StableDiffusionV2:
    def __init__(self):
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

        image = self.pipe(prompt, height=img_size, width=img_size).images[0]

        return image
```


## Code definition: actual API endpoint to live at `/imagine`.

```
app = FastAPI()


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await self.handle.generate.remote(prompt, img_size=img_size)

        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")
```

## Development workflow: deploy the Ray Serve application locally 
The Ray Serve Application is deployed at `http://localhost:8000` by running:
```
serve run app:entrypoint
```

## Development workflow: deploy the model as Anyscale Services for production

After the local test, it's recommended to deploy your model as Anyscale Services for staging/production traffic so that you can:
- enjoy the benefits of fault tolerence of Anyscale Services
- continue to iterate in your workspace and perform rolling upgrade to your Production Service without downtime.


## Summary

This template used [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) to serve many replicas of a stable diffusion model and deploy it as an Anyscale Service for staging or production traffic.
- See this [getting started guide](https://docs.ray.io/en/latest/serve/getting_started.html) for a more detailed walkthrough of Ray Serve.