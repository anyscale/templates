import io
import numpy
import os
import tritonserver
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ray import serve

MODEL_REPOSITORY = f'{os.environ["ANYSCALE_ARTIFACT_STORAGE"]}/triton_model_repository'
app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class TritonDeployment:
    def __init__(self):
        self._triton_server = tritonserver.Server(
            model_repository=MODEL_REPOSITORY,
            model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
            log_info=False,
        )
        self._triton_server.start(wait_until_ready=True)

        # Load model using Triton.
        self._model = None
        if not self._triton_server.model("stable_diffusion_1_5").ready():
            self._model = self._triton_server.load("stable_diffusion_1_5")

            if not self._model.ready():
                raise Exception("Model not ready")

    @app.get("/generate")
    def generate(self, prompt: str) -> StreamingResponse:
        print(f"Generating image for prompt: {prompt}")
        for response in self._model.infer(inputs={"prompt": [[prompt]]}):
            generated_image = (
                numpy.from_dlpack(response.outputs["generated_image"])
                .squeeze()
                .astype(numpy.uint8)
            )
            image_ = Image.fromarray(generated_image)

            # Stream back the image to the caller
            buffer = io.BytesIO()
            image_.save(buffer, "JPEG")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="image/jpeg")


triton_deployment = TritonDeployment.bind()
