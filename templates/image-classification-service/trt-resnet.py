from ray import serve

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import torch
import time
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi import FastAPI, File, UploadFile

app = FastAPI()


@serve.deployment(ray_actor_options={"num_gpus": 0.5})
@serve.ingress(app)
class Classifier:
    def __init__(self):
        import torch_tensorrt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(self.device)
        batch_size = 1 # Batch size is 1 in this example but it can be modified
        self.trt_model_fp32 = torch_tensorrt.compile(resnet50, inputs = [torch_tensorrt.Input((batch_size, 3, 224, 224), dtype=torch.float32)],
            enabled_precisions = torch.float32, # Run with FP32
            workspace_size = 1 << 22
        )
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    @app.post('/')
    async def call(self, file: UploadFile) -> str:
        image_payload_bytes = await file.read()
        img = Image.open(BytesIO(image_payload_bytes))
        images = [img]  # Batch size is 1
        input_tensor = torch.cat(
            [self.preprocess(i).unsqueeze(0) for i in images]
        ).to(self.device)
        with torch.no_grad():
            output = self.trt_model_fp32(input_tensor)
            sm_output = torch.nn.functional.softmax(output[0], dim=0)
        ind = torch.argmax(sm_output)
        return self.categories[ind]
        
model = Classifier.bind()
