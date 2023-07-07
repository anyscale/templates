from ray import serve

from io import BytesIO
from PIL import Image
from starlette.requests import Request
from typing import Dict

import torch
import time
import torch_tensorrt
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
import io
import onnxruntime 
import numpy as np
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@serve.deployment(ray_actor_options={"num_gpus": 0.5})
@serve.ingress(app)
class Classifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_height = 224
        image_width = 224
        batch_size = 1
        x = torch.randn(batch_size, 3, image_height, image_width, requires_grad=True).to(self.device)
        torch_out = resnet50(x)
        buffer = io.BytesIO()
        torch.onnx.export(resnet50,                     # model being run
                  x,                            # model input (or a tuple for multiple inputs)
                  buffer,              # where to save the model (can be a file or file-like object)
                  export_params=True,           # store the trained parameter weights inside the model file
                  opset_version=12,             # the ONNX version to export the model to
                  do_constant_folding=True,     # whether to execute constant folding for optimization
                  input_names = ['input'],      # the model's input names
                  output_names = ['output'])    # the model's output names
        self.onnx_rt32 = onnxruntime.InferenceSession(buffer.getvalue(), providers=['CUDAExecutionProvider'])
        with open("imagenet_classes.txt", "r") as f:
            self.categories = [s.strip() for s in f.readlines()]

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @app.post('/')
    async def call(self, file: UploadFile) -> str:
        image_payload_bytes = await file.read()
        img = Image.open(BytesIO(image_payload_bytes))
        images = [img]  # Batch size is 1
        input_arr = np.concatenate([self.preprocess(i).unsqueeze(0) for i in images])
        ort_outputs = self.onnx_rt32.run([], {'input':input_arr})[0]
        output = ort_outputs.flatten()
        output = self.softmax(output)
        ind = np.argmax(output)
        return self.categories[ind]
        
model = Classifier.bind()