import subprocess
from typing import Any

import json
import numpy as np
import torch
from ray import serve
from starlette.requests import Request


@serve.deployment()
class OnlineMNISTClassifier:
    def __init__(self, remote_path: str, local_path: str, device: str):
        subprocess.run(f"aws s3 cp {remote_path} {local_path}", shell=True, check=True)

        self.device = device
        self.model = torch.jit.load(local_path).to(device).eval()

    async def __call__(self, request: Request) -> dict[str, Any]:
        batch = json.loads(await request.json())
        return await self.predict(batch)

    async def predict(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = torch.tensor(batch["image"]).float().to(self.device)

        with torch.no_grad():
            logits = self.model(images).cpu().numpy()

        batch["predicted_label"] = np.argmax(logits, axis=1)
        return batch


mnist_app = OnlineMNISTClassifier.bind(
    remote_path="s3://anyscale-public-materials/ray-ai-libraries/mnist/model/model.pt",
    local_path="/mnt/cluster_storage/model.pt",
    device="cpu",
)
