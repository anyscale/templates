import asyncio

import torch
import numpy as np
from ray import serve


@serve.deployment()
class MyModelDeployment:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        return torch.nn.Linear(10, 5)

    async def predict(self, data: dict) -> dict:
        result = self.model(torch.tensor(data["features"]))
        return {"prediction": result.detach().numpy().tolist()}


app = MyModelDeployment.bind("path/to/model")

handle = serve.run(app, _local_testing_mode=True)

async def test():
    data = {"features": np.random.rand(10).tolist()}
    result = await handle.predict.remote(data)
    print(f"Result: {result}")

asyncio.run(test())
