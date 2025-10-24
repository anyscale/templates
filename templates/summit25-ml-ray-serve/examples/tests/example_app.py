"""Example application code for testing Ray Serve deployments."""

from typing import Any
from ray import serve
from starlette.requests import Request
from pydantic import BaseModel


class InputRequest(BaseModel):
    input: str


class MyModel:
    """Core business logic - easily unit testable"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Model loading logic"""
        # Simulate model loading
        return f"model_loaded_from_{self.model_path}"

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Core prediction logic"""
        # Simulate prediction logic
        return {"prediction": "test_result", "confidence": 0.95}


@serve.deployment(ray_actor_options={"num_cpus": 1}, max_ongoing_requests=5)
class MyModelDeployment(MyModel):
    """Ray Serve deployment wrapper"""

    def __init__(self, model_path: str):
        super().__init__(model_path)

    async def __call__(self, request: Request) -> dict[str, Any]:
        """HTTP endpoint handler"""
        input_data = await request.json()
        valid_request = InputRequest.model_validate(input_data)
        return self.predict(valid_request.model_dump())
