"""
Simple Ray Serve application for demonstrating service deployment from workspace.
This echo service receives JSON messages and returns them with a timestamp.
"""

from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime

app = FastAPI()

class Message(BaseModel):
    message: str

@serve.deployment
@serve.ingress(app)
class EchoService:
    @app.post("/echo")
    def echo(self, msg: Message):
        return {
            "echo": msg.message,
            "timestamp": datetime.now().isoformat(),
            "service": "workspace-tutorial-service"
        }

# Create the deployment
deployment = EchoService.bind()
