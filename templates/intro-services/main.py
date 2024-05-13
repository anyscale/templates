import logging
import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()
logger = logging.getLogger("ray.serve")

@serve.deployment
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI automatically parses the HTTP request.
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        logger.info("Handling request!")
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()