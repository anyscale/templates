import requests
from fastapi import FastAPI
from ray import serve

fastapi = FastAPI()

@serve.deployment
@serve.ingress(fastapi)
class FastAPIDeployment:
    # FastAPI will automatically parse the HTTP request for us.
    # Check out https://docs.ray.io/en/latest/serve/http-guide.html
    @fastapi.get("/hello")
    def say_hello(self, name: str) -> str:
        return f"Hello {name}!"

my_app = FastAPIDeployment.bind()
