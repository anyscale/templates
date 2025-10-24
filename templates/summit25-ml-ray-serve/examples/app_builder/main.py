from pydantic import BaseModel
from ray import serve
from ray.serve import Application

@serve.deployment
class HelloWorld:
    def __init__(self, message: str):
        self._message = message
        print("Message:", self._message)

    def __call__(self, request):
        return self._message

class HelloWorldArgs(BaseModel):
    message: str

def app_builder(args: HelloWorldArgs) -> Application:
    return HelloWorld.bind(args.message)