from ray import serve
from typing import List
from transformers import pipeline
from fastapi import FastAPI

app = FastAPI()

@serve.deployment
@serve.ingress(app)
class Text2Text:
    def __init__(self) -> None:
        # https://huggingface.co/tasks/text-generation
        self.model = pipeline('text-generation', model='gpt2')

    def ask(self, prompt) -> List[str]:
        predictions = self.model(prompt, max_length = 50, num_return_sequences=1)
        return [str(sequence["generated_text"]) for sequence in predictions]

    @app.post("/query")
    async def query(self, prompt: str) -> List[str]:
        return self.ask(prompt)

deployment = Text2Text.bind()
