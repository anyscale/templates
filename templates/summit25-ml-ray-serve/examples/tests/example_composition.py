"""Example: Multi-deployment application."""

from ray import serve
from ray.serve.handle import DeploymentHandle
from starlette.requests import Request


class TextPreprocessor:
    """Text preprocessing logic"""

    def preprocess(self, text: str) -> dict:
        """Clean and tokenize text"""
        cleaned_text = text.strip().lower()
        tokens = cleaned_text.split()
        return {"tokens": tokens, "length": len(tokens)}


class TextEmbedder:
    """Text embedding logic"""

    def embed(self, tokens: dict) -> list:
        """Convert tokens to embeddings"""
        # Simulate embedding generation
        return [0.1] * tokens["length"]


class TextClassifier:
    """Text classification logic"""

    def classify(self, embeddings: list) -> dict:
        """Classify based on embeddings"""
        # Simulate classification
        return {"label": "positive", "confidence": 0.85}


@serve.deployment
class TextPreprocessorDeployment(TextPreprocessor):
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        return self.preprocess(data["text"])


@serve.deployment
class TextEmbedderDeployment(TextEmbedder):
    async def __call__(self, request: Request) -> list:
        data = await request.json()
        return self.embed(data)


@serve.deployment
class TextClassifierDeployment(TextClassifier):
    async def __call__(self, request: Request) -> dict:
        data = await request.json()
        return self.classify(data["embeddings"])


@serve.deployment
class TextPipelineDeployment:
    """Composed pipeline deployment"""

    def __init__(
        self,
        preprocessor: DeploymentHandle,
        embedder: DeploymentHandle,
        classifier: DeploymentHandle,
    ):
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.classifier = classifier

    async def __call__(self, request: Request) -> dict:
        # Get text from request
        data = await request.json()
        return await self.orchestrate(data)

    async def orchestrate(self, data) -> dict:
        text = data["text"]

        # Step 1: Preprocess
        preprocessed = await self.preprocessor.preprocess.remote(text)

        # Step 2: Embed
        embeddings = await self.embedder.embed.remote(preprocessed)

        # Step 3: Classify
        result = await self.classifier.classify.remote({"embeddings": embeddings})

        return {
            "preprocessed": preprocessed,
            "embeddings": embeddings,
            "classification": result,
        }
