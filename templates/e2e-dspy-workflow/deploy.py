import dspy
from ray import serve
from fastapi import FastAPI
from src import MODEL_PARAMETERS
import json
from starlette.requests import Request
from urllib.parse import urlparse


app = FastAPI()

def read_params(file_path: str):
    """Simple helper for reading dspy program parameters"""
    with open(file_path, "r") as f:
        params = json.load(f)
    return params

class IntentClassification(dspy.Signature):
    """As a part of a banking issue traiging system, classify the intent of a natural language query into one of the 25 labels."""
    intent = dspy.InputField(desc="Intent of the query")
    label = dspy.OutputField(desc="Type of the intent; Should just be one of the 25 labels with no other text")

class IntentClassificationModule(dspy.Module):
    def __init__(self, labels_in_use):
        self.intent_classifier = dspy.ChainOfThought(IntentClassification)
        self.valid_labels = set(labels_in_use)

    def forward(self, text, **predictor_kwargs):
        prediction = self.intent_classifier(intent=text, **predictor_kwargs)
        sanitized_prediction = dspy.Prediction(label=prediction.label.lower().strip().replace(" ", "_"), reasoning=prediction.reasoning)
        if sanitized_prediction.label not in self.valid_labels:
            sanitized_prediction = dspy.Prediction(label="INVALID MODEL OUTPUT")
        return sanitized_prediction

@serve.deployment(
    ray_actor_options={"num_cpus": 0.1},
    autoscaling_config=dict(min_replicas=1, max_replicas=5)
)
class LLMClient:
    def __init__(self, param_path, serve_args):
        params = read_params(param_path)
        self.params = params
        dspy.settings.configure(experimental=True)
        base_url = serve_args["api_base"]
        prefix = serve_args["route_prefix"]
        full_url = base_url + prefix.lstrip('/') if len(prefix.lstrip('/')) else base_url
        api_parameters = {
            "api_base": f"{full_url}/v1",
            "api_key": serve_args["api_key"]
        }
        print("API parameters", api_parameters)
        self.llm = dspy.LM(model="openai/" + self.params["best_model"], **MODEL_PARAMETERS, **api_parameters)
        self.program = IntentClassificationModule(params["labels_in_use"])
        self.program.load(params["best_program_path"])

    async def __call__(self, query: str):
        """Answer the given question and provide sources."""
        with dspy.context(lm=self.llm):
            retrieval_response = self.program(query)
        return retrieval_response.label


def construct_app(args):
    return LLMClient.bind(args["program_param_path"], args["rayllm_args"])
