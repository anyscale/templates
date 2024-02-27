from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import os

input_path = "s3://anonymous@air-example-data/prompts.txt"
ds = ray.data.read_text(
    input_path,
    parallelism=10,
)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)


class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts. The output is a list of RequestOutput objects
        # that contain the prompt, generated text, and other information.
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(" ".join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


ds = ds.map_batches(
    LLMPredictor,
    concurrency=10,
    num_gpus=1,
    batch_size=32,
)

output_path = os.environ.get("ANYSCALE_ARTIFACT_STORAGE") + "/result"
ds.write_parquet(output_path)
