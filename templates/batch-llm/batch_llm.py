from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import os

# Set the Hugging Face token. Replace the following with your token.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please specify environment variable `HF_TOKEN` as Hugging Face token to access models in Hugging Face.")

# Set to the model that you wish to use. Note that using the llama models will require a hugging face token to be set.
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
INPUT_PATH = "s3://anonymous@air-example-data/prompts.txt"

# Initialize Ray with a Runtime Environment.
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": HF_TOKEN},
        "pip": ["vllm"],
    }
)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0)


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model=HF_MODEL)

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


# Apply batch inference for all input data.
ds = ray.data.read_text(INPUT_PATH)
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=4,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=1,
    # Specify the batch size for inference.
    batch_size=32,
)

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
output_path = os.environ.get("ANYSCALE_ARTIFACT_STORAGE") + "/result"
ds.write_parquet(output_path)


# Peek first 10 results.
# NOTE: This is for local testing and debugging. For production use case,
# one should write full result out as shown below.
# outputs = ds.take(limit=10)
# for output in outputs:
#     prompt = output["prompt"]
#     generated_text = output["generated_text"]
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
