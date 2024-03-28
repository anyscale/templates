from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import os

from utils import generate_output_path

# Set the Hugging Face token. Replace the following with your token.
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Please specify environment variable `HF_TOKEN` as Hugging Face token to access models in Hugging Face.")

# Set to the model that you wish to use. Note that using the llama models will require a hugging face token to be set.
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"

# Input path to read input data.
# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
INPUT_PATH = "s3://anonymous@air-example-data/prompts_100.txt"

# Output path to write output result.
output_path = generate_output_path(os.environ.get("ANYSCALE_ARTIFACT_STORAGE"), HF_MODEL)

# Initialize Ray with a Runtime Environment.
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    }
)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=2048)

# The number of LLM instances to use.
num_llm_instances = 4
# The number of GPUs to use per LLM instance.
num_gpus_per_instance = 1

# Mapping of model name to max_model_len supported by model.
model_name_to_args = {
    "mistralai/Mistral-7B-Instruct-v0.1": {"max_model_len": 16832},
    "google/gemma-7b-it": {"max_model_len": 2432},
    "mlabonne/NeuralHermes-2.5-Mistral-7B": {"max_model_len": 16800},
}


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(
            model=HF_MODEL,
            **model_name_to_args.get(HF_MODEL, {}),
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt = []
        generated_text = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


# Apply batch inference for all input data.
ds = ray.data.read_text(INPUT_PATH)
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=num_llm_instances,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=num_gpus_per_instance,
    # Specify the batch size for inference. Set the batch size to as large possible without running out of memory.
    # If you encounter CUDA out-of-memory errors, decreasing batch_size may help.
    batch_size=10,
)

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
ds.write_parquet(output_path)

print(f"Batch inference result is written into {output_path}.")

# Peek first 10 results.
# NOTE: This is for local testing and debugging.
# output_ds = ray.data.read_parquet(output_path)
# outputs = output_ds.take(limit=10)
# for output in outputs:
#     prompt = output["prompt"]
#     generated_text = output["generated_text"]
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
