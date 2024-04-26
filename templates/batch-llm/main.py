from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import os

from util.utils import generate_output_path, get_a10g_or_equivalent_accelerator_type, read_hugging_face_token_from_cache


# Set to the model that you wish to use. Note that using the llama models will require a hugging face token to be set.
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"

# Input path to read input data.
# Read one text file from S3. Ray Data supports reading multiple files
# from cloud storage (such as JSONL, Parquet, CSV, binary format).
INPUT_PATH = "s3://anonymous@air-example-data/prompts_100.txt"

# Name of the column which contains the raw input text.
INPUT_TEXT_COLUMN = "text"

# Output path to write output result.
output_path = generate_output_path(os.environ.get("ANYSCALE_ARTIFACT_STORAGE"), HF_MODEL)

# Read the Hugging Face token from cached file.
HF_TOKEN = read_hugging_face_token_from_cache()

# Initialize Ray with a Runtime Environment.
ray.init(
    runtime_env={
        "env_vars": {"HF_TOKEN": HF_TOKEN},
    }
)

# Create a sampling params object.
sampling_params = SamplingParams(
    n=1,
    temperature=0,
    max_tokens=2048,
    stop=["<|eot_id|>", "<|end_of_text|>"],
)

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

# Mapping of model name to input prompt format.
model_name_to_input_prompt_format = {
    "meta-llama/Llama-2-7b-chat-hf": "[INST] {} [/INST]",
    "mistralai/Mistral-7B-Instruct-v0.1": "[INST] {} [/INST]",
    "google/gemma-7b-it": "<start_of_turn>model\n{}<end_of_turn>\n",
    "mlabonne/NeuralHermes-2.5-Mistral-7B": "<|im_start|>system\nYou are a helpful assistant that will complete the sentence in the given input prompt.<|im_end|>\n<|im_start|>user{}<|im_end|>\n<|im_start|>assistant",
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant. Complete the given prompt in several concise sentences.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
}


def construct_input_prompt(row, text_column):
    """Given the input row with raw text in `text_column` column,
    construct the input prompt for the model."""
    prompt_format = model_name_to_input_prompt_format.get(HF_MODEL)
    if prompt_format:
        row[text_column] = prompt_format.format(row[text_column])
    return row


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self, text_column):
        # Name of column containing the input text.
        self.text_column = text_column

        # Create an LLM.
        self.llm = LLM(
            model=HF_MODEL,
            **model_name_to_args.get(HF_MODEL, {}),
        )

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch[self.text_column], sampling_params)
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
ds = ds.map(
    construct_input_prompt,
    fn_kwargs={"text_column": INPUT_TEXT_COLUMN},
)
ds = ds.map_batches(
    LLMPredictor,
    # Set the concurrency to the number of LLM instances.
    concurrency=num_llm_instances,
    # Specify the number of GPUs required per LLM instance.
    num_gpus=num_gpus_per_instance,
    # Specify the batch size for inference. Set the batch size to as large
    # as possible without running out of memory.
    # If you encounter CUDA out-of-memory errors, decreasing
    # batch_size may help.
    batch_size=5,
    # Pass keyword arguments for the LLMPredictor class.
    fn_constructor_kwargs={"text_column": INPUT_TEXT_COLUMN},
    # Select the accelerator type; A10G or L4.
    accelerator_type=get_a10g_or_equivalent_accelerator_type(),
)

# Write inference output data out as Parquet files to S3.
# Multiple files would be written to the output destination,
# and each task would write one or more files separately.
ds.write_parquet(output_path, try_create_dir=False)

print(f"Batch inference result is written into {output_path}.")

# Peek first 10 results.
# NOTE: This is for local testing and debugging.
# output_ds = ray.data.read_parquet(output_path)
# outputs = output_ds.take(limit=10)
# for output in outputs:
#     prompt = output["prompt"]
#     generated_text = output["generated_text"]
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
