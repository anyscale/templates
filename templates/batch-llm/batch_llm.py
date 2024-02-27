from vllm import LLM, SamplingParams
from typing import Dict
import numpy as np
import ray
import os

# Set the Hugging Face token. Replace the following with your token.
hf_token = "<REPLACE_WITH_YOUR_HUGGING_FACE_USER_TOKEN>"
# Set to the model that you wish to use. Note that using the llama models will require a hugging face token to be set.
hf_model = "meta-llama/Llama-2-7b-chat-hf"
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, max_tokens=4096)


# Initialize Ray with a Runtime Environment - used to set the Environment Variable `HF_TOKEN`
# This line can be removed if setting the Environment Variable using a Cluster Environment or in the job yaml file.
ray.init(runtime_env={"env_vars": {"HF_TOKEN": hf_token}})

input_path = "s3://anonymous@air-example-data/prompts.txt"
ds = ray.data.read_text(
    input_path,
    parallelism=10,
)


class LLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = LLM(model=hf_model)

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
