import argparse
import json
import yaml
import os
from pathlib import Path
from typing import Any, Dict
import ray
import pathlib

starter_model_file_path = pathlib.Path(__file__).parent.resolve() / "models/llama/meta-llama--Llama-2-7b-chat-hf_a10g_tp1.yaml"

accelerator_mapping = {
    "A10G": "accelerator_type:A10G",
    "V100": "accelerator_type:V100",
    "A100-40G": "accelerator_type:A100-40G",
    "A100-80G": "accelerator_type:A100-80G"
}

# python generate_config.py --model-id "mistralai/Mistral-7B-Instruct-v0.1" --accelerator "A100-40G" --tp 1
def generate_config(
    model_id: str,
    accelerator: str,
    tp: int
):
    """
    Args:
        model_id: The name of the model to generate.
        model_size: Parameter size of the model in billions.
        accelerator: Accelerator type for the model
        tp: Tensor parallelism that you want to use
        result_filename: The filename to save the results to.
    """
    # TO DO first validate params - model_id format, accelerator (supported accelerators)
    supported_acclerators = accelerator_mapping.keys()
    if accelerator not in supported_acclerators:
        print(f"Invalid accelerator. Please provide one of the Supported acclerators: {supported_acclerators}")
        return
    resource_type = accelerator_mapping.get(accelerator)
    config_template = {}
    with open(starter_model_file_path, mode="r") as f:
        config_template = yaml.safe_load(f.read())
        print(config_template["engine_config"])

    config_template["engine_config"]["model_id"] = model_id
    config_template["engine_config"]["hf_model_id"] = model_id

    config_template["scaling_config"]["resources_per_worker"] = {
        resource_type: 0.001
    }
    config_template["scaling_config"]["num_workers"] = tp

    model_part = model_id.replace("/", "--")
    result_filename = f"./models/{model_part}_{accelerator}_tp{tp}.yaml"
    with open(result_filename, mode="w") as f:
        yaml.dump(config_template, f)
        print(f"Output model config file saved to {result_filename}")



args = argparse.ArgumentParser(
    description="Generates a LLM model configuration."
)

args.add_argument(
    "--model-id", 
    type=str, 
    required=True, 
    help="Id of the model you are using. Format follows model ids in Hugging face. Ex: mosaicml/mpt-7b-instruct."
)
args.add_argument(
    "--accelerator", 
    type=str,
    default="A10", 
    required=True, 
    help=(
        "Accelerator that you want to use Ex: A100, A10G, L4."
        "(default: %(default)s)"
    )
)
args.add_argument(
    "--tp",
    type=int,
    default=1, 
    help=("Tensor parallelism that you want to use. Ex: 1"),
)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})
    args = args.parse_args()

    generate_config(
        model_id=args.model_id,
        accelerator=args.accelerator,
        tp=args.tp,
    )