import os
from typing import Any, Dict
from enum import Enum
from typing import Dict
import yaml

class CloudProvider(str, Enum):
    AWS = "AWS"
    GCP = "GCP"

AVAILABLE_GPU_TYPES = {
    ("meta-llama/Llama-2-7b-chat-hf", CloudProvider.AWS): ["A10", "A100-40G", "A100-80G"],
    ("meta-llama/Llama-2-13b-chat-hf", CloudProvider.AWS): ["A10", "A100-40G", "A100-80G"],
    ("meta-llama/Llama-2-70b-chat-hf", CloudProvider.AWS): ["A100-40G", "A100-80G"],
    ("mistralai/Mistral-7B-Instruct-v0.1", CloudProvider.AWS): ["A10", "A100-40G", "A100-80G"],
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", CloudProvider.AWS): ["A100-40G", "A100-80G"],
    ("meta-llama/Llama-2-7b-chat-hf", CloudProvider.GCP): ["L4", "A100-40G", "A100-80G"],
    ("meta-llama/Llama-2-13b-chat-hf", CloudProvider.GCP): ["L4", "A100-40G", "A100-80G"],
    ("meta-llama/Llama-2-70b-chat-hf", CloudProvider.GCP): ["A100-40G", "A100-80G"],
    ("mistralai/Mistral-7B-Instruct-v0.1", CloudProvider.GCP): ["L4", "A100-40G", "A100-80G"],
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", CloudProvider.GCP): ["A100-40G", "A100-80G"],
} 

GPU_TYPE_TO_ACCELERATOR_TYPE_MAP = {
    "A10": "accelerator_type:A10G",
    "L4": "accelerator_type:L4",
    "A100-40G": "accelerator_type:A100-40G",
    "A100-80G": "accelerator_type:A100-80G",
}

LLAMA_MODELS = {"meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"}

MODEL_ID_TO_BASE_CONFIGS = {
    "meta-llama/Llama-2-7b-chat-hf": "models/llama/meta-llama--Llama-2-7b-chat-hf.yaml",
    "meta-llama/Llama-2-13b-chat-hf": "models/llama/meta-llama--Llama-2-13b-chat-hf.yaml",
    "meta-llama/Llama-2-70b-chat-hf": "models/llama/meta-llama--Llama-2-70b-chat-hf.yaml",
    "mistralai/Mistral-7B-Instruct-v0.1": "models/mistral/mistralai--Mistral-7B-Instruct-v0.1.yaml",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "models/mistral/mistralai--Mixtral-8x7B-Instruct-v0.1.yaml",
}


def get_model_id() -> str:
    models = {
        1: "meta-llama/Llama-2-7b-chat-hf",
        2: "meta-llama/Llama-2-13b-chat-hf",
        3: "meta-llama/Llama-2-70b-chat-hf",
        4: "mistralai/Mistral-7B-Instruct-v0.1",
        5: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
    return get_user_option(models, "\nPlease select the model type:")

def get_user_option(int_to_names: Dict[int, str], prompt: str) -> str:
    # Reverse mapping for name to number
    names_to_numbers = {v: k for k, v in int_to_names.items()}

    # Display the choices to the user
    print(prompt)
    for number, model in int_to_names.items():
        print(f"[{number}]: {model}")

    # Ask the user for their choice
    while True:
        user_input = input("Enter the number or the name corresponding to your choice: ").strip()

        # Check if input is an integer and valid
        if user_input.isdigit() and int(user_input) in int_to_names:
            return int_to_names[int(user_input)]

        # Check if input is a valid model name
        elif user_input in names_to_numbers:
            return user_input

        else:
            print("Invalid choice. Please enter a valid number or name from the list.")


def get_cloud_provider() -> CloudProvider:
    artifact_storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "")
    if artifact_storage.startswith("s3"):
        return CloudProvider.AWS
    elif artifact_storage.startswith("gs"):
        return CloudProvider.GCP
    else:
        raise RuntimeError("Unable to determine the valid cloud provider. ")

def get_gpu_type(model_id: str, cloud_provider: CloudProvider) -> str:
    gpus = AVAILABLE_GPU_TYPES[(model_id, cloud_provider)]
    number_to_names = {idx+1: val for idx, val in enumerate(gpus)}
    return get_user_option(number_to_names, "\nPlease select the GPU type:")

def get_hf_token() -> str:
    while True:
        user_input = input("\nA Hugging Face Access Token for an account with permissions to download the Meta Llama-2 models is needed. Please enter your access token: ").strip()
        if user_input:
            return user_input
        else:
            print("Invalid entry. Please enter a valid Hugging Face Access Token.")

def read_yaml_file(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)

def get_tensor_parallelism() -> str:
    while True:
        user_input = input("\nPlease enter the tensor parallelism: ").strip()
        if user_input.isdigit() and int(user_input) > 0:
            return user_input
        else:
            print("Invalid entry. Please enter a valid number.")

def populate_configs(base_config: Dict[str, Any], model_id: str, gpu_type: str, tensor_parallelism: str):
    base_config["engine_config"]["model_id"] = model_id
    base_config["engine_config"]["hf_model_id"] = model_id
    accelerator_type = GPU_TYPE_TO_ACCELERATOR_TYPE_MAP.get(gpu_type, "")
    base_config["deployment_config"]["ray_actor_options"]["resources"][accelerator_type] = 0.001
    base_config["scaling_config"]["resources_per_worker"][accelerator_type] = 0.001
    base_config["scaling_config"]["num_workers"] = int(tensor_parallelism)
    return base_config

