import os
from enum import Enum
from typing import Dict

class CloudProvider(str, Enum):
    AWS = "AWS"
    GCP = "GCP"

AVAILABLE_GPU_TYPES = {
    CloudProvider.AWS: ["A10", "A100-40G", "A100-80G"],
    CloudProvider.GCP: ["L4", "A100-40G", "A100-80G"],
} 

def _get_model() -> str:
    models = {
        1: "meta-llama/Llama-2-7b-chat-hf",
        2: "meta-llama/Llama-2-13b-chat-hf",
        3: "meta-llama/Llama-2-70b-chat-hf",
        4: "mistralai/Mistral-7B-Instruct-v0.1",
        5: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    }
    return _get_user_option(models, "Please select the model type:")

def _get_user_option(int_to_names: Dict[int, str], prompt: str) -> str:
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


def _get_cloud_provider() -> CloudProvider:
    artifact_storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", "")
    if artifact_storage.startswith("s3"):
        return CloudProvider.AWS
    elif artifact_storage.startswith("gs"):
        return CloudProvider.GCP
    else:
        raise RuntimeError("Unable to determine the valid cloud provider. ")

def _get_gpu_type(cloud_provider: CloudProvider) -> str:
    gpus = AVAILABLE_GPU_TYPES[cloud_provider]
    number_to_names = {idx+1: val for idx, val in enumerate(gpus)}
    return _get_user_option(number_to_names, "Please select the GPU type")


def main():
    cloud_provider = _get_cloud_provider()
    model = _get_model()
    gpu_type = _get_gpu_type(cloud_provider)
    print(f"cloud_provider is {cloud_provider}")
    print(f"model is {model}")
    print(f"gpu_type is {gpu_type}")


if __name__ == "__main__":
    main()
