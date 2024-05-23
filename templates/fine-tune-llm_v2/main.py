import argparse
import os
import subprocess
import yaml
import random
import string


def _read_yaml_file(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)

def _get_lora_storage_uri() -> str:
    artifact_storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
    artifact_storage = artifact_storage.rstrip("/")
    return f"{artifact_storage}/lora_fine_tuning/"


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Submit a job with a configuration file"
    )
    parser.add_argument(
        "finetune_config",
        type=str,
        help="Path to the fine-tuning configuration YAML file",
    )

    # Parse arguments
    args = parser.parse_args()

    finetune_config_path = args.finetune_config
    training_config = _read_yaml_file(finetune_config_path)

    is_lora = "lora_config" in training_config
    entrypoint = f"llmforge dev finetune {finetune_config_path}"

    if is_lora:
        lora_storage_uri = _get_lora_storage_uri()
        entrypoint += f" --forward-best-checkpoint-remote-uri={lora_storage_uri}"
        print(f"Note: LoRA weights will also be stored inside {lora_storage_uri}")
        
    else:
        lora_storage_uri = None

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        entrypoint = f"WANDB_API_KEY={api_key} {entrypoint}"

    subprocess.run(entrypoint, check=True, shell=True)
    if lora_storage_uri:
        

if __name__ == "__main__":
    main()
