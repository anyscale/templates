import argparse
import os
import subprocess
import yaml
import random
import string
from typing import Dict, Any
import datetime
from pathlib import Path


def _read_yaml_file(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)

def _get_lora_storage_uri() -> str:
    artifact_storage = os.environ.get("ANYSCALE_ARTIFACT_STORAGE")
    artifact_storage = artifact_storage.rstrip("/")
    return f"{artifact_storage}/lora_fine_tuning/"


def generate_model_tag(model_id: str) -> str:
    """
    Constructs a finetuned model ID based on the Anyscale endpoints convention.
    """
    username = os.environ.get("ANYSCALE_USERNAME")
    if username:
        username = username.strip().replace(" ", "")[:5]
        if len(username) < 5:
            padding_char = username[-1] if username else 'a'
            username += padding_char * (5 - len(username))
    else:
        username = "".join(random.choices(string.ascii_lowercase, k=5))
    suffix = "".join(random.choices(string.ascii_lowercase, k=5))
    return f"{model_id}:{username}:{suffix}"

def _get_webhook_config(model_tag: str, is_lora: bool) -> Dict[str, Any]:
    base_url = os.environ.get("ANYSCALE_HOST")
    webhook_base_url = f"{base_url}/v2"
    ft_type = "lora" if is_lora else "full"
    return {
        "webhook_base_url": webhook_base_url,
        "model_tag": model_tag,
        "ft_type": ft_type,
        "type": "PRIVATE_FINETUNING"
    }


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

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Path to store the configs before job submission"
    )

    # Parse arguments
    args = parser.parse_args()

    finetune_config_path = args.finetune_config
    training_config = _read_yaml_file(finetune_config_path)

    is_lora = "lora_config" in training_config

    model_tag = generate_model_tag(training_config["model_id"])

    if is_lora:
        lora_storage_uri = _get_lora_storage_uri()
        # Required for registering the model on Anyscale.
        training_config.update({
            "forward_checkpoint_config": {
                "tag": model_tag,
                "remote_uri": lora_storage_uri
            }
        })
    else:
        lora_storage_uri = None

    webhook_config = _get_webhook_config(model_tag, is_lora)
    training_config.update({
        "webhook_config": webhook_config
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    finetune_config_path_obj = Path(finetune_config_path)
    filename = finetune_config_path_obj.stem
    suffix = finetune_config_path_obj.suffix

    final_filename = f"{filename}_{timestamp}{suffix}"
    llmforge_config_path = output_dir / final_filename
    entrypoint = f"llmforge dev finetune {llmforge_config_path}"


    with open(llmforge_config_path, "w") as f:
        yaml.safe_dump(training_config, f)

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        entrypoint = f"WANDB_API_KEY={api_key} {entrypoint}"

    subprocess.run(entrypoint, check=True, shell=True)
    if lora_storage_uri:
        print(
            f"Note: LoRA weights will also be stored in path {lora_storage_uri} under {model_tag} bucket."
        )


if __name__ == "__main__":
    main()
