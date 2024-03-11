import argparse
import os
import subprocess
import tempfile
import yaml
import random
import string
from pathlib import Path


def _read_yaml_file(file_path):
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


def get_cld_id() -> str:
    return os.environ.get("ANYSCALE_CLOUD_ID") or ""


def get_region() -> str:
    return os.environ.get("ANYSCALE_CLOUD_STORAGE_BUCKET_REGION") or ""


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
        username = username.strip().replace(" ","_")[:5]
        while len(username) < 5:
            username += username[-1]
    else:
        username = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(5))
    return f"{model_id}:{username}:{suffix}"


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Submit a job with configuration files"
    )
    parser.add_argument(
        "job_config", type=str, help="Path to the job configuration YAML file"
    )
    parser.add_argument(
        "finetune_config",
        type=str,
        help="Path to the fine-tuning configuration YAML file",
    )

    # Parse arguments
    args = parser.parse_args()

    job_config_path = args.job_config
    finetune_config_path = args.finetune_config

    job_config = _read_yaml_file(job_config_path)
    training_config = _read_yaml_file(finetune_config_path)

    cld_id = get_cld_id()
    region = get_region()
    job_config["compute_config"]["cloud_id"] = cld_id
    job_config["compute_config"]["region"] = region

    is_lora = "lora_config" in training_config
    entrypoint = f"llmforge dev finetune {finetune_config_path}"

    if is_lora:
        model_tag = generate_model_tag(training_config["model_id"])
        entrypoint += f" --model-tag={model_tag}"
        lora_storage_uri = _get_lora_storage_uri()
        entrypoint += f" --forward-best-checkpoint-remote-uri={lora_storage_uri}"
    else:
        lora_storage_uri = None

    job_config["entrypoint"] = entrypoint
    job_config["name"] = Path(finetune_config_path).stem

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        job_config.setdefault("runtime_env", {}).setdefault("env_vars", {})[
            "WANDB_API_KEY"
        ] = api_key

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, dir=".", suffix=".yaml"
    ) as temp_file:
        yaml.safe_dump(job_config, temp_file)
        temp_file_name = temp_file.name

    # Call `anyscale job submit` on the temporary YAML file
    try:
        subprocess.run(["anyscale", "job", "submit", temp_file_name], check=True)
        if lora_storage_uri:
            print(
                f"Note: LoRA weights will also be stored in path {lora_storage_uri} under {model_tag} bucket."
            )
    finally:
        # Clean up by deleting the temporary file
        os.remove(temp_file_name)
        pass


if __name__ == "__main__":
    main()
