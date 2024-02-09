import sys
import os
import subprocess
import tempfile
from pathlib import Path

import yaml


def read_yaml_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train.py job_config.yaml train_config.yaml")
        sys.exit(1)
    job_config_path = sys.argv[1]
    finetune_config_path = sys.argv[2]

    job_config = read_yaml_file(job_config_path)

    entrypoint = "llmforge dev finetune " + finetune_config_path

    job_config["entrypoint"] = entrypoint
    job_config["name"] = Path(finetune_config_path).stem

    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        job_config["runtime_env"]["env_vars"]["WANDB_API_KEY"] = api_key

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=False, dir=".", suffix=".yaml"
    ) as temp_file:
        yaml.safe_dump(job_config, temp_file)
        temp_file_name = temp_file.name

    # Call `anyscale job submit` on the temporary YAML file
    try:
        subprocess.run(["anyscale", "job", "submit", temp_file_name], check=True)
    finally:
        # Clean up by deleting the temporary file
        os.remove(temp_file_name)


main()
