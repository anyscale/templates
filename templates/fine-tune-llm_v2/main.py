import argparse
import os
import subprocess


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

    entrypoint = f"llmforge dev finetune {finetune_config_path}"

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key:
        entrypoint = f"WANDB_API_KEY={api_key} {entrypoint}"

    subprocess.run(entrypoint, check=True, shell=True)


if __name__ == "__main__":
    main()
