import subprocess

import yaml

from utils import (LLAMA_MODELS, MODEL_ID_TO_BASE_CONFIGS, get_cloud_provider,
                   get_gpu_type, get_hf_token, get_model_id,
                   get_tensor_parallelism, populate_configs, read_yaml_file)

_BASE_SERVE_CONFIG_PATH = "models/base_llm_serve.yaml"
CONFIG_FILE = "llm-serve.yaml"

def main():
    cloud_provider = get_cloud_provider()
    model_id = get_model_id()
    gpu_type = get_gpu_type(model_id, cloud_provider)

    config_file_path = MODEL_ID_TO_BASE_CONFIGS[model_id]
    base_config = read_yaml_file(config_file_path)
    tensor_parallelism = get_tensor_parallelism()

    model_configs = populate_configs(base_config, model_id, gpu_type, tensor_parallelism)

    serve_config = read_yaml_file(_BASE_SERVE_CONFIG_PATH)
    serve_config["applications"][0]["args"]["vllm_base_models"] = [model_configs]

    if model_id in LLAMA_MODELS:
        hf_token = get_hf_token()
    else:
        hf_token = None
    if hf_token:
        serve_config["applications"][0]["runtime_env"]["env_vars"]["HUGGING_FACE_HUB_TOKEN"] = hf_token

    with open(CONFIG_FILE, 'w') as file:
        yaml.dump(serve_config, file)
        print(f"\nWe have written the config file to {CONFIG_FILE}.")

    args = ["serve", "run", CONFIG_FILE]

    print(f"\nRunning the command: {' '.join(args)}" )
    print("----------------")
    subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
