import dspy
import dsp
import os
import yaml
import ray
import litellm
import warnings

from src.constants import LOCAL_API_PARAMETERS, MODEL_PARAMETERS

litellm.set_verbose=False
litellm.suppress_debug_info=True

warnings.filterwarnings("ignore", category=FutureWarning)

# We pass in the env variables so that they are set for ray workers
def init_ray():
    ray.init(runtime_env={"env_vars": {'HF_TOKEN': os.environ['HF_TOKEN'], "HF_HOME": os.environ["HF_HOME"]}, "py_modules": [dspy, dsp]})

def set_dspy_cache_location(local_cache_dir=None):
    cache_dir = local_cache_dir if local_cache_dir is not None else "/home/ray/default/dspy/cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    os.environ["DSP_CACHEDIR"] = cache_dir

def check_env_vars():
    necessary_env_vars = [
        "HF_TOKEN",
        "HF_HOME"
    ]

    for var in necessary_env_vars:
        assert os.environ[var], f"{var} is not set"

def print_serve_and_model_config(serve_config_path):
    from rich import print as rprint

    with open(serve_config_path, 'r') as file:
        config = yaml.safe_load(file)

    rprint("serve_config:")
    rprint(config)
    rprint("="*50)

    model_config_path = config["applications"][0]["args"]["llm_configs"][0]

    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    hf_token = model_config.get("runtime_env", {}).get("env_vars", {}).get("HUGGING_FACE_HUB_TOKEN")
    if hf_token and hf_token.startswith("hf_"):
        model_config["runtime_env"]["env_vars"]["HUGGING_FACE_HUB_TOKEN"] = "REDACTED"

    rprint("model_config:")
    rprint(model_config)
