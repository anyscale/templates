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

def sanity_check_program(model, program, item):
    with dspy.context(lm=model):
        sample_input = item
        print(f"Program input: {sample_input}")
        print(f"Program output label: {program(**sample_input.inputs()).label}")

def get_llama_lms_from_model_names(model_names):
    llama_1b = dspy.LM(model="openai/meta-llama/Llama-3.2-1B-Instruct", **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS)
    finetuned_llamas_1b = {f: dspy.LM(model="openai/" + f, **LOCAL_API_PARAMETERS, **MODEL_PARAMETERS) for f in model_names}
    all_llamas = {**finetuned_llamas_1b, "base": llama_1b}
    return all_llamas

def load_finetuning_kwargs():
    with open("src/finetuning_kwargs.yaml", "r") as f:
        return yaml.safe_load(f)

def get_serve_and_model_config(serve_config_path):
    with open(serve_config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(f"{serve_config_path}:")
    print(yaml.dump(config))
    print("="*50)

    model_config_path = config["applications"][0]["args"]["llm_configs"][0]

    with open(model_config_path, 'r') as file:
        model_config = yaml.safe_load(file)

    print("model_config:")
    print(yaml.dump(model_config))
    return config, model_config
